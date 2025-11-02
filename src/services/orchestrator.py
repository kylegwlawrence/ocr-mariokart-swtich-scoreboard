"""Orchestration service for the complete OCR pipeline.

Coordinates preprocessing, OCR, validation, and result aggregation.
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from .preprocessing_service import PreprocessingService
from .validation_service import ValidationService
from .easyocr_service import EasyOCRService
from .tesseract_service import TesseractService
from .data_service import DataService
from .base_ocr import BaseOCRService
from ..models.config import PipelineConfig, OCREngine, PreprocessingConfig
from ..models.results import OCRPrediction, CellResult, ImageResult

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete OCR pipeline.

    Manages the flow from image loading through preprocessing, OCR,
    validation, and result aggregation with retry logic.
    """

    def __init__(self, config: PipelineConfig, character_csv_path: str = "data/character_info.csv"):
        """Initialize the orchestrator.

        Args:
            config: Complete pipeline configuration.
            character_csv_path: Path to character names CSV.
        """
        self.config = config
        self.preprocessing_service = PreprocessingService()
        self.validation_service = ValidationService(character_csv_path)
        self.data_service = DataService()

        # Initialize OCR services based on config
        self.ocr_services: Dict[str, BaseOCRService] = {}
        self._initialize_ocr_services()

    def _initialize_ocr_services(self) -> None:
        """Initialize OCR service instances from configuration."""
        for engine_config in self.config.ocr_engines:
            engine_name = engine_config.engine.value

            if engine_config.engine == OCREngine.EASYOCR:
                service = EasyOCRService(self.config.grid, engine_config)
            elif engine_config.engine == OCREngine.TESSERACT:
                service = TesseractService(self.config.grid, engine_config)
            else:
                logger.warning(f"Unsupported OCR engine: {engine_name}")
                continue

            self.ocr_services[engine_name] = service
            logger.info(f"Initialized OCR service: {engine_name}")

    def process_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_grid: bool = True,
        save_annotated: bool = True
    ) -> ImageResult:
        """Process a single image through the complete pipeline.

        Args:
            image_path: Path to input image.
            output_dir: Directory for outputs (uses config default if None).
            save_grid: Whether to save grid overlay visualization.
            save_annotated: Whether to save annotated image.

        Returns:
            ImageResult with all predictions and metadata.
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if output_dir is None:
            output_dir = self.config.output_dir

        logger.info(f"Processing image: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Initialize result
        result = ImageResult(
            image_path=image_path,
            timestamp=timestamp
        )

        # Save grid overlay if requested
        if save_grid and self.ocr_services:
            grid_path = self._get_output_path(image_path, output_dir, "grid", timestamp)
            first_service = next(iter(self.ocr_services.values()))
            first_service.draw_grid_overlay(image_path, grid_path)

        # Extract all cells from the image (using original/first preprocessing)
        first_service = next(iter(self.ocr_services.values()))
        cells = first_service.extract_cells_from_image(image)

        result.total_cells = len(cells)
        logger.info(f"Extracted {len(cells)} cells to process")

        # Process each cell with retry logic
        for row_idx, col_idx, col_name, _ in cells:
            cell_result = self._process_cell_with_retry(
                image, row_idx, col_idx, col_name, image_path, timestamp, output_dir
            )
            result.cell_results.append(cell_result)

            if cell_result.best_prediction:
                result.successful_cells += 1

        # Calculate processing time
        result.processing_time_seconds = time.time() - start_time

        logger.info(
            f"Processed {result.successful_cells}/{result.total_cells} cells successfully "
            f"({result.get_success_rate() * 100:.1f}%) in {result.processing_time_seconds:.2f}s"
        )

        # Save annotated image if requested
        if save_annotated and result.get_acceptable_predictions():
            annotated_path = self._get_output_path(image_path, output_dir, "annotated", timestamp)
            first_service.annotate_results(
                image_path,
                result.get_acceptable_predictions(),
                annotated_path
            )

        return result

    def _process_cell_with_retry(
        self,
        original_image: np.ndarray,
        row_idx: int,
        col_idx: int,
        col_name: str,
        image_path: str,
        timestamp: str,
        output_dir: str
    ) -> CellResult:
        """Process a single cell with retry logic across pipelines and engines.

        Args:
            original_image: Original full image.
            row_idx: Row index of the cell.
            col_idx: Column index of the cell.
            col_name: Column name (e.g., 'placement').
            image_path: Path to original image (for metadata).
            timestamp: Processing timestamp.
            output_dir: Output directory.

        Returns:
            CellResult with all attempts and best prediction.
        """
        cell_result = CellResult(row_idx=row_idx, col_idx=col_idx, col_name=col_name)

        # Try each preprocessing pipeline
        for pipeline_idx, preprocessing_config in enumerate(self.config.preprocessing_pipelines):
            if cell_result.best_prediction is not None:
                break  # Found acceptable prediction, stop trying

            if cell_result.attempts >= self.config.max_retries_per_cell:
                break  # Max retries reached

            # Apply preprocessing to full image
            preprocessed_image = self.preprocessing_service.apply_pipeline(
                original_image,
                preprocessing_config,
                save_path=self._get_preprocessing_path(
                    image_path, output_dir, preprocessing_config.name, timestamp
                ) if self.config.save_intermediate else None
            )

            # Extract this specific cell from preprocessed image
            cell_image = self._extract_cell(preprocessed_image, row_idx, col_idx)

            # Try each OCR engine
            for ocr_service in self.ocr_services.values():
                if cell_result.best_prediction is not None:
                    break

                # Extract text
                predictions = ocr_service.extract_text_from_cell(
                    cell_image, col_idx, col_name
                )

                # Validate and enrich each prediction
                for pred in predictions:
                    pred.row_idx = row_idx
                    pred.preprocessing_pipeline = preprocessing_config.name
                    pred.metadata['image_path'] = image_path
                    pred.metadata['timestamp'] = timestamp

                    # Get confidence threshold for this column
                    threshold = ocr_service.engine_config.confidence_thresholds.get(
                        col_name, 0.5
                    )

                    # Validate
                    pred.passes_validation = self.validation_service.validate(
                        pred.text, col_name
                    )
                    pred.meets_threshold = pred.confidence >= threshold
                    pred.is_acceptable = pred.passes_validation and pred.meets_threshold

                    # Add to cell result (this updates best_prediction if acceptable)
                    if cell_result.add_prediction(pred):
                        logger.debug(
                            f"Cell ({row_idx}, {col_idx}) solved with "
                            f"{ocr_service.get_engine_name()}/{preprocessing_config.name}: "
                            f"'{pred.text}' (conf={pred.confidence:.2f})"
                        )
                        break  # Stop trying more OCR engines

        # Log if cell couldn't be solved
        if cell_result.best_prediction is None:
            logger.warning(
                f"Cell ({row_idx}, {col_idx}) '{col_name}' could not be solved "
                f"after {cell_result.attempts} attempts"
            )

        return cell_result

    def _extract_cell(
        self,
        image: np.ndarray,
        row_idx: int,
        col_idx: int
    ) -> np.ndarray:
        """Extract a specific cell from an image.

        Args:
            image: Full image.
            row_idx: Row index.
            col_idx: Column index.

        Returns:
            Cropped cell image.
        """
        h, w = image.shape[:2]

        row_start, row_end = self.config.grid.row_bounds[row_idx]
        col_start, col_end = self.config.grid.column_bounds[col_idx]

        y0 = int(row_start * h)
        y1 = int(row_end * h)
        x0 = int(col_start * w)
        x1 = int(col_end * w)

        return image[y0:y1, x0:x1]

    def _get_output_path(
        self,
        image_path: str,
        output_dir: str,
        output_type: str,
        timestamp: str
    ) -> str:
        """Generate output path for a specific output type.

        Args:
            image_path: Original image path.
            output_dir: Base output directory.
            output_type: Type of output ('grid', 'annotated', 'csv', etc.).
            timestamp: Processing timestamp.

        Returns:
            Output file path.
        """
        image_name = Path(image_path).stem
        timestamp_clean = timestamp.replace(" ", "_").replace(":", "-")

        subdir_map = {
            "grid": "grids",
            "annotated": "annotations",
            "csv": "data",
            "preprocessed": "preprocessed"
        }

        subdir = subdir_map.get(output_type, output_type)
        ext_map = {
            "grid": ".jpg",
            "annotated": ".jpg",
            "csv": ".csv",
            "preprocessed": ".jpg"
        }
        ext = ext_map.get(output_type, ".jpg")

        output_path = Path(output_dir) / subdir / f"{image_name}_{timestamp_clean}{ext}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return str(output_path)

    def _get_preprocessing_path(
        self,
        image_path: str,
        output_dir: str,
        pipeline_name: str,
        timestamp: str
    ) -> str:
        """Generate path for preprocessed image.

        Args:
            image_path: Original image path.
            output_dir: Base output directory.
            pipeline_name: Name of preprocessing pipeline.
            timestamp: Processing timestamp.

        Returns:
            Preprocessed image path.
        """
        image_name = Path(image_path).stem
        timestamp_clean = timestamp.replace(" ", "_").replace(":", "-")

        output_path = (
            Path(output_dir) / "preprocessed" /
            f"{image_name}_{pipeline_name}_{timestamp_clean}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return str(output_path)
