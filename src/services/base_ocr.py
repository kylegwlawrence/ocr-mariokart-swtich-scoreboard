"""Base OCR service interface.

Abstract base class defining the interface for OCR services.
"""

from abc import ABC, abstractmethod
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

from ..models.config import GridConfig, OCREngineConfig
from ..models.results import OCRPrediction

logger = logging.getLogger(__name__)


class BaseOCRService(ABC):
    """Abstract base class for OCR services.

    All OCR implementations (EasyOCR, Tesseract, etc.) should inherit from this
    and implement the required methods.
    """

    def __init__(self, grid_config: GridConfig, engine_config: OCREngineConfig):
        """Initialize the OCR service.

        Args:
            grid_config: Configuration for the scoreboard grid.
            engine_config: OCR engine-specific configuration.
        """
        self.grid_config = grid_config
        self.engine_config = engine_config

    @abstractmethod
    def extract_text_from_cell(
        self,
        cell_image: np.ndarray,
        col_idx: int,
        col_name: str
    ) -> List[OCRPrediction]:
        """Extract text from a single grid cell.

        Args:
            cell_image: Cropped image of the grid cell.
            col_idx: Column index.
            col_name: Column name (e.g., 'placement', 'character_name').

        Returns:
            List of OCR predictions for this cell.
        """
        pass

    def extract_cells_from_image(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int, str, np.ndarray]]:
        """Extract all target cells from an image based on grid configuration.

        Args:
            image: Full preprocessed image.

        Returns:
            List of tuples: (row_idx, col_idx, col_name, cell_image).
        """
        h, w = image.shape[:2]
        cells = []

        for row_idx, (row_start, row_end) in enumerate(self.grid_config.row_bounds):
            y0 = int(row_start * h)
            y1 = int(row_end * h)

            for col_idx in self.grid_config.target_columns:
                col_start, col_end = self.grid_config.column_bounds[col_idx]
                x0 = int(col_start * w)
                x1 = int(col_end * w)

                cell_img = image[y0:y1, x0:x1]
                col_name = self.grid_config.column_names.get(col_idx, f"col_{col_idx}")

                cells.append((row_idx, col_idx, col_name, cell_img))

        logger.debug(f"Extracted {len(cells)} cells from image")
        return cells

    def draw_grid_overlay(
        self,
        image_path: str,
        output_path: str
    ) -> None:
        """Draw grid overlay on image for visualization.

        Args:
            image_path: Path to input image.
            output_path: Path to save annotated image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = image.shape[:2]

        # Draw horizontal lines for rows
        for i, (row_start, row_end) in enumerate(self.grid_config.row_bounds):
            y1 = int(row_start * h)
            y2 = int(row_end * h)
            cv2.line(image, (0, y1), (w, y1), (0, 0, 255), 3)
            cv2.line(image, (0, y2), (w, y2), (0, 0, 255), 3)

            # Label row
            label_y = int((y1 + y2) / 2)
            cv2.putText(
                image, f"Row {i+1}", (10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA
            )

        # Draw vertical lines for columns
        for j, (col_start, col_end) in enumerate(self.grid_config.column_bounds):
            x1 = int(col_start * w)
            x2 = int(col_end * w)
            cv2.line(image, (x1, 0), (x1, h), (0, 0, 255), 3)
            cv2.line(image, (x2, 0), (x2, h), (0, 0, 255), 3)

            # Label column
            label_x = int((x1 + x2) / 2)
            col_name = self.grid_config.column_names.get(j, f"Col {j+1}")
            cv2.putText(
                image, col_name if j in self.grid_config.target_columns else f"Col {j+1}",
                (label_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA
            )

        # Save annotated image
        cv2.imwrite(output_path, image)
        logger.info(f"Grid overlay saved to: {output_path}")

    def annotate_results(
        self,
        image_path: str,
        predictions: List[OCRPrediction],
        output_path: str,
        caption: Optional[str] = None
    ) -> None:
        """Draw bounding boxes and text on image.

        Args:
            image_path: Path to input image.
            predictions: List of OCR predictions.
            output_path: Path to save annotated image.
            caption: Optional caption to add at bottom of image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Draw bounding boxes and text
        for pred in predictions:
            bbox = pred.bounding_box
            x0 = bbox["left"]
            y0 = bbox["top"]
            x1 = x0 + bbox["width"]
            y1 = y0 + bbox["height"]

            # Color code by acceptance
            color = (0, 255, 0) if pred.is_acceptable else (0, 0, 255)

            cv2.rectangle(image, (x0, y0), (x1, y1), color, 4)
            cv2.putText(
                image, pred.text, (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

        # Add caption if provided
        if caption:
            caption_height = 80
            caption_bg = np.full((caption_height, image.shape[1], 3), (255, 255, 255), dtype=np.uint8)

            text_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)[0]
            text_x = int((image.shape[1] - text_size[0]) / 2)
            text_y = int((caption_height + text_size[1]) / 2)

            cv2.putText(
                caption_bg, caption, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA
            )

            image = np.vstack((image, caption_bg))

        # Save annotated image
        cv2.imwrite(output_path, image)
        logger.info(f"Annotated image saved to: {output_path}")

    @abstractmethod
    def get_engine_name(self) -> str:
        """Get the name of this OCR engine.

        Returns:
            Engine name (e.g., 'easyocr', 'tesseract').
        """
        pass
