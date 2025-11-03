"""Result data models.

Dataclasses for storing OCR predictions and results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class OCRPrediction:
    """Single OCR prediction for a grid cell.

    Attributes:
        row_idx: Row index in the grid.
        col_idx: Column index in the grid.
        col_name: Semantic name of the column (e.g., 'placement', 'character_name').
        text: Predicted text.
        confidence: Confidence score (0-1).
        bounding_box: Bounding box (left, top, width, height).
        passes_validation: Whether prediction passes validation rules.
        meets_threshold: Whether confidence meets threshold.
        is_acceptable: Whether prediction is both valid and confident.
        preprocessing_pipeline: Name of preprocessing pipeline used.
        metadata: Additional metadata.
    """
    row_idx: int
    col_idx: int
    col_name: str
    text: str
    confidence: float
    bounding_box: Dict[str, int]  # {left, top, width, height}
    passes_validation: bool = False
    meets_threshold: bool = False
    is_acceptable: bool = False
    preprocessing_pipeline: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export.

        Returns:
            Dictionary representation.
        """
        # Extract preprocessing and OCR engine configs from metadata
        preprocessing_config = self.metadata.get("preprocessing_config", {})
        ocr_engine_config = self.metadata.get("ocr_engine_config", {})

        # Combine both configs into a single runtime_config JSON string
        runtime_config = {
            "preprocessing_config": preprocessing_config,
            "ocr_engine_config": ocr_engine_config
        }

        # Create the base dictionary
        result = {
            "row_idx": self.row_idx,
            "col_idx": self.col_idx,
            "col_name": self.col_name,
            "text": self.text,
            "confidence": self.confidence,
            "left": self.bounding_box["left"],
            "top": self.bounding_box["top"],
            "width": self.bounding_box["width"],
            "height": self.bounding_box["height"],
            "passes_validation": self.passes_validation,
            "meets_threshold": self.meets_threshold,
            "is_acceptable": self.is_acceptable,
            "runtime_config": json.dumps(runtime_config)
        }

        # Add any other metadata fields (excluding the config fields we already handled)
        for key, value in self.metadata.items():
            if key not in ["preprocessing_config", "ocr_engine_config"]:
                result[key] = value

        return result


@dataclass
class CellResult:
    """Results for a single grid cell across all pipeline attempts.

    Attributes:
        row_idx: Row index.
        col_idx: Column index.
        col_name: Column name.
        predictions: All predictions for this cell (across different pipelines).
        best_prediction: The best acceptable prediction, or None.
        attempts: Number of pipeline attempts made.
    """
    row_idx: int
    col_idx: int
    col_name: str
    predictions: List[OCRPrediction] = field(default_factory=list)
    best_prediction: Optional[OCRPrediction] = None
    attempts: int = 0

    def add_prediction(self, prediction: OCRPrediction) -> bool:
        """Add a prediction and update best if acceptable.

        Args:
            prediction: OCR prediction to add.

        Returns:
            True if this prediction is acceptable and we can stop trying.
        """
        self.predictions.append(prediction)
        self.attempts += 1

        if prediction.is_acceptable and self.best_prediction is None:
            self.best_prediction = prediction
            return True

        return False


@dataclass
class ImageResult:
    """Complete results for a single image.

    Attributes:
        image_path: Path to the source image.
        timestamp: Processing timestamp.
        cell_results: Results for each grid cell.
        total_cells: Total number of cells processed.
        successful_cells: Number of cells with acceptable predictions.
        processing_time_seconds: Total processing time.
    """
    image_path: str
    timestamp: str
    cell_results: List[CellResult] = field(default_factory=list)
    total_cells: int = 0
    successful_cells: int = 0
    processing_time_seconds: float = 0.0

    def get_all_predictions(self) -> List[OCRPrediction]:
        """Get all predictions from all cells.

        Returns:
            Flattened list of all predictions.
        """
        predictions = []
        for cell_result in self.cell_results:
            predictions.extend(cell_result.predictions)
        return predictions

    def get_acceptable_predictions(self) -> List[OCRPrediction]:
        """Get only acceptable predictions.

        Returns:
            List of acceptable predictions.
        """
        predictions = []
        for cell_result in self.cell_results:
            if cell_result.best_prediction:
                predictions.append(cell_result.best_prediction)
        return predictions

    def get_success_rate(self) -> float:
        """Calculate success rate.

        Returns:
            Fraction of cells with acceptable predictions (0-1).
        """
        if self.total_cells == 0:
            return 0.0
        return self.successful_cells / self.total_cells
