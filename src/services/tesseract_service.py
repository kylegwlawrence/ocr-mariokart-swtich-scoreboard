"""Tesseract OCR service implementation.

OCR service using the Tesseract library for text extraction.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .base_ocr import BaseOCRService
from ..models.config import GridConfig, OCREngineConfig
from ..models.results import OCRPrediction

logger = logging.getLogger(__name__)


class TesseractService(BaseOCRService):
    """OCR service using Tesseract.

    Extracts text from Mario Kart scoreboard images using the Tesseract OCR engine.
    """

    def __init__(self, grid_config: GridConfig, engine_config: OCREngineConfig):
        """Initialize the Tesseract service.

        Args:
            grid_config: Configuration for the scoreboard grid.
            engine_config: OCR engine-specific configuration.
        """
        super().__init__(grid_config, engine_config)

        # Extract PSM (Page Segmentation Mode) configurations
        self.psm_modes = engine_config.engine_params.get("psm_modes", {
            "placement": 6,
            "character_name": 6,
            "score": 6
        })

        # Validate Tesseract is installed
        self._validate_tesseract()

    def _validate_tesseract(self) -> None:
        """Check if Tesseract is installed."""
        try:
            import pytesseract
            from pytesseract import Output
            # Try to get version to ensure tesseract binary is accessible
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except ImportError:
            raise ImportError(
                "pytesseract is not installed. Install it with: pip install pytesseract"
            )
        except Exception as e:
            raise RuntimeError(
                f"Tesseract binary not found. Please install Tesseract OCR: {e}"
            )

    def _get_tesseract_config(self, col_name: str) -> str:
        """Get Tesseract configuration string for a column type.

        Args:
            col_name: Column name ('placement', 'character_name', 'score').

        Returns:
            Tesseract config string.
        """
        psm = self.psm_modes.get(col_name)

        if col_name in ["placement", "score"]:
            # Digits only
            return f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
        elif col_name == "character_name":
            # Letters and spaces only
            return f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        else:
            # Default: no whitelist
            return f'--oem 3 --psm {psm}'

    def extract_text_from_cell(
        self,
        cell_image: np.ndarray,
        col_idx: int,
        col_name: str
    ) -> List[OCRPrediction]:
        """Extract text from a single grid cell using Tesseract.

        Args:
            cell_image: Cropped image of the grid cell.
            col_idx: Column index.
            col_name: Column name (e.g., 'placement', 'character_name').

        Returns:
            List of OCR predictions for this cell.
        """
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            logger.error("pytesseract not installed")
            return []

        # Get Tesseract config
        config = self._get_tesseract_config(col_name)

        # Run OCR
        try:
            data = pytesseract.image_to_data(
                cell_image,
                config=config,
                output_type=Output.DICT
            )
        except Exception as e:
            logger.error(f"Tesseract failed on cell ({col_idx}, {col_name}): {e}")
            return []

        # Parse results
        predictions = []
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])

            # Skip empty text or low confidence (-1 means no detection)
            if text == "" or conf < 0:
                continue

            # Normalize confidence to 0-1 range (Tesseract uses 0-100)
            conf_normalized = conf / 100.0

            # Create prediction
            prediction = OCRPrediction(
                row_idx=-1,  # Will be set by caller
                col_idx=col_idx,
                col_name=col_name,
                text=text,
                confidence=conf_normalized,
                bounding_box={
                    "left": data['left'][i],
                    "top": data['top'][i],
                    "width": data['width'][i],
                    "height": data['height'][i]
                }
            )

            predictions.append(prediction)

        logger.debug(f"Tesseract found {len(predictions)} predictions in cell ({col_idx}, {col_name})")
        return predictions

    def get_engine_name(self) -> str:
        """Get the name of this OCR engine.

        Returns:
            Engine name.
        """
        return "tesseract"
