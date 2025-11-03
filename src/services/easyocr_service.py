"""EasyOCR service implementation.

OCR service using the EasyOCR library for text extraction.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Any

from .base_ocr import BaseOCRService
from ..models.config import GridConfig, OCREngineConfig
from ..models.results import OCRPrediction

logger = logging.getLogger(__name__)


class EasyOCRService(BaseOCRService):
    """OCR service using EasyOCR.

    Extracts text from Mario Kart scoreboard images using the EasyOCR library.
    """

    def __init__(self, grid_config: GridConfig, engine_config: OCREngineConfig):
        """Initialize the EasyOCR service.

        Args:
            grid_config: Configuration for the scoreboard grid.
            engine_config: OCR engine-specific configuration.
        """
        super().__init__(grid_config, engine_config)

        # Lazy load the reader (only when needed)
        self._reader: Optional[Any] = None

        # Extract engine params
        self.languages = engine_config.engine_params.get("languages", ["en"])
        self.use_gpu = engine_config.engine_params.get("use_gpu", False)

    def _get_reader(self):
        """Lazily initialize the EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
                logger.info(f"Initializing EasyOCR reader (languages={self.languages}, gpu={self.use_gpu})")
                self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except ImportError:
                raise ImportError(
                    "EasyOCR is not installed. Install it with: pip install easyocr"
                )
        return self._reader

    def extract_text_from_cell(
        self,
        cell_image: np.ndarray,
        col_idx: int,
        col_name: str
    ) -> List[OCRPrediction]:
        """Extract text from a single grid cell using EasyOCR.

        Args:
            cell_image: Cropped image of the grid cell.
            col_idx: Column index.
            col_name: Column name (e.g., 'placement', 'character_name').

        Returns:
            List of OCR predictions for this cell.
        """
        reader = self._get_reader()

        # Determine allow/block lists based on column type
        allow_list = None
        block_list = None

        if col_name in ["placement", "score"]:
            # Digits only
            allow_list = "0123456789"
        elif col_name == "character_name":
            # Block digits and special characters, allow letters only
            block_list = """0123456789!"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\"""

        # Run OCR
        try:
            results = reader.readtext(
                cell_image,
                allowlist=allow_list,
                blocklist=block_list
            )
        except Exception as e:
            logger.error(f"EasyOCR failed on cell ({col_idx}, {col_name}): {e}")
            return []

        # Parse results into OCRPrediction objects
        predictions = []
        for res in results:
            box, text, conf = res

            # Skip empty text
            if not text or text.strip() == "":
                continue

            # Calculate bounding box
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            left = int(min(x_coords))
            top = int(min(y_coords))
            width = int(max(x_coords) - left)
            height = int(max(y_coords) - top)

            prediction = OCRPrediction(
                row_idx=-1,  # Will be set by caller
                col_idx=col_idx,
                col_name=col_name,
                text=text.strip(),
                confidence=float(conf),
                bounding_box={
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height
                }
            )

            predictions.append(prediction)

        logger.debug(f"EasyOCR found {len(predictions)} predictions in cell ({col_idx}, {col_name})")
        return predictions

    def get_engine_name(self) -> str:
        """Get the name of this OCR engine.

        Returns:
            Engine name.
        """
        return "easyocr"
