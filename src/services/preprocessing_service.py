"""Preprocessing service for image preparation.

Applies various image preprocessing techniques to improve OCR accuracy.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from ..models.config import PreprocessingConfig, PreprocessingStep

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service for applying image preprocessing pipelines.

    Supports various OpenCV preprocessing methods including:
    - Grayscale conversion
    - Gaussian blur
    - Edge detection (Canny)
    - Morphological operations (dilate, erode)
    - Thresholding (binary, adaptive)
    - Contrast adjustment
    - And more
    """

    def __init__(self):
        """Initialize the preprocessing service."""
        self.supported_methods = {
            "grayscale",
            "gaussian_blur",
            "edge_detection",
            "dilate",
            "erode",
            "threshold",
            "adaptive_threshold",
            "inversion",
            "morphology",
            "blur",
            "contrast",
            "median_blur",
            "bilateral_filter",
        }

    def apply_pipeline(
        self,
        image: np.ndarray,
        config: PreprocessingConfig,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Apply a complete preprocessing pipeline to an image.

        Args:
            image: Input image (BGR or grayscale).
            config: Preprocessing configuration.
            save_path: Optional path to save the processed image.

        Returns:
            Processed image.

        Raises:
            ValueError: If an unsupported preprocessing method is specified.
        """
        result = image.copy()

        logger.debug(f"Applying preprocessing pipeline: {config.name}")

        for step in config.steps:
            if step.method not in self.supported_methods:
                raise ValueError(
                    f"Unsupported preprocessing method: {step.method}. "
                    f"Supported: {self.supported_methods}"
                )

            result = self._apply_step(result, step)
            logger.debug(f"Applied step: {step.method}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, result)
            logger.debug(f"Saved preprocessed image to: {save_path}")

        return result

    def _apply_step(self, image: np.ndarray, step: PreprocessingStep) -> np.ndarray:
        """Apply a single preprocessing step.

        Args:
            image: Input image.
            step: Preprocessing step configuration.

        Returns:
            Processed image.
        """
        method = step.method
        params = step.params

        if method == "grayscale":
            return self._grayscale(image)
        elif method == "gaussian_blur":
            return self._gaussian_blur(image, **params)
        elif method == "edge_detection":
            return self._edge_detection(image, **params)
        elif method == "dilate":
            return self._dilate(image, **params)
        elif method == "erode":
            return self._erode(image, **params)
        elif method == "threshold":
            return self._threshold(image, **params)
        elif method == "adaptive_threshold":
            return self._adaptive_threshold(image, **params)
        elif method == "inversion":
            return self._inversion(image)
        elif method == "morphology":
            return self._morphology(image, **params)
        elif method == "blur":
            return self._blur(image, **params)
        elif method == "contrast":
            return self._contrast(image, **params)
        elif method == "median_blur":
            return self._median_blur(image, **params)
        elif method == "bilateral_filter":
            return self._bilateral_filter(image, **params)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    def _grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _gaussian_blur(
        self,
        image: np.ndarray,
        kernel: tuple = (5, 5),
        sigmaX: int = 0,
        sigmaY: int = 0
    ) -> np.ndarray:
        """Apply Gaussian blur.

        Args:
            image: Input image.
            kernel: Kernel size (must be odd).
            sigmaX: Gaussian kernel standard deviation in X direction.
            sigmaY: Gaussian kernel standard deviation in Y direction.

        Returns:
            Blurred image.
        """
        # Convert list to tuple if needed
        if isinstance(kernel, list):
            kernel = tuple(kernel)

        # Validate kernel size is odd
        if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
            raise ValueError(f"Kernel size must be odd, got {kernel}")

        return cv2.GaussianBlur(image, kernel, sigmaX, sigmaY)

    def _edge_detection(
        self,
        image: np.ndarray,
        hysteresis_min: int = 100,
        hysteresis_max: int = 200
    ) -> np.ndarray:
        """Apply Canny edge detection.

        Args:
            image: Input image (should be grayscale).
            hysteresis_min: Lower threshold for hysteresis.
            hysteresis_max: Upper threshold for hysteresis.

        Returns:
            Edge-detected image.
        """
        return cv2.Canny(image, hysteresis_min, hysteresis_max)

    def _dilate(
        self,
        image: np.ndarray,
        kernel: tuple = (3, 3),
        iterations: int = 1
    ) -> np.ndarray:
        """Apply dilation morphological operation.

        Args:
            image: Input image.
            kernel: Kernel size.
            iterations: Number of iterations.

        Returns:
            Dilated image.
        """
        if isinstance(kernel, list):
            kernel = tuple(kernel)
        kernel_array = np.ones(kernel, np.uint8)
        return cv2.dilate(image, kernel_array, iterations=iterations)

    def _erode(
        self,
        image: np.ndarray,
        kernel: tuple = (3, 3),
        iterations: int = 1
    ) -> np.ndarray:
        """Apply erosion morphological operation.

        Args:
            image: Input image.
            kernel: Kernel size.
            iterations: Number of iterations.

        Returns:
            Eroded image.
        """
        if isinstance(kernel, list):
            kernel = tuple(kernel)
        kernel_array = np.ones(kernel, np.uint8)
        return cv2.erode(image, kernel_array, iterations=iterations)

    def _threshold(
        self,
        image: np.ndarray,
        threshold: int = 200,
        max_value: int = 255
    ) -> np.ndarray:
        """Apply binary thresholding.

        Args:
            image: Input image (grayscale).
            threshold: Threshold value.
            max_value: Maximum value for thresholded pixels.

        Returns:
            Thresholded image.
        """
        _, result = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
        return result

    def _adaptive_threshold(
        self,
        image: np.ndarray,
        max_value: int = 255,
        block_size: int = 15,
        C: int = -2
    ) -> np.ndarray:
        """Apply adaptive thresholding.

        Args:
            image: Input image (grayscale).
            max_value: Maximum value.
            block_size: Size of pixel neighborhood.
            C: Constant subtracted from mean.

        Returns:
            Adaptively thresholded image.
        """
        return cv2.adaptiveThreshold(
            image, max_value,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size, C
        )

    def _inversion(self, image: np.ndarray) -> np.ndarray:
        """Invert image colors."""
        return cv2.bitwise_not(image)

    def _morphology(
        self,
        image: np.ndarray,
        operation: str = "open",
        kernel: tuple = (2, 2)
    ) -> np.ndarray:
        """Apply morphological operation.

        Args:
            image: Input image.
            operation: Morphological operation ('open', 'close', 'gradient', etc.).
            kernel: Kernel size.

        Returns:
            Processed image.
        """
        if isinstance(kernel, list):
            kernel = tuple(kernel)

        kernel_array = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)

        op_map = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }

        morph_op = op_map.get(operation, cv2.MORPH_OPEN)
        return cv2.morphologyEx(image, morph_op, kernel_array)

    def _blur(self, image: np.ndarray, kernel: tuple = (5, 5)) -> np.ndarray:
        """Apply basic blur.

        Args:
            image: Input image.
            kernel: Kernel size.

        Returns:
            Blurred image.
        """
        if isinstance(kernel, list):
            kernel = tuple(kernel)
        return cv2.blur(image, kernel)

    def _contrast(
        self,
        image: np.ndarray,
        alpha: float = 1.0,
        beta: int = 0
    ) -> np.ndarray:
        """Adjust contrast and brightness.

        Args:
            image: Input image.
            alpha: Contrast control (1.0-3.0).
            beta: Brightness control (0-100).

        Returns:
            Adjusted image.
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _median_blur(self, image: np.ndarray, ksize: int = 5) -> np.ndarray:
        """Apply median blur.

        Args:
            image: Input image.
            ksize: Kernel size (must be odd).

        Returns:
            Blurred image.
        """
        return cv2.medianBlur(image, ksize)

    def _bilateral_filter(
        self,
        image: np.ndarray,
        d: int = 9,
        sigmaColor: int = 75,
        sigmaSpace: int = 75
    ) -> np.ndarray:
        """Apply bilateral filter.

        Args:
            image: Input image.
            d: Diameter of pixel neighborhood.
            sigmaColor: Filter sigma in color space.
            sigmaSpace: Filter sigma in coordinate space.

        Returns:
            Filtered image.
        """
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
