"""Unit tests for PreprocessingService."""

import pytest
import numpy as np
import cv2
from src.services.preprocessing_service import PreprocessingService
from src.models.config import PreprocessingConfig, PreprocessingStep


@pytest.fixture
def preprocessing_service():
    """Create a PreprocessingService instance."""
    return PreprocessingService()


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple 100x100 color image
    return np.ones((100, 100, 3), dtype=np.uint8) * 128


class TestPreprocessingService:
    """Test suite for PreprocessingService."""

    def test_grayscale(self, preprocessing_service, sample_image):
        """Test grayscale conversion."""
        result = preprocessing_service._grayscale(sample_image)
        assert len(result.shape) == 2  # Should be 2D (grayscale)
        assert result.shape == (100, 100)

    def test_gaussian_blur(self, preprocessing_service, sample_image):
        """Test Gaussian blur."""
        result = preprocessing_service._gaussian_blur(sample_image, kernel=(5, 5))
        assert result.shape == sample_image.shape

    def test_gaussian_blur_invalid_kernel(self, preprocessing_service, sample_image):
        """Test that even kernel size raises error."""
        with pytest.raises(ValueError):
            preprocessing_service._gaussian_blur(sample_image, kernel=(4, 4))

    def test_dilate(self, preprocessing_service, sample_image):
        """Test dilation."""
        gray = preprocessing_service._grayscale(sample_image)
        result = preprocessing_service._dilate(gray, kernel=(3, 3), iterations=1)
        assert result.shape == gray.shape

    def test_erode(self, preprocessing_service, sample_image):
        """Test erosion."""
        gray = preprocessing_service._grayscale(sample_image)
        result = preprocessing_service._erode(gray, kernel=(3, 3), iterations=1)
        assert result.shape == gray.shape

    def test_threshold(self, preprocessing_service, sample_image):
        """Test binary thresholding."""
        gray = preprocessing_service._grayscale(sample_image)
        result = preprocessing_service._threshold(gray, threshold=127)
        assert result.shape == gray.shape
        # Check that result only contains 0 or 255
        unique_vals = np.unique(result)
        assert len(unique_vals) <= 2

    def test_apply_pipeline(self, preprocessing_service, sample_image):
        """Test applying a complete pipeline."""
        config = PreprocessingConfig(
            name="test_pipeline",
            steps=[
                PreprocessingStep(method="grayscale", params={}),
                PreprocessingStep(method="gaussian_blur", params={"kernel": [5, 5], "sigmaX": 0, "sigmaY": 0})
            ]
        )

        result = preprocessing_service.apply_pipeline(sample_image, config)
        assert result.shape == (100, 100)  # Should be grayscale

    def test_unsupported_method(self, preprocessing_service, sample_image):
        """Test that unsupported methods raise ValueError."""
        config = PreprocessingConfig(
            name="bad_pipeline",
            steps=[PreprocessingStep(method="invalid_method", params={})]
        )

        with pytest.raises(ValueError):
            preprocessing_service.apply_pipeline(sample_image, config)
