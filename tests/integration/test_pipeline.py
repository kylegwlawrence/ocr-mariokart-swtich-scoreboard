"""Integration tests for the complete pipeline."""

import pytest
import json
import numpy as np
import cv2
from pathlib import Path
import tempfile

from src.services.orchestrator import PipelineOrchestrator
from src.models.config import PipelineConfig


@pytest.fixture
def sample_config():
    """Create a minimal test configuration."""
    config_dict = {
        "grid": {
            "row_bounds": [[0, 0.5], [0.5, 1.0]],
            "column_bounds": [[0, 0.33], [0.33, 0.66], [0.66, 1.0]],
            "target_columns": [0, 1, 2],
            "column_names": {"0": "placement", "1": "character_name", "2": "score"}
        },
        "preprocessing_pipelines": [
            {
                "name": "simple_grayscale",
                "priority": 0,
                "steps": [{"method": "grayscale", "params": {}}]
            }
        ],
        "ocr_engines": [
            {
                "engine": "tesseract",
                "confidence_thresholds": {
                    "placement": 0.5,
                    "character_name": 0.5,
                    "score": 0.5
                },
                "engine_params": {
                    "psm_modes": {
                        "placement": 6,
                        "character_name": 6,
                        "score": 6
                    }
                }
            }
        ],
        "max_retries_per_cell": 1,
        "output_dir": "test_outputs",
        "save_intermediate": False
    }
    return PipelineConfig.from_dict(config_dict)


@pytest.fixture
def test_image():
    """Create a simple test image with text."""
    # Create a white background
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Add some text
    cv2.putText(img, "1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "Mario", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "45", (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    cv2.putText(img, "2", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "Luigi", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "38", (450, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    return img


class TestPipelineIntegration:
    """Integration tests for the complete OCR pipeline."""

    def test_config_loading(self):
        """Test loading configuration from JSON."""
        config_path = "src/config/default_config.json"
        if Path(config_path).exists():
            config = PipelineConfig.from_dict(
                json.load(open(config_path))
            )
            assert config is not None
            assert len(config.preprocessing_pipelines) > 0
            assert len(config.ocr_engines) > 0

    def test_pipeline_initialization(self, sample_config):
        """Test initializing the pipeline orchestrator."""
        orchestrator = PipelineOrchestrator(sample_config)
        assert orchestrator is not None
        assert len(orchestrator.ocr_services) > 0

    @pytest.mark.skipif(
        not Path("game_images/inputs/pngs").exists(),
        reason="Test images not available"
    )
    def test_process_real_image(self, sample_config):
        """Test processing a real Mario Kart screenshot (if available)."""
        # Find first PNG image
        test_image_dir = Path("game_images/inputs/pngs")
        images = list(test_image_dir.glob("*.png"))

        if not images:
            pytest.skip("No test images found")

        test_image_path = str(images[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config to use temp directory
            sample_config.output_dir = tmpdir

            # Create orchestrator
            orchestrator = PipelineOrchestrator(sample_config)

            # Process image
            result = orchestrator.process_image(
                test_image_path,
                output_dir=tmpdir,
                save_grid=True,
                save_annotated=True
            )

            # Verify result structure
            assert result is not None
            assert result.total_cells > 0
            assert result.processing_time_seconds > 0

            # Check that some outputs were created
            outputs = list(Path(tmpdir).rglob("*"))
            assert len(outputs) > 0

    def test_end_to_end_with_synthetic_image(self, sample_config, test_image):
        """Test end-to-end pipeline with a synthetic test image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save test image
            test_image_path = Path(tmpdir) / "test_image.png"
            cv2.imwrite(str(test_image_path), test_image)

            # Update config
            sample_config.output_dir = tmpdir

            # Create orchestrator
            orchestrator = PipelineOrchestrator(sample_config)

            # Process image
            result = orchestrator.process_image(
                str(test_image_path),
                output_dir=tmpdir,
                save_grid=False,
                save_annotated=False
            )

            # Basic assertions
            assert result is not None
            assert result.total_cells == 6  # 2 rows x 3 columns
            assert len(result.cell_results) == 6
