"""Unit tests for DataService."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.services.data_service import DataService
from src.models.results import OCRPrediction


@pytest.fixture
def data_service():
    """Create a DataService instance."""
    return DataService()


@pytest.fixture
def sample_predictions():
    """Load sample OCR predictions from fixture file."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_predictions.csv"
    df = pd.read_csv(fixture_path)

    # Convert DataFrame rows to OCRPrediction objects
    predictions = []
    for _, row in df.iterrows():
        # Reconstruct bounding_box from individual columns
        bounding_box = {
            "left": int(row['left']),
            "top": int(row['top']),
            "width": int(row['width']),
            "height": int(row['height'])
        }

        prediction = OCRPrediction(
            row_idx=int(row['row_idx']),
            col_idx=int(row['col_idx']),
            col_name=row['col_name'],
            text=row['text'],
            confidence=float(row['confidence']),
            bounding_box=bounding_box,
            passes_validation=row['passes_validation'] == True,
            meets_threshold=row['meets_threshold'] == True,
            is_acceptable=row['is_acceptable'] == True,
            preprocessing_pipeline=row['preprocessing_pipeline'],
            ocr_engine=row['ocr_engine']
        )
        predictions.append(prediction)

    return predictions


class TestDataService:
    """Test suite for DataService."""

    def test_save_predictions_to_csv(self, data_service, sample_predictions):
        """Test saving predictions to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_predictions.csv"
            data_service.save_predictions_to_csv(sample_predictions, str(csv_path))

            # Verify file was created
            assert csv_path.exists()

            # Load and verify content
            df = pd.read_csv(csv_path)
            assert len(df) == 6
            assert "row_idx" in df.columns
            assert "col_idx" in df.columns
            assert "col_name" in df.columns
            assert "text" in df.columns

            assert "left" in df.columns
            assert "top" in df.columns
            assert "width" in df.columns
            assert "height" in df.columns

            assert "passes_validation" in df.columns
            assert "meets_threshold" in df.columns
            assert "is_acceptable" in df.columns
            assert "preprocessing_pipeline" in df.columns
            assert "psm" in df.columns
            assert "tesseract_config" in df.columns
            assert "image_path" in df.columns
            assert "timestamp" in df.columns
            

    def test_get_best_predictions_per_cell(self, data_service):
        """Test getting best prediction per cell."""
        # Create multiple predictions for same cell
        predictions = [
            OCRPrediction(
                row_idx=0, col_idx=1, col_name="placement", text="1",
                confidence=0.7, bounding_box={"left": 0, "top": 0, "width": 10, "height": 10}
            ),
            OCRPrediction(
                row_idx=0, col_idx=1, col_name="placement", text="2",
                confidence=0.9, bounding_box={"left": 0, "top": 0, "width": 10, "height": 10}
            )
        ]

        # Convert to DataFrame
        df = pd.DataFrame([p.to_dict() for p in predictions])

        # Get best predictions
        best_df = data_service.get_best_predictions_per_cell(df)

        # Should only have 1 row (highest confidence)
        assert len(best_df) == 1
        assert best_df.iloc[0]["text"] == "2"  # Higher confidence prediction

    def test_merge_csv_files(self, data_service, sample_predictions):
        """Test merging multiple CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create two CSV files
            csv1 = tmppath / "file1.csv"
            csv2 = tmppath / "file2.csv"

            data_service.save_predictions_to_csv([sample_predictions[0]], str(csv1))
            data_service.save_predictions_to_csv([sample_predictions[1]], str(csv2))

            # Merge them
            merged_path = tmppath / "merged.csv"
            merged_df = data_service.merge_csv_files(str(tmppath), str(merged_path))

            # Verify
            assert len(merged_df) == 2
            assert merged_path.exists()
            assert "source_file" in merged_df.columns
