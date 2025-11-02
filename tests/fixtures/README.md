# Test Fixtures

This directory contains sample test data files used across the test suite.

## Files

### sample_predictions.csv
Sample OCR predictions with metadata including:
- Cell location (row_idx, col_idx)
- Column name
- Recognized text
- Confidence scores
- Bounding box coordinates
- Validation flags

Used by: `test_data_service.py`, `test_preprocessing.py`

### sample_scores.csv
Sample Mario Kart race score data with columns:
- placement: Final race position (1-12)
- track_name: Name of the race track
- character_name: Character name used
- score: Points earned

Used by: Integration tests

### sample_characters.csv
Sample character roster with:
- name: Character name
- weight: Weight category (very light, light, medium, medium heavy, heavy, very heavy)
- description: Character description

Used by: `test_validation.py`, validation tests

### pipeline_config.json
Example pipeline configuration defining:
- Grid boundaries and column mappings
- Preprocessing pipelines with multiple steps
- OCR engine configurations with confidence thresholds
- Processing parameters

Used by: `test_pipeline.py`, integration tests

### invalid_predictions.csv
Sample predictions with invalid data for testing validation:
- Out-of-range placements (0, 13)
- Invalid character names
- Low confidence scores
- Validation failures

Used by: `test_validation.py`, error handling tests

## Usage in Tests

These fixture files can be:
1. **Loaded directly** in test functions for testing data processing
2. **Referenced** by the fixture functions defined in test files
3. **Extended** with additional test cases as needed

Example usage:
```python
import pandas as pd
df = pd.read_csv("tests/fixtures/sample_predictions.csv")
```
