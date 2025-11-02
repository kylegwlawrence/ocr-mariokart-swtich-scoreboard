# Migration Guide: v1 to v2

This guide helps you migrate from the old monolithic implementation to the new modular architecture.

## Overview of Changes

### Old Structure
```
mario_kart_scores/
├── main.py                    # Monolithic script
├── preprocess_images.py       # Preprocessing class
├── svc_easyocr.py             # EasyOCR service
├── svc_tesseract.py           # Partial Tesseract implementation
├── ocr_search_rules.py        # Validation functions
├── clean_prediction_data.py   # Data cleaning
└── convert_png_to_heic.py     # Image conversion
```

### New Structure
```
mario_kart_scores/
├── main.py                # New CLI entry point
└── src/
    ├── services/              # All services (modular)
    ├── models/                # Data models & configs
    ├── utils/                 # Utilities
    └── config/                # JSON configurations
```

## Key Differences

### 1. Configuration
**Old:** Hardcoded parameters in main.py
```python
preprocess_params = {
    "dilate_kernel": (4,4),
    "iterations_dilate": 1,
    # ... more params
}
```

**New:** JSON configuration files
```json
{
  "preprocessing_pipelines": [
    {
      "name": "edge_detection",
      "steps": [
        {"method": "dilate", "params": {"kernel": [4, 4], "iterations": 1}}
      ]
    }
  ]
}
```

### 2. Running the Pipeline
**Old:**
```bash
python main.py  # Hardcoded image path and params
```

**New:**
```bash
python main.py --image path/to/image.png --config src/config/default_config.json
```

### 3. Imports
**Old:**
```python
from preprocess import PreprocessPipeline
from svc_easyocr import StructuredTableEasyOCR
```

**New:**
```python
from src.services import PreprocessingService, EasyOCRService
from src.services.orchestrator import PipelineOrchestrator
from src.models.config import PipelineConfig
```

### 4. Processing Images
**Old:**
```python
# Manual pipeline setup
preprocesser = PreprocessPipeline(input_path, output_path, pipeline)
preprocesser.run(params)

ocr = StructuredTableEasyOCR()
df = ocr.extract_text(preprocessed_path, csv_path, ...)
```

**New:**
```python
# Automatic orchestration
config = PipelineConfig.from_dict(json.load(open("config.json")))
orchestrator = PipelineOrchestrator(config)
result = orchestrator.process_image(image_path)
```

## Migration Steps

### Step 1: Understand Your Current Workflow

1. Identify which preprocessing parameters you were using
2. Note which OCR engine you were using (EasyOCR or Tesseract)
3. Check if you had custom validation rules

### Step 2: Create a Configuration File

Create a JSON file with your settings:

```json
{
  "grid": {
    "row_bounds": [...],  // Copy from old code
    "column_bounds": [...],
    "target_columns": [1, 3, 5]
  },
  "preprocessing_pipelines": [
    {
      "name": "my_pipeline",
      "priority": 0,
      "steps": [
        // Convert your old preprocessing steps
        {"method": "grayscale", "params": {}},
        {"method": "dilate", "params": {"kernel": [4, 4], "iterations": 1}}
      ]
    }
  ],
  "ocr_engines": [
    {
      "engine": "easyocr",  // or "tesseract"
      "confidence_thresholds": {
        "placement": 0.6,
        "character_name": 0.5,
        "score": 0.7
      }
    }
  ]
}
```

### Step 3: Update Your Scripts

**Old script:**
```python
# Your old script
from preprocess import PreprocessPipeline
from svc_easyocr import StructuredTableEasyOCR

input_path = "image.png"
pipeline = ["greyscale", "gaussian_blur", "edge_detection"]
params = {...}

preprocesser = PreprocessPipeline(input_path, output_path, pipeline)
preprocesser.run(params)

ocr = StructuredTableEasyOCR()
df = ocr.extract_text(preprocessed_path, csv_path, pipeline, params, ...)
```

**New script:**
```python
import json
from src.services.orchestrator import PipelineOrchestrator
from src.models.config import PipelineConfig

# Load configuration
with open("my_config.json") as f:
    config = PipelineConfig.from_dict(json.load(f))

# Process image
orchestrator = PipelineOrchestrator(config)
result = orchestrator.process_image("image.png")

# Save results
from src.services.data_service import DataService
data_service = DataService()
data_service.save_image_result(result, "output.csv")
```

### Step 4: Use the CLI

Instead of writing scripts, use the CLI:

```bash
python main.py --image image.png --config my_config.json
```

## Parameter Mapping

### Preprocessing Methods
Old name → New name:
- `greyscale` → `grayscale`
- All other names remain the same

### Preprocessing Parameters
Old format → New format:

```python
# Old
preprocess_params = {
    "dilate_kernel": (4, 4),
    "iterations_dilate": 1,
    "gaussian_blur_kernel": (5, 5),
    "sigmaX": 0
}

# New
{
  "steps": [
    {
      "method": "dilate",
      "params": {"kernel": [4, 4], "iterations": 1}
    },
    {
      "method": "gaussian_blur",
      "params": {"kernel": [5, 5], "sigmaX": 0, "sigmaY": 0}
    }
  ]
}
```

## Backward Compatibility

The old files are still in the `src/` directory but have been refactored:

- `src/preprocess_images.py` → Use `src/services/preprocessing_service.py`
- `src/svc_easyocr.py` → Use `src/services/easyocr_service.py`
- `src/svc_tesseract.py` → Use `src/services/tesseract_service.py`
- `src/ocr_search_rules.py` → Use `src/services/validation_service.py`
- `src/clean_prediction_data.py` → Use `src/services/data_service.py`
- `src/convert_png_to_heic.py` → Use `src/services/conversion_service.py`

## New Features You Can Use

1. **Multiple Preprocessing Pipelines**: The system will automatically try different pipelines if the first one fails
2. **Multiple OCR Engines**: Can fallback from EasyOCR to Tesseract automatically
3. **Batch Processing**: Process entire folders easily
4. **Better Logging**: See exactly what's happening with detailed logs
5. **Validation**: Automatic validation with retry on failure
6. **Testing**: Run tests to verify everything works

## Common Issues

### Import Errors
**Problem:** `ModuleNotFoundError: No module named 'preprocess'`

**Solution:** Update imports to use `src.services.*`

### Config Not Found
**Problem:** `FileNotFoundError: config.json not found`

**Solution:** Use full path or one of the examples: `src/config/default_config.json`

### Character CSV Not Found
**Problem:** `Character CSV not found`

**Solution:** The CSV moved to `data/character_info.csv`. The ValidationService will find it automatically.

## Need Help?

1. Check the [README.md](README.md) for usage examples
2. Look at example configs in `src/config/`
3. Run tests to see working examples: `pytest tests/`
4. Check logs in `logs/` directory for debugging

## Rollback

If you need to use the old code temporarily:

1. The old `main.py` is still there (but imports are broken)
2. Old service files are in `src/` directory
3. You can fix imports manually or use the new system

**Recommended:** Migrate fully to the new system for better maintainability.
