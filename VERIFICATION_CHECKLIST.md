# Verification Checklist

Use this checklist to verify the refactored Mario Kart OCR system is working correctly.

## ✅ Installation Verification

### 1. Python Environment
```bash
# Check Python version (should be 3.8+)
python --version

# Verify virtual environment is activated
which python  # Should point to .venv/bin/python
```

### 2. System Dependencies
```bash
# Check Tesseract is installed
tesseract --version  # Should show version 4.x or 5.x
```

### 3. Python Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import easyocr; print('EasyOCR installed')"
python -c "import pytesseract; print('pytesseract installed')"
```

## ✅ Project Structure Verification

Check that these directories and files exist:

```bash
# Key directories
ls -d src/services src/models src/utils src/config
ls -d tests/unit tests/integration
ls -d data

# Key files
ls main.py example_usage.py
ls README.md QUICKSTART.md MIGRATION_GUIDE.md
ls requirements.txt requirements.txt

# Config files
ls src/config/*.json

# Data files
ls data/character_info.csv
```

Expected output: All files/directories should be found.

## ✅ Code Import Verification

Test that modules can be imported:

```bash
python -c "from src.services import PreprocessingService; print('✓ PreprocessingService')"
python -c "from src.services import ValidationService; print('✓ ValidationService')"
python -c "from src.services import EasyOCRService; print('✓ EasyOCRService')"
python -c "from src.services import TesseractService; print('✓ TesseractService')"
python -c "from src.services import PipelineOrchestrator; print('✓ PipelineOrchestrator')"
python -c "from src.models.config import PipelineConfig; print('✓ PipelineConfig')"
python -c "from src.models.results import OCRPrediction; print('✓ OCRPrediction')"
```

All imports should succeed without errors.

## ✅ Configuration Verification

Test that configs load correctly:

```bash
python -c "
import json
from src.models.config import PipelineConfig

with open('src/config/default_config.json') as f:
    config = PipelineConfig.from_dict(json.load(f))
    print('✓ default_config.json loaded')
    print(f'  - {len(config.preprocessing_pipelines)} pipelines')
    print(f'  - {len(config.ocr_engines)} OCR engines')
"
```

## ✅ Basic Functionality Tests

### Test 1: Validation Service
```bash
python -c "
from src.services.validation_service import ValidationService

validator = ValidationService()
assert validator.validate('5', 'placement') == True
assert validator.validate('Mario', 'character_name') == True
assert validator.validate('45', 'score') == True
print('✓ Validation service working')
"
```

### Test 2: Preprocessing Service
```bash
python -c "
import cv2
import numpy as np
from src.services.preprocessing_service import PreprocessingService
from src.models.config import PreprocessingConfig, PreprocessingStep

service = PreprocessingService()
img = np.ones((100, 100, 3), dtype=np.uint8) * 128

config = PreprocessingConfig(
    name='test',
    steps=[PreprocessingStep(method='grayscale', params={})]
)

result = service.apply_pipeline(img, config)
assert result.shape == (100, 100)
print('✓ Preprocessing service working')
"
```

### Test 3: Data Service
```bash
python -c "
from src.services.data_service import DataService

service = DataService()
print('✓ Data service initialized')
"
```

## ✅ End-to-End Test (If Test Images Available)

### Test with a real image:
```bash
# Check if test images exist
ls game_images/inputs/pngs/*.png 2>/dev/null | head -1

# If images exist, run a test
python main.py --image game_images/inputs/pngs/img_8038.png --log-level INFO
```

Check outputs:
```bash
# Verify outputs were created
ls game_images/outputs/data/*.csv
ls game_images/outputs/annotations/*.jpg
ls game_images/outputs/grids/*.jpg
```

### Test batch processing:
```bash
python main.py --folder game_images/inputs/pngs --merge-results
```

Check merged outputs:
```bash
ls game_images/outputs/data/merged_predictions.csv
ls game_images/outputs/data/scoreboard.csv
```

## ✅ Unit Tests (Optional)

Run the test suite:

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Expected: All tests should pass.

## ✅ CLI Help Verification

Test that CLI help works:

```bash
python main.py --help
```

Should show usage information with all options.

## ✅ Example Script Verification

Run the example script:

```bash
python example_usage.py
```

This runs all example functions. Check for errors.

## ✅ Logging Verification

Test that logging works:

```bash
# Run with debug logging
python main.py --image game_images/inputs/pngs/img_8038.png --log-level DEBUG

# Check log file was created
ls logs/*.log
```

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution:** Ensure you're in the project root directory and virtual environment is activated.

### Issue: "Tesseract not found"
**Solution:** Install Tesseract OCR on your system (see README.md).

### Issue: "Character CSV not found"
**Solution:** Verify `data/character_info.csv` exists. If not, move it from `old/` folder.

### Issue: Import errors from old files
**Solution:** Old files have been moved to `old/` folder. Use new imports from `src.services.*`.

### Issue: "No images found"
**Solution:** Place test images in `game_images/inputs/pngs/` directory.

## Verification Complete ✓

If all checks passed, your installation is working correctly!

**Next Steps:**
1. Read the [QUICKSTART.md](QUICKSTART.md) for basic usage
2. Explore [example_usage.py](example_usage.py) for programmatic usage
3. Customize configs in `src/config/` for your needs
4. Process your Mario Kart screenshots!

## Getting Help

If any checks failed:
1. Check the error message carefully
2. Review installation steps in README.md
3. Check logs in `logs/` directory for detailed errors
4. Ensure all dependencies are installed
5. Verify Python version is 3.8+

Happy OCR-ing! 🏎️
