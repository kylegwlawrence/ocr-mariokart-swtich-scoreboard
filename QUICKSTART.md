# Quick Start Guide

Get up and running with Mario Kart Scoreboard OCR in 5 minutes!

## Prerequisites

- Python 3.8+
- Tesseract OCR installed on your system

## Installation

### 1. Clone/Download the repository
```bash
cd mario_kart_scores
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

## Quick Test

### Process a single image

```bash
python main.py --image game_images/inputs/pngs/img_8038.png
```

This will:
1. Load the image
2. Apply preprocessing
3. Run OCR on the scoreboard
4. Save results to `game_images/outputs/`

### Check the outputs

Look in `game_images/outputs/` for:
- `data/*.csv` - CSV with extracted data
- `annotations/*.jpg` - Original image with detected text boxes
- `grids/*.jpg` - Visualization of the detection grid

## Process Multiple Images

```bash
python main.py --folder game_images/inputs/pngs
```

After processing, merge all results:
```bash
python main.py --folder game_images/inputs/pngs --merge-results
```

This creates:
- `game_images/outputs/data/merged_predictions.csv` - All predictions combined
- `game_images/outputs/data/scoreboard.csv` - Clean scoreboard table

## Convert iPhone Images

If you have HEIC images from iPhone:

```bash
python main.py --convert-heic game_images/inputs/heics
```

This converts all HEIC files to PNG.

## Configuration

### Use different OCR engines

**EasyOCR only:**
```bash
python main.py --image image.png --config src/config/easyocr_only_config.json
```

**Tesseract only:**
```bash
python main.py --image image.png --config src/config/tesseract_only_config.json
```

**Both (with automatic fallback):**
```bash
python main.py --image image.png --config src/config/default_config.json
```

### Adjust preprocessing

Edit `src/config/default_config.json` and modify the `preprocessing_pipelines` section:

```json
{
  "preprocessing_pipelines": [
    {
      "name": "my_custom_pipeline",
      "priority": 0,
      "steps": [
        {"method": "grayscale", "params": {}},
        {"method": "gaussian_blur", "params": {"kernel": [7, 7], "sigmaX": 0, "sigmaY": 0}},
        {"method": "edge_detection", "params": {"hysteresis_min": 150, "hysteresis_max": 225}}
      ]
    }
  ]
}
```

## Understanding Results

### CSV Output

Each CSV contains:

| Column | Description |
|--------|-------------|
| `row_idx` | Row in the grid (0-11 for 12 players) |
| `col_idx` | Column index |
| `col_name` | Column type (placement, character_name, score) |
| `text` | Extracted text |
| `confidence` | OCR confidence (0-1) |
| `is_acceptable` | Whether prediction passed validation |
| `preprocessing_pipeline` | Which preprocessing was used |
| `ocr_engine` | Which OCR engine was used |

### Merged Scoreboard

The scoreboard CSV has columns:
- `row` - Player rank
- `placement` - Place (1-12)
- `character_name` - Character name
- `score` - Points earned

## Debugging

### Enable debug logging

```bash
python main.py --image image.png --log-level DEBUG
```

Check logs in `logs/` directory for detailed execution traces.

### Save intermediate preprocessing steps

Edit your config file and set:
```json
{
  "save_intermediate": true
}
```

This saves all preprocessing steps to `game_images/outputs/preprocessed/` for debugging.

### Check grid alignment

The grid overlay image shows how the detection grid is positioned. If text is being missed:

1. Check `game_images/outputs/grids/` to see grid alignment
2. Adjust `row_bounds` and `column_bounds` in config if needed

## Common Issues

### "ModuleNotFoundError: No module named 'easyocr'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "Tesseract not found"

Install Tesseract OCR on your system (see Installation section above).

### Low accuracy

Try:
1. Different preprocessing configurations
2. Adjust confidence thresholds in config
3. Use both OCR engines with automatic fallback
4. Enable debug logging to see what's failing

### "Character CSV not found"

The character database should be at `data/character_info.csv`. If moved, update the path in code or move it back.

## Next Steps

- Read the full [README.md](README.md) for advanced usage
- Check [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) if upgrading from v1
- Run tests: `pytest tests/`
- Create custom configurations for your specific needs

## Getting Help

1. Check logs in `logs/` directory
2. Run with `--log-level DEBUG` for detailed output
3. Review configuration in `src/config/` for examples
4. Check the test files in `tests/` for usage examples

Happy OCR-ing! 🏎️
