# Mario Kart Scoreboard OCR

A modular Python application for extracting post-race scoreboard data from Nintendo Switch Mario Kart screenshots using OCR (Optical Character Recognition).

## Features

- **Multi-Engine OCR**: Supports both EasyOCR and Tesseract with automatic fallback
- **Adaptive Preprocessing**: Multiple preprocessing pipelines with automatic retry logic
- **Smart Validation**: Validates predictions against known Mario Kart characters and game rules
- **Batch Processing**: Process single images or entire folders
- **HEIC Support**: Automatically convert iPhone HEIC images to PNG
- **Detailed Logging**: Comprehensive logging with configurable levels
- **Extensible Architecture**: Clean, modular design with well-defined interfaces
- **Configuration-Driven**: JSON-based configuration for easy customization

## Architecture

```
mario_kart_scores/
├── src/
│   ├── services/           # Core OCR and processing services
│   │   ├── base_ocr.py              # Abstract OCR interface
│   │   ├── easyocr_service.py       # EasyOCR implementation
│   │   ├── tesseract_service.py     # Tesseract implementation
│   │   ├── preprocessing_service.py  # Image preprocessing
│   │   ├── validation_service.py    # Prediction validation
│   │   ├── data_service.py          # CSV/data management
│   │   ├── conversion_service.py    # HEIC to PNG conversion
│   │   └── orchestrator.py          # Pipeline orchestration
│   ├── models/            # Data models and configuration
│   │   ├── config.py               # Configuration dataclasses
│   │   └── results.py              # Result dataclasses
│   ├── utils/             # Utility functions
│   │   ├── logging_config.py       # Logging setup
│   │   └── file_utils.py           # File operations
│   └── config/            # Configuration files
│       ├── default_config.json
│       ├── easyocr_only_config.json
│       └── tesseract_only_config.json
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── data/                  # Data files
│   └── character_info.csv
├── game_images/           # Input/output images
│   ├── inputs/
│   │   ├── heics/        # Original HEIC files
│   │   └── pngs/         # Converted PNG files
│   └── outputs/
│       ├── preprocessed/  # Preprocessed images
│       ├── grids/         # Grid overlay visualizations
│       ├── annotations/   # Annotated results
│       └── data/          # CSV predictions
├── main.py            # Main CLI entry point
└── requirements.txt       # Python dependencies
```

## Installation

### 1. Install System Dependencies

#### Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

**Other platforms:** See [Tesseract installation guide](https://github.com/tesseract-ocr/tesseract)

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # Includes dev dependencies
```

## Usage

### Basic Usage

Process a single image:
```bash
python main.py --image game_images/inputs/pngs/img_8038.png
```

Process all images in a folder:
```bash
python main.py --folder game_images/inputs/pngs
```

### Advanced Usage

Use a custom configuration:
```bash
python main.py --image image.png --config src/config/easyocr_only_config.json
```

Specify output directory:
```bash
python main.py --folder game_images/inputs/pngs --output-dir custom_outputs
```

Convert HEIC images to PNG:
```bash
python main.py --convert-heic game_images/inputs/heics
```

Enable debug logging:
```bash
python main.py --image image.png --log-level DEBUG
```

Merge all results after batch processing:
```bash
python main.py --folder game_images/inputs/pngs --merge-results
```

### Command Line Options

```
  --image IMAGE           Path to a single image file
  --folder FOLDER         Path to a folder containing images
  --convert-heic FOLDER   Convert HEIC images in folder to PNG
  --config CONFIG         Path to configuration JSON file (default: src/config/default_config.json)
  --output-dir DIR        Output directory (overrides config file)
  --merge-results         Merge all CSV results in output folder
  --log-level LEVEL       Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
  --log-dir DIR           Directory for log files (default: logs)
```

## Configuration

The application uses JSON configuration files to control behavior. See `src/config/` for examples.

### Key Configuration Sections

#### Grid Configuration
Defines the scoreboard grid overlay:
```json
{
  "grid": {
    "row_bounds": [[0, 0.08333], [0.08333, 0.16666], ...],
    "column_bounds": [[0, 0.07], [0.07, 0.16], ...],
    "target_columns": [1, 3, 5],
    "column_names": {
      "1": "placement",
      "3": "character_name",
      "5": "score"
    }
  }
}
```

#### Preprocessing Pipelines
Multiple pipelines with fallback:
```json
{
  "preprocessing_pipelines": [
    {
      "name": "edge_detection_primary",
      "priority": 0,
      "steps": [
        {"method": "grayscale", "params": {}},
        {"method": "gaussian_blur", "params": {"kernel": [5, 5]}},
        {"method": "edge_detection", "params": {"hysteresis_min": 100, "hysteresis_max": 200}}
      ]
    },
    {
      "name": "edge_detection_secondary",
      "priority": 1,
      "steps": [
        {"method": "grayscale", "params": {}},
        {"method": "gaussian_blur", "params": {"kernel": [7, 7]}},
        {"method": "edge_detection", "params": {"hysteresis_min": 120, "hysteresis_max": 200}}
      ]
    }...
  ]
}
```

#### OCR Engines
Configure multiple OCR engines:
```json
{
  "ocr_engines": [
    {
      "engine": "easyocr",
      "confidence_thresholds": {
        "placement": 0.6,
        "character_name": 0.5,
        "score": 0.7
      },
      "engine_params": {
        "languages": ["en"],
        "use_gpu": false
      }
    },
    {
      "engine": "tesseract",
      "confidence_thresholds": {
        "placement": 0.5,
        "character_name": 0.4,
        "score": 0.6
      },
      "engine_params": {
        "config": "--psm 6 --oem 3"
      }
    }
  ]
}
```

## Pipeline Flow

1. **Image Loading**: Load PNG or convert from HEIC
2. **Grid Extraction**: Divide image into cells based on grid configuration
3. **Preprocessing**: Apply preprocessing pipeline(s) to enhance text clarity
4. **OCR Extraction**: Run OCR engine(s) on each cell
5. **Validation**: Validate predictions against game rules (placement 1-12, known characters, etc.)
6. **Retry Logic**: If validation fails, try next preprocessing pipeline or OCR engine
7. **Result Aggregation**: Collect best predictions for each cell
8. **Output Generation**: Save CSVs, annotated images, grid visualizations

## Output Files

After processing, you'll find:

- **CSV Files** (`game_images/outputs/data/`): Detailed predictions with confidence scores
- **Annotated Images** (`game_images/outputs/annotations/`): Original images with bounding boxes
- **Grid Overlays** (`game_images/outputs/grids/`): Visualizations of the detection grid
- **Preprocessed Images** (`game_images/outputs/preprocessed/`): Intermediate preprocessing steps (if enabled)
- **Log Files** (`logs/`): Detailed execution logs

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -r requirements.txt  # Includes dev dependencies

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

## Development

### Adding a New OCR Engine

1. Create a new service class inheriting from `BaseOCRService`
2. Implement `extract_text_from_cell()` and `get_engine_name()`
3. Register the engine in `PipelineOrchestrator._initialize_ocr_services()`
4. Add configuration support in `models/config.py`

### Adding a New Preprocessing Method

1. Add the method to `PreprocessingService`
2. Add the method name to `supported_methods`
3. Update `_apply_step()` to handle the new method
4. Document parameters in configuration examples

## Troubleshooting

### Tesseract not found
- Ensure Tesseract is installed and in your PATH
- Try running `tesseract --version` to verify installation

### Low accuracy
- Try different preprocessing configurations
- Adjust confidence thresholds in config file
- Enable `save_intermediate: true` to debug preprocessing steps

### Import errors
- Ensure you're in the virtual environment: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt  # Includes dev dependencies`

## Project Status

This is version 2.0 - a complete refactoring from the original monolithic implementation to a modular, microservices-based architecture with help from Claude code.

### What's New in v2.0
- Modular service architecture
- Multi-engine OCR support with automatic fallback
- JSON-based configuration
- Comprehensive test suite
- Improved logging and error handling
- CLI with rich options
- Type hints throughout

## License

MIT License

## Acknowledgments

- Built with [EasyOCR](https://github.com/JaidedAI/EasyOCR) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- Uses OpenCV for image preprocessing
- Character data from Mario Kart 8 Deluxe
