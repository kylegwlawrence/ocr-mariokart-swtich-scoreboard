"""Configuration data models.

Dataclasses for managing pipeline, OCR, and preprocessing configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class OCREngine(Enum):
    """Supported OCR engines."""
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"


@dataclass
class GridConfig:
    """Configuration for the scoreboard grid overlay.

    Attributes:
        row_bounds: List of (start, end) tuples as fractions of image height.
        column_bounds: List of (start, end) tuples as fractions of image width.
        target_columns: Column indices to process (e.g., [1, 3, 5] for placement, character, score).
        column_names: Mapping of column index to semantic name.
    """
    row_bounds: List[Tuple[float, float]]
    column_bounds: List[Tuple[float, float]]
    target_columns: List[int] = field(default_factory=lambda: [1, 3, 5])
    column_names: Dict[int, str] = field(default_factory=lambda: {
        1: "placement",
        3: "character_name",
        5: "score"
    })

    def __post_init__(self):
        """Validate configuration."""
        for start, end in self.row_bounds:
            if not (0 <= start < end <= 1):
                raise ValueError(f"Invalid row bounds: ({start}, {end})")
        for start, end in self.column_bounds:
            if not (0 <= start < end <= 1):
                raise ValueError(f"Invalid column bounds: ({start}, {end})")


@dataclass
class PreprocessingStep:
    """Single preprocessing step configuration.

    Attributes:
        method: Name of the preprocessing method. Supported methods:
            - "grayscale": Convert image to grayscale (no params needed)
            - "gaussian_blur": Apply Gaussian blur. Params: kernel (list of 2 odd ints),
              sigmaX (int, default 0), sigmaY (int, default 0)
            - "edge_detection": Detect edges using Canny algorithm.
              Params: hysteresis_min (int, default 100), hysteresis_max (int, default 200)
            - "dilate": Expand white regions. Params: kernel (list of 2 ints, default [3,3]),
              iterations (int, default 1)
            - "erode": Shrink white regions. Params: kernel (list of 2 ints, default [3,3]),
              iterations (int, default 1)
            - "threshold": Apply binary threshold. Params: threshold (int, default 200),
              max_value (int, default 255)
            - "adaptive_threshold": Adaptive thresholding per neighborhood. Params: max_value
              (int, default 255), block_size (int, default 15, must be odd), C (int, default -2)
            - "inversion": Invert image colors (no params needed)
            - "morphology": Morphological operations. Params: operation (str: "open", "close",
              "gradient", "tophat", "blackhat", default "open"), kernel (list of 2 ints, default [2,2])
            - "blur": Basic blur filter. Params: kernel (list of 2 ints, default [5,5])
            - "contrast": Adjust contrast and brightness. Params: alpha (float, 1.0-3.0,
              default 1.0), beta (int, 0-100, default 0)
            - "median_blur": Median filter for noise reduction. Params: ksize (int, default 5, must be odd)
            - "bilateral_filter": Edge-preserving filter. Params: d (int, default 9),
              sigmaColor (int, default 75), sigmaSpace (int, default 75)
        params: Parameters specific to each method, passed as a dictionary.
    """
    method: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    """Configuration for a preprocessing pipeline.

    A pipeline consists of a sequence of preprocessing steps applied to images
    to enhance text clarity before OCR. Multiple pipelines can be configured
    with different priorities for fallback behavior.

    Attributes:
        name: Human-readable name for this pipeline (e.g., "edge_detection_primary").
        steps: List of preprocessing steps to apply in order. Steps are applied
            sequentially to transform the image.
        priority: Priority order for fallback (lower = try first). Priority 0 is
            the primary pipeline, priority 1 is secondary, etc. If OCR fails on
            a cell with one pipeline, the next priority pipeline is attempted.
    """
    name: str
    steps: List[PreprocessingStep]
    priority: int = 0


@dataclass
class OCREngineConfig:
    """Configuration for a specific OCR engine.

    Attributes:
        engine: Which OCR engine to use (OCREngine.EASYOCR or OCREngine.TESSERACT).
        confidence_thresholds: Minimum confidence scores (0.0-1.0) required for
            predictions to be accepted per column type. Keys should match column_names
            in GridConfig (e.g., "placement", "character_name", "score"). Predictions
            below the threshold are rejected and trigger retry with next pipeline/engine.
        engine_params: Engine-specific parameters:
            - EasyOCR: {"languages": ["en"], "use_gpu": false/true}
            - Tesseract: {"config": "--psm 6 --oem 3"} (Tesseract command-line options)
              PSM modes: 6 (assume single block), 7 (single text line), etc.
              OEM modes: 0 (legacy), 1 (neural), 2 (legacy+neural), 3 (auto)
    """
    engine: OCREngine
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "placement": 0.6,
        "character_name": 0.5,
        "score": 0.7
    })
    engine_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Complete pipeline configuration.

    Defines the entire OCR pipeline: how to divide the scoreboard into cells,
    how to preprocess images, and which OCR engines to use with what parameters.

    Attributes:
        grid: Grid configuration defining cell boundaries and target columns.
        preprocessing_pipelines: List of preprocessing pipelines to try in priority
            order. If OCR with one pipeline fails, the next pipeline is attempted.
        ocr_engines: List of OCR engine configurations to try in order. If one
            engine's confidence is below threshold, the next engine is attempted.
        max_retries_per_cell: Maximum number of preprocessing pipelines to attempt
            per cell before giving up (default: 3). Limited by the number of
            configured pipelines.
        output_dir: Base directory for saving results, logs, and intermediate images.
        save_intermediate: If True, saves preprocessed images for each pipeline
            attempt (useful for debugging preprocessing effectiveness).
    """
    grid: GridConfig
    preprocessing_pipelines: List[PreprocessingConfig]
    ocr_engines: List[OCREngineConfig]
    max_retries_per_cell: int = 3
    output_dir: str = "game_images/outputs"
    save_intermediate: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from a dictionary.

        Args:
            config_dict: Configuration dictionary (typically from JSON).

        Returns:
            PipelineConfig instance.
        """
        # Parse grid config
        grid_data = config_dict["grid"]
        grid = GridConfig(
            row_bounds=grid_data["row_bounds"],
            column_bounds=grid_data["column_bounds"],
            target_columns=grid_data.get("target_columns", [1, 3, 5]),
            column_names=grid_data.get("column_names", {1: "placement", 3: "character_name", 5: "score"})
        )

        # Parse preprocessing pipelines
        preprocessing_pipelines = []
        for pp_data in config_dict["preprocessing_pipelines"]:
            steps = [
                PreprocessingStep(method=step["method"], params=step.get("params", {}))
                for step in pp_data["steps"]
            ]
            preprocessing_pipelines.append(
                PreprocessingConfig(
                    name=pp_data["name"],
                    steps=steps,
                    priority=pp_data.get("priority", 0)
                )
            )

        # Sort by priority
        preprocessing_pipelines.sort(key=lambda x: x.priority)

        # Parse OCR engine configs
        ocr_engines = []
        for engine_data in config_dict["ocr_engines"]:
            ocr_engines.append(
                OCREngineConfig(
                    engine=OCREngine(engine_data["engine"]),
                    confidence_thresholds=engine_data.get("confidence_thresholds", {}),
                    engine_params=engine_data.get("engine_params", {})
                )
            )

        return cls(
            grid=grid,
            preprocessing_pipelines=preprocessing_pipelines,
            ocr_engines=ocr_engines,
            max_retries_per_cell=config_dict.get("max_retries_per_cell", 3),
            output_dir=config_dict.get("output_dir", "game_images/outputs"),
            save_intermediate=config_dict.get("save_intermediate", False)
        )
