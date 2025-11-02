#!/usr/bin/env python3
"""
Example usage of the Mario Kart OCR library.

This script demonstrates how to use the refactored services programmatically
instead of through the CLI.
"""

import json
from pathlib import Path

# Import services
from src.services.orchestrator import PipelineOrchestrator
from src.services.data_service import DataService
from src.services.conversion_service import ConversionService
from src.models.config import PipelineConfig
from src.utils.logging_config import setup_logging


def example_1_basic_processing():
    """Example 1: Basic image processing with default config."""
    print("\n" + "="*60)
    print("Example 1: Basic Image Processing")
    print("="*60)

    # Setup logging
    setup_logging(log_level="INFO")

    # Load configuration
    with open("src/config/default_config.json") as f:
        config = PipelineConfig.from_dict(json.load(f))

    # Create orchestrator
    orchestrator = PipelineOrchestrator(config)

    # Process an image
    image_path = "game_images/inputs/pngs/img_8038.png"

    if Path(image_path).exists():
        result = orchestrator.process_image(
            image_path=image_path,
            save_grid=True,
            save_annotated=True
        )

        # Print summary
        print(f"\nProcessed: {result.image_path}")
        print(f"Success Rate: {result.get_success_rate() * 100:.1f}%")
        print(f"Processing Time: {result.processing_time_seconds:.2f}s")
        print(f"Successful Cells: {result.successful_cells}/{result.total_cells}")

        # Show some predictions
        acceptable = result.get_acceptable_predictions()
        print(f"\nFound {len(acceptable)} acceptable predictions:")
        for pred in acceptable[:5]:  # Show first 5
            print(f"  Row {pred.row_idx}, {pred.col_name}: '{pred.text}' (confidence: {pred.confidence:.2f})")
    else:
        print(f"Image not found: {image_path}")


def example_2_custom_config():
    """Example 2: Using a custom configuration."""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)

    setup_logging(log_level="INFO")

    # Create a minimal custom configuration
    custom_config = {
        "grid": {
            "row_bounds": [
                [0, 0.08333], [0.08333, 0.16666], [0.16666, 0.25],
                [0.25, 0.33333], [0.33333, 0.41666], [0.41666, 0.5],
                [0.5, 0.58333], [0.58333, 0.66666], [0.66666, 0.75],
                [0.75, 0.83333], [0.83333, 0.91666], [0.91666, 1.0]
            ],
            "column_bounds": [
                [0, 0.07], [0.07, 0.16], [0.16, 0.24],
                [0.24, 0.7], [0.7, 0.915], [0.915, 1.0]
            ],
            "target_columns": [1, 3, 5],
            "column_names": {"1": "placement", "3": "character_name", "5": "score"}
        },
        "preprocessing_pipelines": [
            {
                "name": "simple_pipeline",
                "priority": 0,
                "steps": [
                    {"method": "grayscale", "params": {}},
                    {"method": "gaussian_blur", "params": {"kernel": [5, 5], "sigmaX": 0, "sigmaY": 0}}
                ]
            }
        ],
        "ocr_engines": [
            {
                "engine": "tesseract",
                "confidence_thresholds": {
                    "placement": 0.5,
                    "character_name": 0.4,
                    "score": 0.6
                },
                "engine_params": {
                    "psm_modes": {"placement": 6, "character_name": 6, "score": 6}
                }
            }
        ],
        "max_retries_per_cell": 1,
        "output_dir": "game_images/outputs",
        "save_intermediate": False
    }

    # Create config object
    config = PipelineConfig.from_dict(custom_config)

    print(f"Created custom config with:")
    print(f"  - {len(config.preprocessing_pipelines)} preprocessing pipeline(s)")
    print(f"  - {len(config.ocr_engines)} OCR engine(s)")
    print(f"  - {config.max_retries_per_cell} max retries per cell")


def example_3_save_and_merge_results():
    """Example 3: Processing multiple images and merging results."""
    print("\n" + "="*60)
    print("Example 3: Batch Processing and Merging")
    print("="*60)

    setup_logging(log_level="INFO")

    # Load config
    with open("src/config/easyocr_only_config.json") as f:
        config = PipelineConfig.from_dict(json.load(f))

    orchestrator = PipelineOrchestrator(config)
    data_service = DataService()

    # Find all PNG images
    image_dir = Path("game_images/inputs/pngs")
    if not image_dir.exists():
        print(f"Directory not found: {image_dir}")
        return

    images = list(image_dir.glob("*.png"))[:3]  # Process first 3 images

    if not images:
        print("No images found")
        return

    print(f"Processing {len(images)} images...")

    results = []
    for i, image_path in enumerate(images, 1):
        print(f"\nProcessing {i}/{len(images)}: {image_path.name}")

        result = orchestrator.process_image(
            str(image_path),
            save_grid=False,
            save_annotated=True
        )

        # Save individual CSV
        csv_path = f"game_images/outputs/data/example_{image_path.stem}.csv"
        data_service.save_image_result(result, csv_path, include_all_predictions=True)

        results.append(result)

        print(f"  Success: {result.get_success_rate() * 100:.1f}%")

    print(f"\n{len(results)} images processed!")

    # Merge results
    print("\nMerging results...")
    merged_df = data_service.merge_csv_files(
        "game_images/outputs/data",
        "game_images/outputs/data/example_merged.csv",
        pattern="example_*.csv"
    )

    print(f"Merged {len(merged_df)} predictions into single file")

    # Create scoreboard table
    scoreboard = data_service.create_scoreboard_table(
        merged_df,
        "game_images/outputs/data/example_scoreboard.csv"
    )

    if not scoreboard.empty:
        print(f"\nScoreboard preview (first 5 rows):")
        print(scoreboard.head())


def example_4_heic_conversion():
    """Example 4: Converting HEIC images."""
    print("\n" + "="*60)
    print("Example 4: HEIC Conversion")
    print("="*60)

    conversion_service = ConversionService()

    heic_folder = "game_images/inputs/heics"
    if not Path(heic_folder).exists():
        print(f"HEIC folder not found: {heic_folder}")
        return

    print(f"Converting HEIC images from: {heic_folder}")

    count = conversion_service.batch_convert_heic_to_png(
        heic_folder,
        output_folder="game_images/inputs/pngs"
    )

    print(f"Converted {count} HEIC images to PNG")


def example_5_access_services_directly():
    """Example 5: Using individual services directly."""
    print("\n" + "="*60)
    print("Example 5: Using Services Directly")
    print("="*60)

    from src.services.validation_service import ValidationService

    # Create validation service
    validator = ValidationService()

    # Test some validations
    test_cases = [
        ("1", "placement"),
        ("13", "placement"),
        ("Mario", "character_name"),
        ("NotACharacter", "character_name"),
        ("45", "score"),
        ("abc", "score"),
    ]

    print("\nValidation Tests:")
    for text, col_type in test_cases:
        is_valid = validator.validate(text, col_type)
        status = "✓" if is_valid else "✗"
        print(f"  {status} '{text}' as {col_type}: {is_valid}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Mario Kart OCR - Example Usage")
    print("="*60)

    # Run examples
    example_1_basic_processing()
    example_2_custom_config()
    example_3_save_and_merge_results()
    example_4_heic_conversion()
    example_5_access_services_directly()

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
