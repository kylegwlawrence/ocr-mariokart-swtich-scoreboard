#!/usr/bin/env python3
"""
Mario Kart Scoreboard OCR - Main Entry Point

Extracts text from Nintendo Switch Mario Kart scoreboard screenshots.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from src.utils.logging_config import setup_logging, get_logger
from src.utils.file_utils import get_files_by_extension
from src.services.orchestrator import PipelineOrchestrator
from src.services.data_service import DataService
from src.services.conversion_service import ConversionService
from src.models.config import PipelineConfig

logger = get_logger(__name__)


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from JSON file.

    Args:
        config_path: Path to configuration JSON file.

    Returns:
        PipelineConfig object.
    """
    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return PipelineConfig.from_dict(config_dict)


def process_single_image(
    image_path: str,
    orchestrator: PipelineOrchestrator,
    data_service: DataService,
    output_dir: str
) -> None:
    """Process a single image through the pipeline.

    Args:
        image_path: Path to input image.
        orchestrator: Pipeline orchestrator instance.
        data_service: Data service instance.
        output_dir: Directory for outputs.
    """
    logger.info(f"Processing image: {image_path}")

    # Process the image
    result = orchestrator.process_image(
        image_path=image_path,
        output_dir=output_dir,
        save_grid=True,
        save_annotated=True
    )

    # Save predictions to CSV
    csv_path = Path(output_dir) / "data" / f"{Path(image_path).stem}_{result.timestamp.replace(' ', '_').replace(':', '-')}.csv"
    data_service.save_image_result(result, str(csv_path), include_all_predictions=True)

    # Log summary
    logger.info(
        f"Image processing complete: "
        f"{result.successful_cells}/{result.total_cells} cells successful "
        f"({result.get_success_rate() * 100:.1f}%) "
        f"in {result.processing_time_seconds:.2f}s"
    )


def process_batch(
    image_paths: List[str],
    orchestrator: PipelineOrchestrator,
    data_service: DataService,
    output_dir: str
) -> None:
    """Process a batch of images.

    Args:
        image_paths: List of image paths.
        orchestrator: Pipeline orchestrator instance.
        data_service: Data service instance.
        output_dir: Directory for outputs.
    """
    logger.info(f"Processing batch of {len(image_paths)} images")

    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"Processing image {i}/{len(image_paths)}")
        try:
            process_single_image(image_path, orchestrator, data_service, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}", exc_info=True)

    logger.info("Batch processing complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mario Kart Scoreboard OCR - Extract text from scoreboard images"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single image file"
    )
    input_group.add_argument(
        "--folder",
        type=str,
        help="Path to a folder containing images"
    )
    input_group.add_argument(
        "--convert-heic",
        type=str,
        help="Convert HEIC images in folder to PNG"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default_config.json",
        help="Path to configuration JSON file (default: src/config/default_config.json)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config file)"
    )
    parser.add_argument(
        "--merge-results",
        action="store_true",
        help="Merge all CSV results in output folder"
    )

    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level, log_dir=args.log_dir)
    logger.info("Mario Kart Scoreboard OCR started")

    # Handle HEIC conversion
    if args.convert_heic:
        conversion_service = ConversionService()
        output_folder = Path(args.convert_heic) / "converted"
        count = conversion_service.batch_convert_heic_to_png(
            args.convert_heic,
            str(output_folder)
        )
        logger.info(f"Converted {count} HEIC images to PNG in {output_folder}")
        return

    # Load configuration
    config = load_config(args.config)

    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir

    # Initialize services
    orchestrator = PipelineOrchestrator(config)
    data_service = DataService()

    # Process image(s)
    if args.image:
        # Single image
        process_single_image(args.image, orchestrator, data_service, config.output_dir)

    elif args.folder:
        # Batch processing
        image_paths = get_files_by_extension(args.folder, '.png')
        image_paths.extend(get_files_by_extension(args.folder, '.jpg'))
        image_paths.extend(get_files_by_extension(args.folder, '.jpeg'))

        if not image_paths:
            logger.error(f"No images found in {args.folder}")
            return

        process_batch(image_paths, orchestrator, data_service, config.output_dir)

    # Merge results if requested
    if args.merge_results:
        logger.info("Merging CSV results...")
        data_folder = Path(config.output_dir) / "data"
        merged_path = data_folder / "merged_predictions.csv"
        merged_df = data_service.merge_csv_files(str(data_folder), str(merged_path))

        # Create scoreboard table
        scoreboard_path = data_folder / "scoreboard.csv"
        data_service.create_scoreboard_table(merged_df, str(scoreboard_path))

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
