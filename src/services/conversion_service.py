"""Image conversion service.

Handles conversion between image formats (e.g., HEIC to PNG).
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ConversionService:
    """Service for converting images between formats.

    Primarily handles HEIC to PNG conversion for iPhone images.
    """

    def __init__(self):
        """Initialize the conversion service."""
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Validate required libraries are installed."""
        try:
            import pillow_heif
            from PIL import Image
        except ImportError as e:
            logger.warning(
                f"Image conversion dependencies not fully installed: {e}. "
                "Install with: pip install pillow pillow-heif"
            )

    def convert_heic_to_png(self, input_path: str, output_path: str) -> bool:
        """Convert a HEIC image to PNG format.

        Args:
            input_path: Path to input HEIC file.
            output_path: Path to save output PNG file.

        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            import pillow_heif
            from PIL import Image

            # Read HEIC file
            heif_file = pillow_heif.read_heif(input_path)

            # Convert to PIL Image
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw"
            )

            # Save as PNG
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, "PNG")

            logger.info(f"Converted {input_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to convert {input_path}: {e}")
            return False

    def batch_convert_heic_to_png(
        self,
        input_folder: str,
        output_folder: Optional[str] = None
    ) -> int:
        """Batch convert all HEIC images in a folder to PNG.

        Args:
            input_folder: Folder containing HEIC images.
            output_folder: Folder to save PNG images. If None, saves to same folder.

        Returns:
            Number of images successfully converted.
        """
        input_path = Path(input_folder)

        if not input_path.exists():
            logger.error(f"Input folder does not exist: {input_folder}")
            return 0

        # Find all HEIC files
        heic_files = list(input_path.glob("*.heic")) + list(input_path.glob("*.HEIC"))

        if not heic_files:
            logger.info(f"No HEIC files found in {input_folder}")
            return 0

        logger.info(f"Found {len(heic_files)} HEIC files to convert")

        # Convert each file
        converted_count = 0
        for heic_file in heic_files:
            # Determine output path
            if output_folder:
                output_path = Path(output_folder) / f"{heic_file.stem}.png"
            else:
                output_path = heic_file.with_suffix('.png')

            # Convert
            if self.convert_heic_to_png(str(heic_file), str(output_path)):
                converted_count += 1

        logger.info(f"Successfully converted {converted_count}/{len(heic_files)} images")
        return converted_count

    def convert_image(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "PNG"
    ) -> bool:
        """Convert an image to a different format (generic).

        Args:
            input_path: Path to input image.
            output_path: Path to save output image.
            output_format: Output format (e.g., 'PNG', 'JPEG').

        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            from PIL import Image

            image = Image.open(input_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, output_format)

            logger.info(f"Converted {input_path} -> {output_path} ({output_format})")
            return True

        except Exception as e:
            logger.error(f"Failed to convert {input_path}: {e}")
            return False
