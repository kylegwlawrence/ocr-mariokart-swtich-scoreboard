"""File utility functions.

Helper functions for file and directory operations.
"""

import os
import shutil
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist.

    Args:
        directory: Path to directory.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def delete_files_in_directory(directory: str, pattern: str = "*") -> int:
    """Delete all files matching pattern in a directory.

    Args:
        directory: Path to directory.
        pattern: Glob pattern for files to delete (default: all files).

    Returns:
        Number of files deleted.

    Raises:
        ValueError: If directory doesn't exist.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    deleted_count = 0
    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            file_path.unlink()
            logger.debug(f"Deleted file: {file_path.name}")
            deleted_count += 1

    logger.info(f"Deleted {deleted_count} file(s) from {directory}")
    return deleted_count


def get_files_by_extension(directory: str, extension: str) -> List[str]:
    """Get all files with a specific extension in a directory.

    Args:
        directory: Path to directory.
        extension: File extension (e.g., '.png', '.csv').

    Returns:
        List of file paths.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        return []

    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension

    files = [str(f) for f in dir_path.glob(f"*{extension}")]
    logger.debug(f"Found {len(files)} {extension} files in {directory}")
    return files
