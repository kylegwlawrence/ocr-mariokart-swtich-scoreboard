"""Validation service for OCR predictions.

Validates OCR predictions against known rules for Mario Kart scoreboards.
"""

import csv
import logging
from pathlib import Path
from typing import Set, Optional

logger = logging.getLogger(__name__)


class ValidationService:
    """Service for validating OCR predictions.

    Validates placement ranks, character names, and scores against
    known constraints for Mario Kart scoreboards.
    """

    def __init__(self, character_csv_path: str = "data/character_info.csv"):
        """Initialize the validation service.

        Args:
            character_csv_path: Path to the character info CSV file.
        """
        self.character_csv_path = character_csv_path
        self._valid_characters: Optional[Set[str]] = None
        self._load_character_names()

    def _load_character_names(self) -> None:
        """Load valid character names from CSV file (cached)."""
        if self._valid_characters is not None:
            return  # Already loaded

        self._valid_characters = set()

        if not Path(self.character_csv_path).exists():
            logger.warning(f"Character CSV not found: {self.character_csv_path}")
            return

        try:
            with open(self.character_csv_path, mode='r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader, None)  # Skip header

                for row in csv_reader:
                    if row and len(row) > 0:
                        # Normalize: lowercase, strip whitespace, remove spaces
                        normalized_name = row[0].lower().strip().replace(' ', '')
                        self._valid_characters.add(normalized_name)

            logger.info(f"Loaded {len(self._valid_characters)} valid character names")

        except Exception as e:
            logger.error(f"Error loading character names: {e}")
            self._valid_characters = set()

    def validate_placement(self, text: str) -> bool:
        """Validate placement text (should be 1-12).

        Args:
            text: Text to validate.

        Returns:
            True if valid placement, False otherwise.
        """
        normalized = text.lower().strip().replace(' ', '')

        if not normalized.isdigit():
            return False

        try:
            num = int(normalized)
            return 1 <= num <= 12
        except ValueError:
            return False

    def validate_character_name(self, text: str) -> bool:
        """Validate character name against known characters.

        Args:
            text: Text to validate.

        Returns:
            True if valid character name, False otherwise.
        """
        # Ensure characters are loaded
        if self._valid_characters is None:
            self._load_character_names()

        # Check if text contains only letters and spaces
        if not text.replace(" ", "").isalpha():
            return False

        # Normalize and check against valid characters
        normalized = text.lower().strip().replace(' ', '')

        if normalized in self._valid_characters:
            return True

        logger.debug(f"Character name '{text}' not found in valid characters")
        return False

    def validate_score(self, text: str) -> bool:
        """Validate score text (should be positive digits).

        Args:
            text: Text to validate.

        Returns:
            True if valid score, False otherwise.
        """
        normalized = text.lower().strip().replace(' ', '')

        if not normalized.isdigit():
            return False

        try:
            num = int(normalized)
            # Arbitrary max score (scores shouldn't be crazy high)
            return 0 < num < 10000
        except ValueError:
            return False

    def validate(self, text: str, column_type: str) -> bool:
        """Validate text based on column type.

        Args:
            text: Text to validate.
            column_type: Type of column ('placement', 'character_name', 'score').

        Returns:
            True if valid, False otherwise.

        Raises:
            ValueError: If column_type is not recognized.
        """
        column_type_lower = column_type.lower()

        if column_type_lower in ["placement", "rank"]:
            return self.validate_placement(text)
        elif column_type_lower in ["character_name", "character", "name"]:
            return self.validate_character_name(text)
        elif column_type_lower == "score":
            return self.validate_score(text)
        else:
            raise ValueError(
                f"Unknown column type: {column_type}. "
                "Must be one of: placement, character_name, score"
            )
