"""Unit tests for ValidationService."""

import pytest
from src.services.validation_service import ValidationService


@pytest.fixture
def validation_service():
    """Create a ValidationService instance for testing."""
    return ValidationService(character_csv_path="data/character_info.csv")


class TestValidationService:
    """Test suite for ValidationService."""

    def test_validate_placement_valid(self, validation_service):
        """Test valid placement values."""
        assert validation_service.validate_placement("1")
        assert validation_service.validate_placement("6")
        assert validation_service.validate_placement("12")

    def test_validate_placement_invalid(self, validation_service):
        """Test invalid placement values."""
        assert not validation_service.validate_placement("0")
        assert not validation_service.validate_placement("13")
        assert not validation_service.validate_placement("abc")
        assert not validation_service.validate_placement("")

    def test_validate_score_valid(self, validation_service):
        """Test valid score values."""
        assert validation_service.validate_score("1")
        assert validation_service.validate_score("45")
        assert validation_service.validate_score("9999")

    def test_validate_score_invalid(self, validation_service):
        """Test invalid score values."""
        assert not validation_service.validate_score("0")
        assert not validation_service.validate_score("10000")
        assert not validation_service.validate_score("abc")
        assert not validation_service.validate_score("")

    def test_validate_character_name_valid(self, validation_service):
        """Test valid character names."""
        assert validation_service.validate_character_name("Mario")
        assert validation_service.validate_character_name("Luigi")
        assert validation_service.validate_character_name("Baby Mario")

    def test_validate_character_name_invalid(self, validation_service):
        """Test invalid character names."""
        assert not validation_service.validate_character_name("123")
        assert not validation_service.validate_character_name("NotACharacter")
        assert not validation_service.validate_character_name("")

    def test_validate_generic(self, validation_service):
        """Test generic validate method."""
        assert validation_service.validate("5", "placement")
        assert validation_service.validate("Mario", "character_name")
        assert validation_service.validate("45", "score")

    def test_validate_invalid_column_type(self, validation_service):
        """Test that invalid column types raise ValueError."""
        with pytest.raises(ValueError):
            validation_service.validate("test", "invalid_column")
