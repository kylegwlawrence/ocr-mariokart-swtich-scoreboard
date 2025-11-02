"""Data service for CSV operations and data cleaning.

Handles merging, cleaning, and exporting OCR prediction data.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from ..models.results import OCRPrediction, ImageResult

logger = logging.getLogger(__name__)


class DataService:
    """Service for managing OCR prediction data.

    Handles CSV export, merging multiple results, and data cleaning operations.
    """

    def __init__(self):
        """Initialize the data service."""
        pass

    def save_predictions_to_csv(
        self,
        predictions: List[OCRPrediction],
        output_path: str
    ) -> None:
        """Save OCR predictions to a CSV file.

        Args:
            predictions: List of OCR predictions.
            output_path: Path to save CSV file.
        """
        if not predictions:
            logger.warning("No predictions to save")
            return

        # Convert predictions to dictionaries
        data = [pred.to_dict() for pred in predictions]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions)} predictions to {output_path}")

    def save_image_result(
        self,
        result: ImageResult,
        output_path: str,
        include_all_predictions: bool = True
    ) -> None:
        """Save ImageResult to CSV.

        Args:
            result: Image processing result.
            output_path: Path to save CSV file.
            include_all_predictions: If True, saves all predictions. If False, only acceptable ones.
        """
        if include_all_predictions:
            predictions = result.get_all_predictions()
        else:
            predictions = result.get_acceptable_predictions()

        self.save_predictions_to_csv(predictions, output_path)

    def merge_csv_files(
        self,
        input_folder: str,
        output_path: str,
        pattern: str = "*.csv"
    ) -> pd.DataFrame:
        """Merge multiple CSV files into a single DataFrame.

        Args:
            input_folder: Folder containing CSV files.
            output_path: Path to save merged CSV.
            pattern: Glob pattern for CSV files to merge.

        Returns:
            Merged DataFrame.
        """
        input_path = Path(input_folder)

        if not input_path.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")

        # Find all CSV files
        csv_files = list(input_path.glob(pattern))

        if not csv_files:
            logger.warning(f"No CSV files found matching pattern '{pattern}' in {input_folder}")
            return pd.DataFrame()

        logger.info(f"Found {len(csv_files)} CSV files to merge")

        # Read and merge
        dataframes = []
        failed_files = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add source file column
                df['source_file'] = csv_file.name
                dataframes.append(df)
                logger.debug(f"Loaded {len(df)} rows from {csv_file.name}")
            except Exception as e:
                logger.error(f"Failed to read {csv_file.name}: {e}")
                failed_files.append(csv_file.name)

        if not dataframes:
            logger.error("No CSV files could be loaded")
            return pd.DataFrame()

        # Concatenate all DataFrames
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Save merged CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)

        logger.info(
            f"Merged {len(dataframes)} CSV files ({len(merged_df)} total rows) -> {output_path}"
        )

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")

        return merged_df

    def clean_predictions_data(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Clean and process OCR predictions DataFrame.

        Performs operations like:
        - Removing duplicates
        - Parsing metadata columns
        - Creating aggregate columns
        - Filtering by quality metrics

        Args:
            df: Input DataFrame.
            output_path: Optional path to save cleaned data.

        Returns:
            Cleaned DataFrame.
        """
        cleaned_df = df.copy()

        # Parse preprocessing_pipeline if it's a string
        if 'preprocessing_pipeline' in cleaned_df.columns:
            logger.debug("Parsing preprocessing_pipeline column")
            # Already parsed, no need for literal_eval

        # Create a unique configuration identifier
        if 'preprocessing_pipeline' in cleaned_df.columns and 'ocr_engine' in cleaned_df.columns:
            cleaned_df['config_id'] = (
                cleaned_df['preprocessing_pipeline'].astype(str) +
                "_" +
                cleaned_df['ocr_engine'].astype(str)
            )

        # Filter only acceptable predictions if column exists
        if 'is_acceptable' in cleaned_df.columns:
            acceptable_count = cleaned_df['is_acceptable'].sum()
            total_count = len(cleaned_df)
            logger.info(
                f"Acceptable predictions: {acceptable_count}/{total_count} "
                f"({100 * acceptable_count / total_count:.1f}%)"
            )

        # Sort by confidence (descending)
        if 'confidence' in cleaned_df.columns:
            cleaned_df = cleaned_df.sort_values('confidence', ascending=False)

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned data to {output_path}")

        return cleaned_df

    def get_best_predictions_per_cell(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the best prediction for each grid cell.

        Args:
            df: DataFrame with all predictions.

        Returns:
            DataFrame with one row per cell (the best prediction).
        """
        if 'row_idx' not in df.columns or 'col_idx' not in df.columns:
            logger.error("DataFrame missing required columns: row_idx, col_idx")
            return df

        # Group by cell and get highest confidence prediction
        best_predictions = (
            df.sort_values('confidence', ascending=False)
            .groupby(['row_idx', 'col_idx'])
            .first()
            .reset_index()
        )

        logger.info(f"Reduced {len(df)} predictions to {len(best_predictions)} best predictions")
        return best_predictions

    def create_scoreboard_table(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Create a structured scoreboard table from predictions.

        Pivots the data so each row represents one player/rank with
        columns for placement, character, and score.

        Args:
            df: DataFrame with predictions.
            output_path: Optional path to save the scoreboard table.

        Returns:
            Structured scoreboard DataFrame.
        """
        if 'row_idx' not in df.columns or 'col_name' not in df.columns or 'text' not in df.columns:
            logger.error("DataFrame missing required columns for pivot")
            return pd.DataFrame()

        # Get best prediction per cell first
        best_df = self.get_best_predictions_per_cell(df)

        # Pivot to create scoreboard structure
        try:
            scoreboard = best_df.pivot_table(
                index='row_idx',
                columns='col_name',
                values='text',
                aggfunc='first'
            ).reset_index()

            # Rename index to 'row'
            scoreboard.rename(columns={'row_idx': 'row'}, inplace=True)

            # Sort by row
            scoreboard = scoreboard.sort_values('row')

            logger.info(f"Created scoreboard table with {len(scoreboard)} rows")

            # Save if output path provided
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                scoreboard.to_csv(output_path, index=False)
                logger.info(f"Saved scoreboard table to {output_path}")

            return scoreboard

        except Exception as e:
            logger.error(f"Failed to create scoreboard table: {e}")
            return pd.DataFrame()
