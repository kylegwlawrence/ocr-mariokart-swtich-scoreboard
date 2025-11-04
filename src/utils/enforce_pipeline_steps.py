"""Migration script to enforce all preprocessing steps in existing prediction CSVs.

Updates runtime_config in all prediction CSV files to include all 13 preprocessing
methods, with empty parameters for steps that weren't used.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _load_pipeline_constants() -> Dict[str, Any]:
    """Load pipeline constants from JSON file.

    Returns:
        Dictionary containing all preprocessing methods and their parameters.
    """
    constants_path = Path(__file__).parent.parent.parent / "data" / "pipeline_constants.json"
    try:
        with open(constants_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Pipeline constants file not found at {constants_path}")
        raise


def _get_complete_preprocessing_config(preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure preprocessing config includes all possible preprocessing methods.

    Takes the actual preprocessing config with only used steps and returns
    a complete config with all possible methods, setting unused ones to empty params.

    Args:
        preprocessing_config: The actual preprocessing config with used steps.

    Returns:
        Complete preprocessing config with all methods.
    """
    constants = _load_pipeline_constants()
    template_methods = constants.get("preprocessing_methods", {})

    # Start with a copy of the config
    complete_config = preprocessing_config.copy()

    # Build a map of actual steps by method name for quick lookup
    actual_steps = {}
    if "steps" in preprocessing_config and isinstance(preprocessing_config["steps"], list):
        for step in preprocessing_config["steps"]:
            if isinstance(step, dict) and "method" in step:
                actual_steps[step["method"]] = step

    # Create complete steps list with all methods
    complete_steps = []
    for method_name, method_template in template_methods.items():
        if method_name in actual_steps:
            # Use actual step with real parameters
            step = actual_steps[method_name].copy()
            step["used"] = True
            complete_steps.append(step)
        else:
            # Use template with empty parameters (enforced but not used)
            complete_steps.append({
                "method": method_name,
                "params": method_template.get("params", {}),
                "used": False
            })

    # Update the config with complete steps
    complete_config["steps"] = complete_steps
    return complete_config


def migrate_csv_file(csv_path: Path, backup: bool = True) -> Dict[str, Any]:
    """Migrate a single CSV file to enforce all preprocessing steps.

    Args:
        csv_path: Path to the CSV file to migrate.
        backup: If True, creates a backup of the original file.

    Returns:
        Dictionary with migration results.
    """
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return {"success": False, "error": "File not found"}

    # Create backup if requested
    if backup:
        backup_path = csv_path.with_stem(csv_path.stem + "_backup")
        try:
            with open(csv_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup for {csv_path}: {e}")
            return {"success": False, "error": f"Backup failed: {e}"}

    # Read the CSV file
    rows = []
    fieldnames = None
    updated_count = 0

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            for row in reader:
                if 'runtime_config' in row:
                    try:
                        runtime_config = json.loads(row['runtime_config'])

                        # Update preprocessing_config if present
                        if 'preprocessing_config' in runtime_config:
                            preprocessing_config = runtime_config['preprocessing_config']
                            complete_config = _get_complete_preprocessing_config(preprocessing_config)
                            runtime_config['preprocessing_config'] = complete_config
                            updated_count += 1

                        # Re-serialize to JSON
                        row['runtime_config'] = json.dumps(runtime_config)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipped row with invalid JSON in runtime_config: {e}")
                        continue

                rows.append(row)

    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return {"success": False, "error": f"Read failed: {e}"}

    # Write the updated CSV file
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Updated {csv_path}: {updated_count} rows migrated")
        return {
            "success": True,
            "file": str(csv_path),
            "rows_updated": updated_count,
            "total_rows": len(rows)
        }

    except Exception as e:
        logger.error(f"Error writing CSV file {csv_path}: {e}")
        return {"success": False, "error": f"Write failed: {e}"}


def migrate_all_prediction_files(data_dir: Path = None, backup: bool = True) -> List[Dict[str, Any]]:
    """Migrate all CSV files in the predictions data directory.

    Args:
        data_dir: Path to the predictions data directory. Defaults to outputs/predictions/data.
        backup: If True, creates backups of original files.

    Returns:
        List of migration results for each file.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "outputs" / "predictions" / "data"

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []

    # Find all CSV files
    csv_files = sorted(data_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

    results = []
    successful = 0
    failed = 0

    for csv_file in csv_files:
        logger.info(f"Migrating {csv_file.name}...")
        result = migrate_csv_file(csv_file, backup=backup)
        results.append(result)

        if result["success"]:
            successful += 1
        else:
            failed += 1

    # Summary
    logger.info("=" * 60)
    logger.info(f"Migration Summary: {successful} successful, {failed} failed")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import sys

    # Allow custom data directory via command line argument
    data_dir = None
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    results = migrate_all_prediction_files(data_dir=data_dir, backup=True)

    # Print summary
    for result in results:
        if result["success"]:
            print(f"✓ {result['file']}: {result['rows_updated']}/{result['total_rows']} rows updated")
        else:
            print(f"✗ {result.get('file', 'Unknown')}: {result['error']}")
