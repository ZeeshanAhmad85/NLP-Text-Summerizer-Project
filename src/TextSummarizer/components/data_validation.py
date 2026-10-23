import os
from pathlib import Path
from src.TextSummarizer.logging import logger
from src.TextSummarizer.entity import DataValidationConfig


class DataValidation:
    """Validates the structure and integrity of the extracted dataset."""

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """Checks if all required dataset files or subdirectories exist.

        Writes verification outcome to the STATUS_FILE.

        Returns:
            bool: True if all required files exist, False otherwise.
        """
        try:
            validation_status = True
            dataset_dir = Path("artifacts/data_ingestion/samsum_dataset")
            
            # If dataset directory does not exist, check fallback inside artifacts
            if not dataset_dir.exists():
                logger.warning(f"Dataset directory not found at {dataset_dir}. Checking artifacts root...")
                potential_dirs = list(Path("artifacts").glob("**/samsum_dataset"))
                if potential_dirs:
                    dataset_dir = potential_dirs[0]

            status_path = Path(self.config.STATUS_FILE)
            status_path.parent.mkdir(parents=True, exist_ok=True)

            if not dataset_dir.exists():
                validation_status = False
                with open(status_path, "w", encoding="utf-8") as f:
                    f.write(f"Validation status: {validation_status}\nMissing: {dataset_dir}")
                logger.error(f"Dataset root '{dataset_dir}' does not exist.")
                return False

            existing_files = os.listdir(dataset_dir)
            missing_files = []

            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in existing_files:
                    validation_status = False
                    missing_files.append(required_file)

            with open(status_path, "w", encoding="utf-8") as f:
                f.write(f"Validation status: {validation_status}\n")
                if missing_files:
                    f.write(f"Missing files: {missing_files}\n")

            if validation_status:
                logger.info(f"Data validation passed. All required files present: {self.config.ALL_REQUIRED_FILES}")
            else:
                logger.warning(f"Data validation failed. Missing files: {missing_files}")

            return validation_status

        except Exception as e:
            logger.error(f"Exception during data validation: {e}")
            raise e
