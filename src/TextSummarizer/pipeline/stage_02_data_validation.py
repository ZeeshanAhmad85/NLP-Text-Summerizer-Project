from src.TextSummarizer.config.configuration import ConfigurationManager
from src.TextSummarizer.components.data_validation import DataValidation
from src.TextSummarizer.logging import logger

STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self) -> bool:
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        is_valid = data_validation.validate_all_files_exist()
        if not is_valid:
            raise ValueError("Data validation failed: Required dataset files are missing.")
        return is_valid


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
