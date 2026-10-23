from pathlib import Path
from src.TextSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.TextSummarizer.utils.common import read_yaml, create_directories
from src.TextSummarizer.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)


class ConfigurationManager:
    """Manages reading configuration settings and hyperparameters to instantiate stage-specific dataclasses."""

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=str(config.source_URL),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=str(config.STATUS_FILE),
            ALL_REQUIRED_FILES=list(config.ALL_REQUIRED_FILES)
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            tokenizer_name=Path(config.tokenizer_name)
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=Path(config.model_ckpt),
            num_train_epochs=int(params.num_train_epochs),
            warmup_steps=int(params.warmup_steps),
            per_device_train_batch_size=int(params.per_device_train_batch_size),
            weight_decay=float(params.weight_decay),
            logging_steps=int(params.logging_steps),
            evaluation_strategy=str(params.evaluation_strategy),
            eval_steps=int(params.eval_steps),
            save_steps=float(params.save_steps),
            gradient_accumulation_steps=int(params.gradient_accumulation_steps)
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            metric_file_name=Path(config.metric_file_name)
        )