from pathlib import Path
from src.TextSummarizer.config.configuration import ConfigurationManager


def test_configuration_manager_initialization(sample_config_yaml, sample_params_yaml):
    cm = ConfigurationManager(
        config_filepath=sample_config_yaml,
        params_filepath=sample_params_yaml
    )
    assert cm.config is not None
    assert cm.params is not None


def test_get_data_ingestion_config(sample_config_yaml, sample_params_yaml):
    cm = ConfigurationManager(sample_config_yaml, sample_params_yaml)
    cfg = cm.get_data_ingestion_config()
    assert cfg.source_URL == "https://example.com/data.zip"
    assert isinstance(cfg.local_data_file, Path)


def test_get_data_validation_config(sample_config_yaml, sample_params_yaml):
    cm = ConfigurationManager(sample_config_yaml, sample_params_yaml)
    cfg = cm.get_data_validation_config()
    assert "train" in cfg.ALL_REQUIRED_FILES
    assert cfg.STATUS_FILE.endswith("status.txt")


def test_get_model_trainer_config(sample_config_yaml, sample_params_yaml):
    cm = ConfigurationManager(sample_config_yaml, sample_params_yaml)
    cfg = cm.get_model_trainer_config()
    assert cfg.num_train_epochs == 1
    assert cfg.eval_steps == 50
    assert isinstance(cfg.eval_steps, int)
    assert cfg.save_steps == 500.0


def test_get_model_evaluation_config(sample_config_yaml, sample_params_yaml):
    cm = ConfigurationManager(sample_config_yaml, sample_params_yaml)
    cfg = cm.get_model_evaluation_config()
    assert isinstance(cfg.model_path, Path)
    assert isinstance(cfg.metric_file_name, Path)
