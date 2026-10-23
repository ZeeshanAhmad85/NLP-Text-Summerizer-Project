import os
import sys
import tempfile
from pathlib import Path
import pytest
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_yaml(temp_dir):
    config_dict = {
        "artifacts_root": str(temp_dir / "artifacts"),
        "data_ingestion": {
            "root_dir": str(temp_dir / "artifacts" / "data_ingestion"),
            "source_URL": "https://example.com/data.zip",
            "local_data_file": str(temp_dir / "artifacts" / "data_ingestion" / "data.zip"),
            "unzip_dir": str(temp_dir / "artifacts" / "data_ingestion")
        },
        "data_validation": {
            "root_dir": str(temp_dir / "artifacts" / "data_validation"),
            "STATUS_FILE": str(temp_dir / "artifacts" / "data_validation" / "status.txt"),
            "ALL_REQUIRED_FILES": ["train", "test", "validation"]
        },
        "data_transformation": {
            "root_dir": str(temp_dir / "artifacts" / "data_transformation"),
            "data_path": str(temp_dir / "artifacts" / "data_ingestion" / "samsum_dataset"),
            "tokenizer_name": "google/pegasus-cnn_dailymail"
        },
        "model_trainer": {
            "root_dir": str(temp_dir / "artifacts" / "model_trainer"),
            "data_path": str(temp_dir / "artifacts" / "data_transformation" / "samsum_dataset"),
            "model_ckpt": "google/pegasus-cnn_dailymail"
        },
        "model_evaluation": {
            "root_dir": str(temp_dir / "artifacts" / "model_evaluation"),
            "data_path": str(temp_dir / "artifacts" / "data_transformation" / "samsum_dataset"),
            "model_path": str(temp_dir / "artifacts" / "model_trainer" / "pegasus-samsum-model"),
            "tokenizer_path": str(temp_dir / "artifacts" / "model_trainer" / "tokenizer"),
            "metric_file_name": str(temp_dir / "artifacts" / "model_evaluation" / "metrics.csv")
        }
    }
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
    return config_path


@pytest.fixture
def sample_params_yaml(temp_dir):
    params_dict = {
        "TrainingArguments": {
            "num_train_epochs": 1,
            "warmup_steps": 100,
            "per_device_train_batch_size": 1,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 50,
            "save_steps": 500,
            "gradient_accumulation_steps": 4
        }
    }
    params_path = temp_dir / "params.yaml"
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.dump(params_dict, f)
    return params_path
