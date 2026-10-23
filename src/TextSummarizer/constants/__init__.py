from pathlib import Path

# Compute project root based on this file's location: src/TextSummarizer/constants/__init__.py
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent.parent

_CONFIG_PATH_DEFAULT = Path("config/config.yaml")
_PARAMS_PATH_DEFAULT = Path("params.yaml")

CONFIG_FILE_PATH = _CONFIG_PATH_DEFAULT if _CONFIG_PATH_DEFAULT.exists() else (_PROJECT_ROOT / "config" / "config.yaml")
PARAMS_FILE_PATH = _PARAMS_PATH_DEFAULT if _PARAMS_PATH_DEFAULT.exists() else (_PROJECT_ROOT / "params.yaml")