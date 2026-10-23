import os
import json
from pathlib import Path
from typing import Any, Union
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from src.TextSummarizer.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns contents wrapped in a ConfigBox.

    Args:
        path_to_yaml (Path): Path to YAML file.

    Raises:
        ValueError: If YAML file is empty or content is None.
        FileNotFoundError: If the file does not exist.
        Exception: For any parsing or I/O error.

    Returns:
        ConfigBox: ConfigBox containing the YAML data.
    """
    if not path_to_yaml.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path_to_yaml}")

    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file '{path_to_yaml}' is empty.")
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"YAML file '{path_to_yaml}' is empty or invalid.")
    except Exception as e:
        logger.error(f"Error loading YAML from {path_to_yaml}: {e}")
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Creates a list of directories if they do not already exist.

    Args:
        path_to_directories (list): List of directory paths (str or Path).
        verbose (bool, optional): Whether to log creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(str(path), exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """Computes file size in kilobytes.

    Args:
        path (Path): Path to the target file.

    Returns:
        str: Human-readable file size in KB.
    """
    if not path.exists():
        return "0 KB"
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"{size_in_kb} KB"


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves dictionary data to a JSON file.

    Args:
        path (Path): Destination file path.
        data (dict): Dictionary to serialize.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved successfully at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads JSON data from file and returns ConfigBox.

    Args:
        path (Path): JSON file path.

    Returns:
        ConfigBox: ConfigBox containing JSON data.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)