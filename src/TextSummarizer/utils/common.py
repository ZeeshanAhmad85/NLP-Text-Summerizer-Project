
import os 
from box.exceptions import BoxValueError
import yaml
from src.TextSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    """ read yaml and returns
    
    Args:
        path_to_yaml(str): path like input

    Raises:
        BoxValue: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
           
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            logger.info(f"yaml file loaded successfully from {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    



@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    """create directories

    Args:
        path_to_directories (list): list of directories
        verbose (bool, optional): ignore if multiple directories is to be created. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")    



@ensure_annotations
def get_size(path:Path)->str:
    """get size in kb

    Args:
        path (Path): path to file

    Returns:
        str: size in kb
    """
    size_in_kb=round(os.path.getsize(path)/1024)

    return f"{size_in_kb} KB"         

    