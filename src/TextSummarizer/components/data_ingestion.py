import os
import zipfile
import urllib.request
from pathlib import Path
from src.TextSummarizer.logging import logger
from src.TextSummarizer.utils.common import get_size
from src.TextSummarizer.entity import DataIngestionConfig


class DataIngestion:
    """Handles downloading and extracting the summarization dataset."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """Downloads the dataset archive from source_URL if not already present.

        Returns:
            str: Path to the downloaded local archive file.
        """
        local_file = Path(self.config.local_data_file)
        local_file.parent.mkdir(parents=True, exist_ok=True)

        if not local_file.exists():
            logger.info(f"Downloading dataset from: {self.config.source_URL} ...")
            filename, headers = urllib.request.urlretrieve(
                url=self.config.source_URL,
                filename=str(local_file)
            )
            logger.info(f"Downloaded file '{filename}' with headers:\n{headers}")
        else:
            logger.info(f"Dataset archive already exists: '{local_file}' (Size: {get_size(local_file)})")

        return str(local_file)

    def extract_zip_file(self):
        """Extracts the zip archive into the configured unzip_dir directory."""
        unzip_path = Path(self.config.unzip_dir)
        unzip_path.mkdir(parents=True, exist_ok=True)
        local_file = Path(self.config.local_data_file)

        if not local_file.exists():
            raise FileNotFoundError(f"Cannot extract archive: {local_file} does not exist.")

        logger.info(f"Extracting '{local_file}' to '{unzip_path}' ...")
        with zipfile.ZipFile(str(local_file), "r") as zip_ref:
            zip_ref.extractall(str(unzip_path))
        logger.info(f"Extraction completed successfully at: {unzip_path}")
