import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_from_disk
from src.TextSummarizer.logging import logger
from src.TextSummarizer.entity import DataTransformationConfig


class DataTransformation:
    """Tokenizes and transforms the raw dialogue dataset into tensor features for Seq2Seq training."""

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(str(config.tokenizer_name))

    def convert_examples_to_features(self, example_batch):
        """Encodes dialogue inputs and summary targets into tokenized features.

        Args:
            example_batch (dict): Batch containing 'dialogue' and 'summary' lists.

        Returns:
            dict: Encoded input_ids, attention_mask, and target labels.
        """
        input_encodings = self.tokenizer(
            example_batch["dialogue"],
            max_length=1024,
            truncation=True,
            padding=False
        )

        # Handle both modern and older versions of transformers tokenizer API
        try:
            target_encodings = self.tokenizer(
                text_target=example_batch["summary"],
                max_length=128,
                truncation=True,
                padding=False
            )
        except TypeError:
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(
                    example_batch["summary"],
                    max_length=128,
                    truncation=True,
                    padding=False
                )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]
        }

    def convert(self):
        """Loads dataset from disk, maps feature conversion, and saves tokenized dataset."""
        logger.info(f"Loading raw dataset from: {self.config.data_path}")
        dataset_samsum = load_from_disk(str(self.config.data_path))

        logger.info("Tokenizing and mapping dataset to Seq2Seq features...")
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)

        save_path = Path(self.config.root_dir) / "samsum_dataset"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_samsum_pt.save_to_disk(str(save_path))
        logger.info(f"Transformed dataset saved to: {save_path}")
