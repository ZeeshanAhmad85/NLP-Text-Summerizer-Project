import os
from pathlib import Path
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk
from src.TextSummarizer.logging import logger
from src.TextSummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    """Orchestrates fine-tuning of the Seq2Seq summarization model using Hugging Face Trainer."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """Executes the training process and persists trained model weights and tokenizer."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model training device selected: {device}")

        logger.info(f"Loading tokenizer and model checkpoint: {self.config.model_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.model_ckpt))
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_ckpt)).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        logger.info(f"Loading tokenized dataset from: {self.config.data_path}")
        dataset_samsum_pt = load_from_disk(str(self.config.data_path))

        training_args_dict = {
            "output_dir": str(self.config.root_dir),
            "num_train_epochs": self.config.num_train_epochs,
            "warmup_steps": self.config.warmup_steps,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_train_batch_size,
            "weight_decay": self.config.weight_decay,
            "logging_steps": self.config.logging_steps,
            "eval_steps": self.config.eval_steps,
            "save_steps": int(self.config.save_steps),
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "fp16": torch.cuda.is_available(),
            "report_to": "none"
        }

        # Handle deprecation: evaluation_strategy -> eval_strategy in modern transformers
        try:
            trainer_args = TrainingArguments(
                evaluation_strategy=self.config.evaluation_strategy,
                **training_args_dict
            )
        except (TypeError, ValueError):
            trainer_args = TrainingArguments(
                eval_strategy=self.config.evaluation_strategy,
                **training_args_dict
            )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        logger.info("Starting model training loop...")
        trainer.train()

        model_save_dir = Path(self.config.root_dir) / "pegasus-samsum-model"
        tokenizer_save_dir = Path(self.config.root_dir) / "tokenizer"

        model_save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving fine-tuned model checkpoint to: {model_save_dir}")
        model_pegasus.save_pretrained(str(model_save_dir))

        logger.info(f"Saving tokenizer to: {tokenizer_save_dir}")
        tokenizer.save_pretrained(str(tokenizer_save_dir))
