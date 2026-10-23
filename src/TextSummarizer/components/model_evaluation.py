import os
import json
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from src.TextSummarizer.logging import logger
from src.TextSummarizer.entity import ModelEvaluationConfig


def generate_batch_sized_chunks(list_of_elements: list, batch_size: int):
    """Yields successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


class ModelEvaluation:
    """Evaluates the fine-tuned Seq2Seq summarization model on test dialogue datasets using ROUGE metrics."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _load_metric(self):
        """Loads ROUGE metric safely using evaluate or fallback to datasets.load_metric."""
        try:
            import evaluate
            return evaluate.load("rouge"), "evaluate"
        except Exception:
            from datasets import load_metric
            return load_metric("rouge"), "datasets"

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        metric_source: str,
        model,
        tokenizer,
        batch_size: int = 2,
        device: str = "cpu",
        column_text: str = "dialogue",
        column_summary: str = "summary"
    ) -> dict:
        """Computes ROUGE scores across batches of test samples."""
        article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

        logger.info(f"Evaluating {len(dataset[column_text])} samples in {len(article_batches)} batches...")
        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            with torch.no_grad():
                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    length_penalty=0.8,
                    num_beams=4,
                    max_length=128
                )

            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for s in summaries
            ]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        raw_score = metric.compute()

        # Normalize score format across library versions
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        normalized_scores = {}
        for rn in rouge_names:
            if rn in raw_score:
                val = raw_score[rn]
                # If old datasets score object with .mid.fmeasure
                if hasattr(val, "mid"):
                    normalized_scores[rn] = float(val.mid.fmeasure)
                elif isinstance(val, (float, int)):
                    normalized_scores[rn] = float(val)
                else:
                    normalized_scores[rn] = float(val)

        return normalized_scores

    def evaluate(self, sample_limit: int = None):
        """Runs evaluation on test dataset and persists metrics to CSV and JSON.

        Args:
            sample_limit (int, optional): Optional limit on number of test samples (useful for quick validation).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model evaluation device: {device}")

        logger.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))

        logger.info(f"Loading model from: {self.config.model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path)).to(device)

        logger.info(f"Loading test dataset from: {self.config.data_path}")
        dataset_samsum_pt = load_from_disk(str(self.config.data_path))
        test_data = dataset_samsum_pt["test"]

        if sample_limit and sample_limit < len(test_data):
            logger.info(f"Subsampling evaluation dataset to {sample_limit} examples.")
            test_data = test_data.select(range(sample_limit))

        metric, metric_source = self._load_metric()

        scores = self.calculate_metric_on_test_ds(
            dataset=test_data,
            metric=metric,
            metric_source=metric_source,
            model=model,
            tokenizer=tokenizer,
            batch_size=2,
            device=device,
            column_text="dialogue",
            column_summary="summary"
        )

        metric_path = Path(self.config.metric_file_name)
        metric_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([scores], index=["pegasus"])
        df.to_csv(str(metric_path), index=False)
        logger.info(f"Evaluation metrics saved to CSV: {metric_path}")

        json_path = metric_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        logger.info(f"Evaluation metrics saved to JSON: {json_path}")

        return scores
