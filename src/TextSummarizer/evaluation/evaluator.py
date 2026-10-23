import os
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import nltk
from nltk.tokenize import sent_tokenize
from src.TextSummarizer.logging import logger

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


class Lead3Baseline:
    """Heuristic Extractive Baseline: Selects the first three sentences of each document.

    This serves as the standard academic benchmark (Nallapati et al., 2016) against which
    abstractive neural summarizers are traditionally evaluated.
    """

    def summarize(self, text: str) -> str:
        """Extracts the first 3 sentences from the provided text."""
        try:
            sentences = sent_tokenize(text.strip())
            return " ".join(sentences[:3])
        except Exception:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return " ".join(lines[:3]) if lines else text.strip()

    def batch_summarize(self, texts: List[str]) -> List[str]:
        return [self.summarize(t) for t in texts]


class SummarizationEvaluator:
    """Comprehensive, reproducible evaluation framework for abstractive summarization models.

    Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum against reference ground-truth
    summaries, benchmarks against Lead-3 baseline, measures inference latency, and
    exports structured reports in JSON and CSV.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._set_deterministic_seed(seed)
        self.rouge_metric, self.metric_framework = self._initialize_metric()

    def _set_deterministic_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _initialize_metric(self):
        try:
            import evaluate
            return evaluate.load("rouge"), "evaluate"
        except Exception:
            try:
                from datasets import load_metric
                return load_metric("rouge"), "datasets"
            except Exception as e:
                logger.warning(f"Could not load Hugging Face evaluate/datasets rouge metric: {e}")
                return None, "none"

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculates ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.

        Args:
            predictions (List[str]): List of generated summary strings.
            references (List[str]): List of reference gold-standard summary strings.

        Returns:
            Dict[str, float]: Dictionary mapping metric names to F1 scores.
        """
        if self.rouge_metric is None:
            # Fallback to direct rouge_score if available
            try:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
                accum = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
                n = max(1, len(predictions))
                for p, r in zip(predictions, references):
                    scores = scorer.score(r, p)
                    for k in accum:
                        accum[k] += scores[k].fmeasure
                return {k: round(v / n, 4) for k, v in accum.items()}
            except Exception as e:
                logger.error(f"ROUGE computation error: {e}")
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

        if self.metric_framework == "evaluate":
            raw = self.rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
            return {k: round(float(v), 4) for k, v in raw.items() if k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
        else:
            self.rouge_metric.add_batch(predictions=predictions, references=references)
            raw = self.rouge_metric.compute(use_stemmer=True)
            normalized = {}
            for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                if k in raw:
                    normalized[k] = round(float(raw[k].mid.fmeasure), 4)
            return normalized

    def evaluate_model_vs_baseline(
        self,
        dialogues: List[str],
        references: List[str],
        model_predictions: List[str],
        model_name: str = "PEGASUS-FineTuned",
        inference_time_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs comparative evaluation between the neural model and the Lead-3 baseline.

        Args:
            dialogues (List[str]): Source dialogues.
            references (List[str]): Reference summaries.
            model_predictions (List[str]): Predictions from the trained model.
            model_name (str): Identifier of the model.
            inference_time_sec (float, optional): Total wall-clock time in seconds.

        Returns:
            Dict[str, Any]: Structured evaluation report with metrics and sample outputs.
        """
        lead3 = Lead3Baseline()
        lead3_predictions = lead3.batch_summarize(dialogues)

        model_rouge = self.compute_rouge(model_predictions, references)
        lead3_rouge = self.compute_rouge(lead3_predictions, references)

        n_samples = len(dialogues)
        avg_latency_ms = (inference_time_sec / n_samples * 1000) if (inference_time_sec and n_samples > 0) else None

        report = {
            "evaluation_metadata": {
                "num_samples": n_samples,
                "random_seed": self.seed,
                "model_name": model_name,
                "total_inference_time_seconds": round(inference_time_sec, 2) if inference_time_sec else None,
                "avg_latency_ms_per_sample": round(avg_latency_ms, 2) if avg_latency_ms else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            },
            "metrics_comparison": {
                model_name: model_rouge,
                "Lead-3 (Baseline)": lead3_rouge
            },
            "sample_qualitative_outputs": [
                {
                    "dialogue": dialogues[i][:250] + ("..." if len(dialogues[i]) > 250 else ""),
                    "reference": references[i],
                    "model_summary": model_predictions[i],
                    "lead3_summary": lead3_predictions[i]
                }
                for i in range(min(3, n_samples))
            ]
        }
        return report

    def save_report(self, report: Dict[str, Any], output_dir: Path):
        """Saves evaluation report to JSON and CSV formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "evaluation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Evaluation JSON report saved to: {json_path}")

        # Flatten metrics to CSV
        metrics = report["metrics_comparison"]
        df = pd.DataFrame.from_dict(metrics, orient="index")
        csv_path = output_dir / "metrics_summary.csv"
        df.to_csv(csv_path)
        logger.info(f"Evaluation metrics CSV saved to: {csv_path}")
