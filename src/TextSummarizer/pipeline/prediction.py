from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.TextSummarizer.logging import logger
from src.TextSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    """Production inference pipeline for generating abstractive text summaries."""

    DEFAULT_FALLBACK_MODEL = "google/pegasus-cnn_dailymail"

    def __init__(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        """Initializes the prediction pipeline with specified or auto-discovered model and tokenizer paths."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None

        try:
            config = ConfigurationManager()
            eval_config = config.get_model_evaluation_config()
            configured_model = Path(eval_config.model_path)
            configured_tokenizer = Path(eval_config.tokenizer_path)
        except Exception:
            configured_model = Path("artifacts/model_trainer/pegasus-samsum-model")
            configured_tokenizer = Path("artifacts/model_trainer/tokenizer")

        self.model_path = model_path or (str(configured_model) if configured_model.exists() else self.DEFAULT_FALLBACK_MODEL)
        self.tokenizer_path = tokenizer_path or (str(configured_tokenizer) if configured_tokenizer.exists() else self.DEFAULT_FALLBACK_MODEL)

    def _load_model_and_tokenizer(self):
        """Lazy loader for model and tokenizer to optimize memory and startup latency."""
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer from: {self.tokenizer_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        if self._model is None:
            logger.info(f"Loading model checkpoint from: {self.model_path} onto {self.device}")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            self._model.eval()

    def validate_input(self, text: Any) -> str:
        """Validates and sanitizes input text for summarization.

        Args:
            text (Any): Input text to validate.

        Raises:
            TypeError: If input is not a string.
            ValueError: If input is empty, whitespace only, or too short.

        Returns:
            str: Stripped and sanitized text.
        """
        if text is None or not isinstance(text, str):
            raise TypeError(f"Input text must be a string. Received: {type(text).__name__}")

        sanitized = text.strip()
        if not sanitized:
            raise ValueError("Input text cannot be empty or solely whitespace.")

        if len(sanitized.split()) < 3:
            raise ValueError("Input text is too short to summarize (minimum 3 words required).")

        return sanitized

    def predict(
        self,
        text: str,
        max_length: int = 128,
        min_length: int = 15,
        num_beams: int = 4,
        length_penalty: float = 0.8
    ) -> Dict[str, Any]:
        """Generates an abstractive summary for the provided input text.

        Args:
            text (str): Input text or dialogue to summarize.
            max_length (int, optional): Maximum generated summary token length. Defaults to 128.
            min_length (int, optional): Minimum generated summary token length. Defaults to 15.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            length_penalty (float, optional): Length penalty exponent. Defaults to 0.8.

        Returns:
            Dict[str, Any]: Dictionary containing 'summary', 'input_tokens', 'output_tokens', and 'device'.
        """
        clean_text = self.validate_input(text)
        self._load_model_and_tokenizer()

        # Tokenize with safety truncation to handle long context up to 1024 tokens
        inputs = self._tokenizer(
            clean_text,
            max_length=1024,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_token_count = int(input_ids.shape[1])

        # Ensure generation constraints are logical
        adjusted_min_len = min(min_length, max_length - 5) if max_length > 5 else 1

        with torch.no_grad():
            output_tokens = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=max(1, adjusted_min_len),
                num_beams=max(1, num_beams),
                length_penalty=length_penalty,
                early_stopping=True
            )

        summary = self._tokenizer.decode(
            output_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        output_token_count = int(output_tokens.shape[1])

        return {
            "summary": summary,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "device": self.device
        }
