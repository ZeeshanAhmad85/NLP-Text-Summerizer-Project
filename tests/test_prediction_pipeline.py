from unittest.mock import MagicMock
import torch
from src.TextSummarizer.pipeline.prediction import PredictionPipeline


def test_prediction_pipeline_mocked_inference():
    pipeline = PredictionPipeline()

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Mock tokenizer output
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102, 103, 104]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]])
    }
    mock_model.generate.return_value = torch.tensor([[201, 202, 203]])
    mock_tokenizer.decode.return_value = "This is a concise mock summary."

    pipeline._tokenizer = mock_tokenizer
    pipeline._model = mock_model

    text = "Amanda: Can you share Betty's number? Hannah: Yes, here it is 555-0199."
    result = pipeline.predict(text, max_length=64, min_length=5, num_beams=2)

    assert result["summary"] == "This is a concise mock summary."
    assert result["input_tokens"] == 4
    assert result["output_tokens"] == 3
    assert "device" in result
    mock_model.generate.assert_called_once()
