import pytest
from src.TextSummarizer.pipeline.prediction import PredictionPipeline


def test_validate_input_valid():
    pipeline = PredictionPipeline()
    valid_text = "John: Hello Mary, are we meeting at five? Mary: Yes, see you soon!"
    cleaned = pipeline.validate_input(valid_text)
    assert cleaned == valid_text


def test_validate_input_none():
    pipeline = PredictionPipeline()
    with pytest.raises(TypeError):
        pipeline.validate_input(None)


def test_validate_input_non_string():
    pipeline = PredictionPipeline()
    with pytest.raises(TypeError):
        pipeline.validate_input(12345)


def test_validate_input_empty():
    pipeline = PredictionPipeline()
    with pytest.raises(ValueError):
        pipeline.validate_input("")


def test_validate_input_whitespace():
    pipeline = PredictionPipeline()
    with pytest.raises(ValueError):
        pipeline.validate_input("   \n\t  ")


def test_validate_input_too_short():
    pipeline = PredictionPipeline()
    with pytest.raises(ValueError):
        pipeline.validate_input("Hi there")
