from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "NLP-Text-Summarizer"
    assert "device" in data


def test_ui_index_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PEGASUS Text Summarizer" in response.text


def test_predict_validation_error_short_text():
    response = client.post("/predict", json={"text": "Hi"})
    assert response.status_code == 422 or response.status_code == 400


def test_predict_validation_error_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422 or response.status_code == 400


def test_predict_endpoint_success():
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = {
        "summary": "Mocked test summary outcome.",
        "input_tokens": 15,
        "output_tokens": 5,
        "device": "cpu"
    }

    with patch("app.get_prediction_pipeline", return_value=mock_pipeline):
        response = client.post(
            "/predict",
            json={
                "text": "Hannah: Hey, do you have Betty's contact info? Amanda: Sure, let me send it over right away.",
                "max_length": 64,
                "min_length": 5,
                "num_beams": 2
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "Mocked test summary outcome."
        assert data["input_tokens"] == 15
        assert data["output_tokens"] == 5
        assert data["device"] == "cpu"
