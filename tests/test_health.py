from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "missing_weights", "model_error"}
    assert "detector_weights" in payload
    assert "recognizer_model_dir" in payload
    assert "recognizer_missing_files" in payload
