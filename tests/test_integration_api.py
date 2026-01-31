import pytest
from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


@pytest.mark.parametrize(
    "expected_status",
    [200],
)
def test_health_check(expected_status: int) -> None:
    """
    Verifica que el health check responde correctamente.
    """
    response = client.get("/health")

    assert response.status_code == expected_status
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


@pytest.mark.parametrize(
    "payload, expected_status",
    [
        (
            {
                "features": [
                    [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]
                ]
            },
            200,
        ),
        (
            {"features": [[1.0, 2.0]]},  # features inválidas
            400,
        ),
    ],
)
def test_predict_endpoint(payload: dict, expected_status: int) -> None:
    """
    Verifica inferencias válidas e inputs inválidos.
    """
    response = client.post("/predict", json=payload)

    assert response.status_code == expected_status

    if expected_status == 200:
        body = response.json()
        assert "predictions" in body
        assert isinstance(body["predictions"], list)