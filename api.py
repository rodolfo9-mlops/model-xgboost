from _future_ import annotations

import os
from typing import List

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from train import DiabetesXGBoostTrainer


MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")


class PredictionRequest(BaseModel):
    """Esquema de entrada para inferencia."""
    features: List[List[float]]


class PredictionResponse(BaseModel):
    """Esquema de salida para inferencia."""
    predictions: List[float]


class HealthResponse(BaseModel):
    """Respuesta del health check."""
    status: str
    model_loaded: bool


class DiabetesInferenceService:
    """
    Servicio de inferencia para el modelo de diabetes.
    """

    def _init_(self, model_path: str) -> None:
        """
        Inicializa el servicio.

        Args:
            model_path: Ruta al artefacto del modelo.
        """
        self.model_path: str = model_path
        self.model: XGBRegressor | None = None
        self.scaler: StandardScaler | None = None

        self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        """
        Carga el modelo entrenado o entrena uno nuevo si no existe.
        """
        if not os.path.exists(self.model_path):
            trainer = DiabetesXGBoostTrainer(model_dir=MODEL_DIR)
            trainer.train()

        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]

    def is_ready(self) -> bool:
        """
        Indica si el servicio está listo para inferencia.

        Returns:
            True si el modelo y el scaler están cargados.
        """
        return self.model is not None and self.scaler is not None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza inferencias.

        Args:
            X: Matriz de features.

        Returns:
            Predicciones del modelo.

        Raises:
            RuntimeError: Si el modelo no está cargado.
        """
        if not self.is_ready():
            raise RuntimeError("Modelo no cargado.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# ---------- FastAPI ----------

app = FastAPI(
    title="Diabetes XGBoost Inference API",
    version="1.0.0",
)

inference_service = DiabetesInferenceService(model_path=MODEL_PATH)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check del servicio.

    Returns:
        Estado del servicio y del modelo.
    """
    return HealthResponse(
        status="ok",
        model_loaded=inference_service.is_ready(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Endpoint de inferencia.

    Args:
        request: Features de entrada.

    Returns:
        Predicciones del modelo.
    """
    try:
        X = np.array(request.features, dtype=float)

        if X.ndim != 2 or X.shape[1] != 10:
            raise ValueError("Cada registro debe tener exactamente 10 features.")

        predictions = inference_service.predict(X)

        return PredictionResponse(predictions=predictions.tolist())

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc