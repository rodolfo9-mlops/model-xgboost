from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


class DiabetesXGBoostTrainer:
    """
    Entrenador de un modelo XGBoost para el dataset Diabetes.

    Responsabilidad única:
    - Preprocesar los datos
    - Entrenar el modelo
    - Persistir el artefacto entrenado en disco
    """

    def __init__(
        self,
        model_dir: str = "model",
        random_state: int = 42,
    ) -> None:
        """
        Inicializa el entrenador.

        Args:
            model_dir: Directorio donde se guardará el modelo entrenado.
            random_state: Semilla para reproducibilidad.
        """
        self.model_dir: str = model_dir
        self.random_state: int = random_state

        self.scaler: StandardScaler = StandardScaler()
        self.model: XGBRegressor = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
        )

    def preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesa los datos de entrada.

        Actualmente:
        - Escala las variables numéricas usando StandardScaler.

        Args:
            X: Matriz de features.
            y: Vector target.

        Returns:
            Tuple con X transformado y y sin modificar.
        """
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def train(self) -> None:
        """
        Ejecuta el flujo completo de entrenamiento y guarda el modelo.

        Flujo:
        1. Carga datos
        2. Preprocesa
        3. Entrena modelo
        4. Guarda artefactos en ./model
        """
        # Cargar dataset
        dataset = load_diabetes()
        X: np.ndarray = dataset.data
        y: np.ndarray = dataset.target

        # Preprocesamiento
        X_processed, y_processed = self.preprocess(X, y)

        # Split (solo para entrenamiento; no evaluación)
        X_train, _, y_train, _ = train_test_split(
            X_processed,
            y_processed,
            test_size=0.2,
            random_state=self.random_state,
        )

        # Entrenamiento
        self.model.fit(X_train, y_train)

        # Persistencia
        os.makedirs(self.model_dir, exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
            },
            os.path.join(self.model_dir, "model.joblib"),
        )


if __name__ == "__main__":
    trainer = DiabetesXGBoostTrainer()
    trainer.train()