import numpy as np
import pytest

from train import DiabetesXGBoostTrainer


@pytest.fixture(scope="module")
def trainer() -> DiabetesXGBoostTrainer:
    """Fixture que inicializa el trainer."""
    return DiabetesXGBoostTrainer(model_dir="model_test")


@pytest.mark.parametrize(
    "n_samples",
    [1, 10, 50],
)
def test_preprocess_shape(trainer: DiabetesXGBoostTrainer, n_samples: int) -> None:
    """
    Verifica que el preprocesamiento no altere la dimensionalidad.
    """
    X = np.random.rand(n_samples, 10)
    y = np.random.rand(n_samples)

    X_processed, y_processed = trainer.preprocess(X, y)

    assert X_processed.shape == (n_samples, 10)
    assert y_processed.shape == (n_samples,)


@pytest.mark.parametrize(
    "n_samples",
    [5, 20],
)
def test_training_creates_model(trainer: DiabetesXGBoostTrainer, n_samples: int) -> None:
    """
    Verifica que el entrenamiento genera un modelo entrenado.
    """
    X = np.random.rand(n_samples, 10)
    y = np.random.rand(n_samples)

    X_processed, y_processed = trainer.preprocess(X, y)
    trainer.model.fit(X_processed, y_processed)

    assert trainer.model is not None