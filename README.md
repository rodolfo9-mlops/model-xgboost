# 🩺 Diabetes ML API

Proyecto de *Machine Learning end-to-end* que entrena un modelo de predicción de diabetes y lo expone mediante una API con *FastAPI*, siguiendo buenas prácticas de **MLOps, Docker y CI**.

---

## 🚀 ¿Qué hace este proyecto?

- 📊 Entrena un modelo de *XGBoost*
- 💾 Guarda el modelo entrenado en /model
- 🌐 Expone un endpoint de inferencia con *FastAPI*
- ❤️ Incluye *health check*
- 🐳 Se ejecuta en *Docker*
- 🤖 Tiene *CI con GitHub Actions*
- 🔒 Usa imágenes separadas para *dev* y *prod* (CMD vs ENTRYPOINT)

---

## 🧠 Arquitectura general

```text
.
├── api.py              # API FastAPI (inferencia + healthcheck)
├── train.py            # Entrenamiento del modelo
├── model/              # Modelos entrenados (artefactos)
├── tests/              # Tests unitarios e integrales
├── requirements.txt    # Dependencias Python
├── Dockerfile          # Multi-stage (base / prod)
├── docker-compose.yml  # Entorno local de desarrollo
└── .github/workflows   # CI con GitHub Actions


⸻

🐍 Entrenamiento del modelo
	•	El script train.py:
	•	Preprocesa los datos
	•	Entrena un modelo XGBoost
	•	Guarda el modelo en la carpeta model/

👉 La API entrena automáticamente si no existe un modelo.

⸻

🌐 API (FastAPI)

Endpoints principales

Método	Endpoint	Descripción
GET	/health	Health check
POST	/predict	Realiza una inferencia

La API carga el modelo desde /model al iniciar.

⸻

🐳 Docker (multi-stage)

Este proyecto usa un solo Dockerfile con dos stages:

🧰 base (dev / test)
	•	Usa CMD
	•	Flexible
	•	Permite pytest, bash, scripts
	•	Usado por Docker Compose y CI

🔒 prod (release)
	•	Usa ENTRYPOINT
	•	Imagen bloqueada
	•	Ideal para producción
	•	Se construye en PR hacia main

⸻

🧪 Tests
	•	Tests escritos con pytest
	•	Se ejecutan sin Docker en CI
	•	Bloquean merges a dev si fallan

pytest


⸻

🤖 CI con GitHub Actions

✅ PR hacia dev
	•	Corre pytest
	•	No usa Docker
	•	Bloquea el merge si falla

🏗️ PR hacia main
	•	Solo hace docker build --target prod
	•	Valida que la imagen de producción construye bien
	•	No hace deploy ni push

⸻

🧑‍💻 Desarrollo local

docker compose up --build

La API queda disponible en:

http://localhost:8000


⸻

❤️ Filosofía del proyecto
	•	🔹 Separación clara entre dev y prod
	•	🔹 Un solo Dockerfile, sin duplicación
	•	🔹 CI rápido y con propósito
	•	🔹 Infraestructura simple y explícita
	•	🔹 Fácil de extender a Cloud Run / GKE

⸻

✨ Próximos pasos (opcional)
	•	📈 Coverage con pytest
	•	🔐 Imagen sin root
	•	📦 Cache de Docker layers
	•	☁️ Deploy automático
	•	🔎 Observabilidad y logging estructurado