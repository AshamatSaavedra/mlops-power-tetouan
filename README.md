# MLOps Power Consumption — Tetouan

Pipeline completo de Machine Learning y MLOps para modelar el consumo energético en tres zonas de la ciudad de Tetouan.  
Este proyecto implementa buenas prácticas de ingeniería, versionado de datos, experiment tracking, modularización de código y reproducibilidad usando **DVC**, **MLflow**, **Scikit-Learn**, **FastAPI**, **Docker**, y una arquitectura basada en **Cookiecutter Data Science**.

**Repositorio**: https://github.com/AshamatSaavedra/mlops-power-tetouan

---

## Objetivo del Proyecto

Construir un pipeline reproducible de extremo a extremo para:

- Preprocesamiento y generación de features  
- Entrenamiento y selección de modelos por zona  
- Registro y comparación de experimentos mediante MLflow  
- Versionado de datasets y modelos con DVC  
- Despliegue mediante FastAPI + Docker  
- Evaluación de *data drift* sin reentrenamiento  

---

## Arquitectura General del Proyecto

### 📦 Vista General del Pipeline
 ┌──────────────────┐
        │    Datos Raw      │
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Preprocesamiento │  dvc stage: preprocess
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Feature Engineering │  dvc stage: features
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │  Modelado por Zona │  dvc stage: modeling
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │   Métricas + MLflow │
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │   Predicción API  │  FastAPI + Docker
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │ Evaluación Drift │
        └──────────────────┘
## Estructura del Proyecto

Basada en *Cookiecutter Data Science*:

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_power_tetouan and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops_power_tetouan   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_power_tetouan a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
---

# Instalación

## 1. Clona el repo

git clone https://github.com/AshamatSaavedra/mlops-power-tetouan.git
cd mlops-power-tetouan

## 2. Instala dependencias (via Poetry)

Copy code
poetry install

## 3. Activa el entorno

Copy code
poetry shell
# Ejecución del Pipeline (DVC)

1. Preprocesamiento + Features
dvc repro preprocess
dvc repro features
2. Entrenamiento de los Modelos
dvc repro modeling
3. Pipeline completo

dvc repro

# MLflow UI
mlflow ui --backend-store-uri mlruns/
Abrir:
http://127.0.0.1:5000

Incluye:
MAE, RMSE, R² por modelo y zona
parámetros utilizados
artefactos (modelos .pkl)
comparaciones lado a lado
Modelos Entrenados
Por zona se entrenaron:
Linear Regression
RidgeCV
LassoCV
RandomForestRegressor (con GridSearchCV)
GradientBoostingRegressor

Resultados (Resumen)
Los mejores modelos en las tres zonas fueron:

✔ Random Forest (en todas las zonas)

Desempeño Final
Zona	MAE	RMSE	R²
Zone 1	973.33	1742.70	0.94
Zone 2	704.43	1419.04	0.93
Zone 3	841.54	2114.40	0.90

Los modelos lineales mostraron bajo desempeño (R² ~ 0.55–0.68), confirmando fuertemente la no linealidad del consumo energético.

# Fase de Data Drift
Se agregó un pipeline para evaluar el drift sin reentrenamiento, comparando:

MAE base vs MAE con drift
RMSE base vs RMSE con drift
R² base vs R² con drift

Cambios porcentuales
Ejemplo de resultados:

zone1:
  MAE_change_pct: 1.42%
  RMSE_change_pct: -0.05%

zone2:
  MAE_change_pct: 3.15%
  RMSE_change_pct: -0.06%

zone3:
  MAE_change_pct: 7.37%
  RMSE_change_pct: -0.08%
Esto permite detectar degradación temprana sin necesidad de reentrenar inmediatamente.

# API de Inferencia (FastAPI)
Ejecutar:
uvicorn mlops_power_tetouan.api.main:app --reload

Endpoint principal:
POST /predict
Ejemplo de request:

{
    "zone": "zone1",
    "data": {
        "DateTime": "2018-01-01 00:10:00",
        "Temperature": 6.4,
        "Humidity": 74.5,
        "Wind Speed": 0.083,
        "general diffuse flows": 0.07,
        "diffuse flows": 0.085,
        "mixed_type_col": 811
    }
}

Despliegue con Docker
Construir imagen:

docker build -t tetouan-api .
Ejecutar contenedor:

docker run -p 8000:8000 tetouan-api

# Conclusiones Principales
El pipeline es totalmente reproducible mediante DVC.

MLflow permite una gestión profesional de experimentos.

Las features temporales, cíclicas e interacciones mejoraron significativamente el rendimiento.

Random Forest fue el mejor modelo en todas las zonas.

Se agrega una fase robusta de detección de drift.