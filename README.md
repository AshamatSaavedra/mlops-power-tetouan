# MLOps Power Consumption — Tetouan  

Pipeline completo de Machine Learning y MLOps para modelar el consumo energético en tres zonas de la ciudad de Tetouan.  
Este proyecto implementa buenas prácticas de ingeniería, versionado de datos, experiment tracking, modularización de código y reproducibilidad usando **DVC**, **MLflow**, **Scikit-Learn** y una arquitectura basada en **Cookiecutter Data Science**.

---

## Objetivo del Proyecto

Construir un pipeline reproducible de extremo a extremo para:

- Preprocesamiento y generación de features
- Entrenamiento y selección de modelos por zona
- Registro y comparación de experimentos
- Versionado de datasets y modelos
- Reproducibilidad total vía DVC

---

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

## Instalación

## Clona el repo:

git clone https://github.com/usuario/mlops_power_tetouan.git
cd mlops_power_tetouan

Instala dependencias (via Poetry):

poetry install

Activa el entorno:

poetry shell

## Ejecución del Pipeline (DVC)
1. Preprocesamiento + Generación de Features

Genera scaled.csv y guarda el pipeline de features:

dvc repro preprocess

2. Entrenamiento de Modelos

Entrena los 5 modelos para cada zona y registra todos los experimentos en MLflow:

dvc repro modeling

3. Pipeline completo
dvc repro

## MLflow UI

Para visualizar modelos, métricas y comparaciones:

mlflow ui --backend-store-uri mlruns/


Luego abre en el navegador:

    http://127.0.0.1:5000

Aquí podrás ver:

MAE, RMSE, R² por modelo y zona

parámetros utilizados

artefactos (modelos .pkl)

comparaciones lado a lado

## Modelos Entrenados

Se entrenan los siguientes modelos por zona:

Linear Regression

RidgeCV

LassoCV

RandomForestRegressor (con GridSearchCV)

GradientBoostingRegressor

## Resultados (Resumen)

Los mejores modelos en las tres zonas fueron:

Random Forest (todas las zonas)

Con desempeños:

Zona	MAE	RMSE	R²
Zone 1	973.33	1742.70	0.94
Zone 2	704.43	1419.04	0.93
Zone 3	841.54	2114.40	0.90

Los modelos lineales tuvieron mal desempeño (R² ~ 0.55–0.68), evidenciando fuerte no linealidad en el consumo energético.

Conclusiones Principales

El pipeline es totalmente reproducible mediante DVC.

El uso de MLflow permite una gestión profesional de experimentos.

Las features temporales, cíclicas e interacciones mejoraron notablemente el rendimiento.

Random Forest fue el mejor modelo en todas las zonas.

El proyecto quedó listo para pasar a una Fase 3 (Deploy + API + CI/CD).
