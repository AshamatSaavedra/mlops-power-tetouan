# MLOps – Power Consumption of Tetouan City  
## José Ashamat Jaimes Saavedra – A01736690  
### Maestría en Inteligencia Artificial — Fase 1

---

# Objetivo del Proyecto
El objetivo es analizar, limpiar, transformar y modelar el dataset **Power Consumption of Tetouan City** utilizando las mejores prácticas de MLOps, asegurando reproducibilidad, versionado de datos y trazabilidad mediante **DVC**, así como una estructura modular para poder escalar a pipelines más complejos en Fase 2.

El análisis se centra en construir modelos que permitan predecir el consumo energético de las tres zonas de la ciudad:

- **Zone 1 Power Consumption**
- **Zone 2 Power Consumption**
- **Zone 3 Power Consumption**

---

# Estructura del Proyecto
project/
│
├── data/
│ ├── raw/ # Dataset original
│ ├── interim/ # Limpieza parcial
│ └── processed/ # Datos limpios, escalados y PCA
│
├── models/ # Modelos entrenados (.pkl) y métricas (.json)
│
├── scripts/
│ ├── preprocess_n_save.py
│ ├── run_pca.py
│ └── run_modeling.py
│
├── src/
│ ├── data/
│ │ ├── load.py
│ │ └── clean.py
│ ├── features/
│ │ ├── preprocessing.py
│ │ └── pca.py
│ └── models/
│ └── train.py
│
├── dvc.yaml
├── pyproject.toml (Poetry)
└── README.md

---

# 1. Limpieza y Análisis Exploratorio (EDA)

Se realizaron los siguientes procesos:

✅ Corrección de tipos de datos  
✅ Conversión robusta de fechas (`format=mixed`, `dayfirst=True`)  
✅ Eliminación de caracteres no numéricos  
✅ Detección y corrección de outliers  
✅ Imputación por interpolación temporal  
✅ Imputación contextual (radiación = 0 en horario nocturno)  
✅ Limpieza de columnas irrelevantes  
✅ Normalización con RobustScaler  
✅ PCA exploratorio (3 componentes principales)

Los EDA incluyen:

- Histogramas  
- Boxplots  
- Análisis temporal por zonas  
- Matriz de correlación  
- Relaciones bivariadas  
- Distribución por hora del día

---

# 2. Preprocesamiento

El pipeline de preprocesamiento realiza:

✅ Escalado de todas las features numéricas con **RobustScaler**  
✅ Generación de `scaled.csv`  
✅ PCA exploratorio (opcional): `pca_components.csv`

---

# 3. Modelado

Se entrenaron modelos para **cada una de las 3 zonas**:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor (con GridSearchCV)

Los resultados finales mostraron que **RandomForest** es el mejor modelo en las tres zonas:

### ✅ Resultados finales del mejor modelo por zona

| Zona | MAE | RMSE | R² | Mejor Modelo |
|------|------|--------|--------|----------------|
| Zone 1 | 0.303 | 0.440 | 0.538 | Random Forest |
| Zone 2 | 0.316 | 0.456 | 0.541 | Random Forest |
| Zone 3 | 0.313 | 0.470 | 0.641 | Random Forest |

📌 Todos los modelos entrenados se guardan en la carpeta `models/` en formato `.pkl`.  
📌 Sus métricas se guardan en `.json`.

---

# 4. Reproducibilidad con DVC

DVC se utilizó para versionar:

- Datos intermedios (`interim`)  
- Datos procesados (`scaled.csv`)  
- Resultados de PCA  
- Modelos entrenados  
- Métricas

✅ Todos los pipelines son reproducibles con:

dvc repro
Nota: Los datos son ignorados en git, pero versionados por DVC.

# 5. Cómo ejecutar el proyecto
✅ 1. Instalar dependencias
poetry install

✅ 2. Activar el entorno
poetry shell

✅ 3. Descargar datos (ya incluidos en /data/raw)

✅ 4. Ejecutar el pipeline completo
dvc repro

Paso alternativo: Ejecutar scripts manualmente

Preprocesamiento:

poetry run python scripts/preprocess_n_save.py


PCA:

poetry run python scripts/run_pca.py


Modelado:

poetry run python scripts/run_modeling.py

# 6. Conclusiones Fase 1

Se realizó un EDA completo y robusto.

Todos los pasos de procesamiento fueron sistematizados.

El proyecto cuenta con un pipeline reproducible bajo estándares MLOps.

Los resultados de modelado indican que Random Forest es el modelo con mejor desempeño base.

El proyecto queda listo para escalar a Fase 2 con:

Cookiecutter

Pipelines sklearn

MLflow

Feature engineering avanzado


Tracking de experimentos
