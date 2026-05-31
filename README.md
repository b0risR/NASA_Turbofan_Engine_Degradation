# NASA CMAPSS Turbofan Engine Degradation — Machine Learning Project

Predicción de Vida Útil Remanente (RUL) de motores turbofan usando el dataset **N-CMAPSS DS01** de la NASA, con foco en la clase de vuelo **Fc = 3**.

---

## Dataset

**N-CMAPSS (New Commercial Modular Aero-Propulsion System Simulation)**  
Archivo: `N-CMAPSS_DS01-005.h5` (HDF5, no incluido en el repositorio)

| Grupo | Variables | Descripción |
|---|---|---|
| `W` | Condiciones de vuelo | Altitud, Mach, TRA, T2 |
| `X_s` | Sensores físicos | Temperaturas, presiones, velocidades |
| `A` | Variables auxiliares | `hs` excluida (no observable en motor real) |
| `Y` | Target | RUL en ciclos |

El filtro aplicado es **Fc = 3** (clase de vuelo de media-larga distancia). Los datos se dividen en `dev` (entrenamiento/validación) y `test` tal como provee el dataset original.

---

## Arquitectura: Predecir → Agregar

El enfoque diferencial del proyecto es una arquitectura de **dos fases**:

1. **Predicción individual**: el modelo predice un valor de RUL por cada lectura de sensor (observación individual dentro de un ciclo).
2. **Agregación por ciclo**: las predicciones individuales de cada ciclo se agregan mediante la **mediana** (robusta a outliers de despegue/aterrizaje) para obtener la predicción definitiva del ciclo.

Este diseño permite además obtener **incertidumbre**: la desviación estándar (`RUL_std`) de las predicciones intra-ciclo actúa como proxy de confianza del modelo en cada punto de la vida del motor.

---

## Feature Engineering

Se construyen tres bloques de features sobre los sensores base:

- **`pos_relativa`**: posición normalizada [0, 1] dentro del ciclo. Permite distinguir fases de vuelo (despegue, crucero, descenso), diferenciando cuando los valores de sensor son similares.
- **Rolling inter-ciclo**: tendencia acumulada de degradación entre ciclos. Media calculada en ventanas de 5 y 10 ciclos, más pendiente lineal (slope) sobre ventana de 10 ciclos.
- **Estadísticos intra-ciclo**: máximo y desviación estándar del ciclo actual, para capturar comportamiento bajo máxima demanda y estabilidad del sensor.

Total: **118 features adicionales creadas**.

---

## Modelo: XGBoost con Optuna

- **Modelo Baseline**: `XGBRegressor` con parámetros estándar + early stopping (30 rondas).
- **Fase de Optimización**: 50 trials con **algoritmo Optuna**; la métrica de validación para encontrar los mejores hiper parámetros se calcula sobre métricas *agregadas por cada ciclo entero* (no por cada observación), alineando la optimización y la evaluación final del modelo usando la misma métrica calculada.
- **Split**: leave-one-unit-out temporal — todos los ciclos de una unidad van íntegros a train o validación, nunca mezclados, esto es, se utiliza la data de motores turbofan diferentes para train, evaluación, y test.

---

## Evaluación

| Métrica | Descripción |
|---|---|
| RMSE | Error cuadrático medio (ciclos) |
| MAE | Error absoluto medio (ciclos) |
| R² | Coeficiente de determinación |
| NASA Score (S) | Métrica asimétrica: penaliza más las predicciones tardías (`RUL_pred > RUL_real`) que las adelantadas, dado que predecir vida restante cuando el motor ya está fallando es más peligroso |


---

## Artefactos guardados

Los tres archivos necesarios para inferencia se han almacenado en `models/`:

| Archivo | Contenido |
|---|---|
| `xgb_fc3_model.joblib` | `XGBRegressor` entrenado |
| `xgb_fc3_scaler.joblib` | `StandardScaler` ajustado sobre dev |
| `xgb_fc3_features.joblib` | Lista ordenada de las 118 features (`FEATURE_NAMES`) |

---

## Estructura del repositorio

```
├── EDA_N-CMAPSS_Fc3.ipynb          # Análisis exploratorio del dataset
├── XGBoost_N-CMAPSS_Fc3.ipynb      # Pipeline FE → entrenamiento → evaluación
├── desarrollo_teorico.ipynb         # Marco teórico y justificación metodológica
├── models/
│   ├── xgb_fc3_model.joblib
│   ├── xgb_fc3_scaler.joblib
│   └── xgb_fc3_features.joblib
└── README.md
```

---

## Requisitos

```
python >= 3.11
xgboost
optuna
scikit-learn
h5py
numpy
pandas
matplotlib
seaborn
scipy
joblib
```

---

## Referencia

> Arias Chao, M., Kulkarni, C., Goebel, K., & Fink, O. (2021).  
> *Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics.*  
> NASA Ames Research Center.  
> [https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
