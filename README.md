# 📊 Predicción de Churn — Regresión Logística & Árbol de Decisión

> Modelo de clasificación para identificar clientes con riesgo de evasión en una empresa de telecomunicaciones.

---

## 📋 Índice

1. [Descripción](#descripción)
2. [Capacidades del proyecto](#capacidades-del-proyecto)
3. [Tecnologías usadas](#tecnologías-usadas)
4. [Cómo acceder al proyecto](#cómo-acceder-al-proyecto)

---

## 📝 Descripción

Este proyecto implementa un pipeline completo de Machine Learning para predecir la **evasión de clientes (churn)** en una empresa de telecomunicaciones. A partir de variables demográficas, de contrato y de consumo, se entrenaron y compararon dos modelos de clasificación supervisada con el objetivo de responder tres preguntas clave de negocio:

- ¿Quiénes son los clientes con **mayor riesgo de evasión**?
- ¿Qué **variables influyen más** en este comportamiento?
- ¿Qué **perfil de cliente** necesita que la empresa se mantenga más cerca?

El modelo ganador por ROC-AUC fue la **Regresión Logística (AUC = 0.8446)**, logrando identificar correctamente el **80% de los clientes** que realmente abandonan el servicio.

---

## ⚙️ Capacidades del proyecto

| Capacidad | Detalle |
|---|---|
| 🧹 **Limpieza de datos** | Imputación de nulos, eliminación de columnas redundantes (`Charges.Daily`), conversión de tipos |
| 🔠 **Encoding** | One-Hot Encoding de variables categóricas con `drop_first` para evitar multicolinealidad |
| 🤖 **Modelado dual** | Entrenamiento paralelo de Regresión Logística y Árbol de Decisión con `class_weight='balanced'` |
| 📊 **Evaluación comparativa** | Métricas completas: ROC-AUC, Accuracy, Precision, Recall, F1-Score y validación cruzada 5-fold |
| 📈 **Visualizaciones** | Matrices de confusión, curvas ROC superpuestas, importancia de variables y árbol de decisión visual |
| 🔍 **Análisis de negocio** | Respuestas directas a las 3 preguntas con tablas de acción por segmento de riesgo |
| 💾 **Exportación** | CSV con probabilidades de churn de ambos modelos (`prob_rl`, `prob_dt`, `prob_promedio`) por cliente |

### Perfil de alto riesgo detectado — *"Cliente Digital Sin Ancla"*

```
✗ Contrato mes a mes
✗ Servicio de Fibra Óptica
✗ Menos de 12 meses de antigüedad (tenure bajo)
✗ Sin TechSupport ni OnlineSecurity activos
✗ Facturación sin papel (PaperlessBilling)
✗ Pago no automático
✗ Senior Citizen sin pareja ni dependientes
```

---

## 🛠️ Tecnologías usadas

| Librería | Versión recomendada | Uso |
|---|---|---|
| `pandas` | ≥ 1.5 | Manipulación y análisis de datos |
| `numpy` | ≥ 1.23 | Operaciones numéricas |
| `scikit-learn` | ≥ 1.2 | Modelos, pipelines, métricas y validación |
| `matplotlib` | ≥ 3.6 | Visualizaciones base |
| `seaborn` | ≥ 0.12 | Visualizaciones estadísticas |
| `IPython` | incluido en Colab | Renderizado Markdown en notebooks |

> **Entorno:** Google Colab (Python 3.10+)

---

## 🚀 Cómo acceder al proyecto

### 1. Clonar o descargar el archivo

Descarga el archivo `churn_model.py` y súbelo a tu entorno de Google Colab, o copia su contenido en una celda de código.

### 2. Instalar dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

> En Google Colab todas las librerías ya están preinstaladas. No es necesario instalar nada adicional.

### 3. Cargar tu dataset

Sube tu archivo CSV a Colab y actualiza la ruta en la línea correspondiente:

```python
# En Google Colab puedes subir el archivo así:
from google.colab import files
uploaded = files.upload()

# Luego actualiza esta línea con el nombre de tu archivo:
df = pd.read_csv("tu_archivo.csv")
```

El dataset debe contener las siguientes columnas:

```
Churn, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
Charges.Monthly, Charges.Total
```

### 4. Ejecutar el notebook

Corre todas las celdas en orden. Al finalizar obtendrás:

- 📊 Gráficos comparativos de ambos modelos
- 📋 Análisis de las 3 preguntas de negocio en formato Markdown
- 📁 Archivo `clientes_alto_riesgo.csv` con los clientes identificados

---

*Proyecto desarrollado con fines de análisis de retención de clientes.*
