# ⚡ MACHINE LEARNING: QUICK REFERENCE GUIDE
## Resumo Executivo & Referência Rápida para Implementação

---

## 📋 ÍNDICE RÁPIDO

### Seção 1: Setup e Imports
### Seção 2: Pipeline Completo (Template)
### Seção 3: Algoritmos Essenciais
### Seção 4: Métricas e Validação
### Seção 5: Feature Engineering
### Seção 6: Hyperparameter Tuning
### Seção 7: Deploy
### Seção 8: Troubleshooting

---

## 🔧 SEÇÃO 1: SETUP E IMPORTS

### Imports Essenciais
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Para visualizações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```

### Configuração Rápida
```python
# Random seed para reprodutibilidade
SEED = 42
np.random.seed(SEED)

# Verificar dados
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()

# Visualizar distribuições
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()
```

---

## 🏗️ SEÇÃO 2: PIPELINE COMPLETO (TEMPLATE RÁPIDO)

### Template Básico (Copy-Paste)
```python
# 1. CARREGAR E EXPLORAR
df = pd.read_csv('data.csv')
print(df.head())
print(df.isnull().sum())

# 2. PREPARAR DADOS
X = df.drop('target', axis=1)
y = df['target']

# 3. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. ESCALAR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. TREINAR
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 6. PREVER
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# 7. AVALIAR
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.3f}")
print("\n" + classification_report(y_test, y_pred))
```

### Pipeline Profissional (Com Preprocessamento)
```python
# Definir features
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'gender']

# Transformadores
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Combinar
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline final
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Usar
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
score = full_pipeline.score(X_test, y_test)
```

---

## 🤖 SEÇÃO 3: ALGORITMOS ESSENCIAIS (QUICK REFERENCE)

### Classificação

|        Algoritmo       |      Quando Usar         |          Params Importantes         |        Vantagens      |
|------------------------|--------------------------|-------------------------------------|-----------------------|
| **LogisticRegression** | Baseline, dados pequenos | max_iter=1000, C=1.0                | Rápido, interpretável |
| **RandomForest**       | Geral, dados médios      | n_estimators=100, max_depth=10      | Robusto, não linear   |
| **XGBoost**            | Competições, performance | n_estimators=100, learning_rate=0.1 | Alto performance      |
| **LightGBM**           | Dados grandes, rápido    | num_leaves=31, learning_rate=0.1    | Muito rápido          |
| **SVM**                | Dados pequenos, alta dim | kernel='rbf', C=1.0                 | Teórico, robusto      |
| **KNN**                |  Verificação rápida      | n_neighbors=5                       | Simples, não-param    |

### Regressão

|          Algoritmo        |       Quando Usar        | Params Importantes |
|---------------------------|--------------------------|--------------------|
| **LinearRegression**      | Baseline, relação linear | Nenhum crítico     |
| **RandomForestRegressor** | Geral, não-linear        | n_estimators=100   |
| **XGBRegressor**          | Performance alta         | n_estimators=100   |
| **SVR**                   | Dados pequenos           | kernel='rbf'       |

### Quick Code
```python
# Classificação
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regressão
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Unsupervised
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Todos usam .fit() e .predict()!
```

---

## 📊 SEÇÃO 4: MÉTRICAS E VALIDAÇÃO (CHEAT SHEET)

### Escolher Métrica Correta

```python
# CLASSIFICAÇÃO BINÁRIA
if classes_balanceadas:
    metrica = "accuracy"  # ou F1
else:
    metrica = "f1"  # ou "roc_auc"

if precisao_critica:  # Spam, ads
    usar_precision = True
    
if recall_critico:  # Fraud, cancer
    usar_recall = True

# REGRESSÃO
if dados_normais:
    usar = "rmse"
else:
    usar = "mae"  # Robusto a outliers

# CLUSTERING (SEM LABELS)
usar = "silhouette_score"  # -1 a 1, maior melhor
```

### Calcular Rápido

```python
# Classificação
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)

# Regressão
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Clustering
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
```

### Validação Cruzada

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Simples
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Estratificada (para classificação)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

---

## 🔨 SEÇÃO 5: FEATURE ENGINEERING (QUICK TIPS)

### Missing Values
```python
# Ver missings
df.isnull().sum()

# Remover
df.dropna()

# Imputar
from sklearn.impute import SimpleImputer, KNNImputer

# Média/Mediana
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# KNN (melhor)
knn_imp = KNNImputer(n_neighbors=5)
X_imputed = knn_imp.fit_transform(X)

# Forward fill (série temporal)
df.fillna(method='ffill')
```

### Outliers
```python
# IQR Method
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['col'] < Q1 - 1.5*IQR) | (df['col'] > Q3 + 1.5*IQR)
df_clean = df[~outliers]

# Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(X))
df_clean = X[(z_scores < 3).all(axis=1)]

# Isolation Forest
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05)
labels = iso.fit_predict(X)
X_clean = X[labels == 1]
```

### Scaling
```python
# StandardScaler (Recomendado)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# RobustScaler (Com outliers)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### Encoding
```python
# One-Hot
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
X_encoded = ohe.fit_transform(X_cat)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_encoded = le.fit_transform(X_cat)

# Target Encoding
target_map = df.groupby('feature')['target'].mean()
X_encoded = X_cat.map(target_map)
```

### Balanceamento
```python
# SMOTE (Oversampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)

# Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_balanced, y_balanced = rus.fit_resample(X, y)

# Pesos
model = RandomForestClassifier(class_weight='balanced')
```

---

## 🎯 SEÇÃO 6: HYPERPARAMETER TUNING (RÁPIDO)

### GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    params,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid.fit(X_train, y_train)
print(f"Best: {grid.best_params_}")
best_model = grid.best_estimator_
```

### Optuna (Melhor)
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500)
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

### RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV

randomized = RandomizedSearchCV(
    RandomForestClassifier(),
    params,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42
)
randomized.fit(X_train, y_train)
```

---

## 🚀 SEÇÃO 7: DEPLOY

### Salvar Modelo
```python
# Joblib
import joblib
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')

# MLflow
import mlflow
mlflow.sklearn.log_model(model, "model")
```

### FastAPI Mínima
```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = joblib.load('model.pkl')

class Input(BaseModel):
    features: list

@app.post("/predict")
def predict(data: Input):
    pred = model.predict([data.features])[0]
    return {"prediction": int(pred)}

# Rodar: uvicorn main:app --reload
```

### Streamlit Mínima
```python
import streamlit as st
import joblib

st.title("ML App")
model = joblib.load('model.pkl')

age = st.slider('Age', 18, 100)
income = st.number_input('Income')

if st.button('Predict'):
    pred = model.predict([[age, income]])[0]
    st.write(f"Prediction: {pred}")

# Rodar: streamlit run app.py
```

---

## 🐛 SEÇÃO 8: TROUBLESHOOTING

### Problema: Overfitting
```
Sintoma: Train accuracy 95%, Test accuracy 70%

Solução:
1. Aumentar dados
2. Regularização: max_depth, min_samples_split
3. Adicionar dropout (redes neurais)
4. Feature selection
5. Early stopping
```

### Problema: Underfitting
```
Sintoma: Train accuracy 60%, Test accuracy 60%

Solução:
1. Modelo mais complexo
2. Adicionar features
3. Remover regularização
4. Treinar mais épocas
5. Ajustar hiperparâmetros
```

### Problema: Data Leakage
```
Sintoma: Performance ótima em validation, ruim em produção

Solução:
1. Split antes de transformar
2. Usar Pipeline
3. Fit em train, transform em test
4. Remover features futuro-dependentes
```

### Problema: Desbalanceamento
```
Sintoma: Modelo prevê sempre classe maioria

Solução:
1. SMOTE (oversampling)
2. Undersampling
3. Pesos de classe
4. Métrica correta (F1, não Accuracy)
5. Threshold customizado
```

### Problema: Performance Ruim
```
Checklist:
☐ EDA completa?
☐ Features relevantes?
☐ Dados limpos?
☐ Scaling apropriado?
☐ Cross-validation robusto?
☐ Baseline simples testado?
☐ Métrica correta?
☐ Dados suficientes?
```

---

## 📈 QUICK REFERENCE TABLE

### Quando Usar Cada Algoritmo

|       Problema        |       Algoritmo     |           Razão          |
|-----------------------|---------------------|--------------------------|
| Baseline simples      | Logistic Regression | Rápido, interpretável    |
| Produção, high stakes | XGBoost / LightGBM  | Melhor performance       |
| Dados pequenos        | SVM                 | Generaliza bem           |
| Exploração rápida     | Random Forest       | Robusto, rápido          |
| Cluster discovery     | K-Means             | Simples, interpretável   |
| Anomalias             | Isolation Forest    | Específico para outliers |
| Visualização          | PCA / t-SNE         | Reduz dimensionalidade   |

### Tamanho de Dataset

```
Dados pequenos (< 1000)     → SVM, Decision Trees, KNN
Dados médios (1K - 1M)      → Random Forest, Logistic Reg
Dados grandes (> 1M)        → XGBoost, LightGBM, Neural Net
Dados massivos (distribuído) → Spark MLlib, Dask
```

### Complexidade de Tempo

```
Linear Regression   O(n)
K-Means            O(n * k * i)
Random Forest      O(n * m * log(n))
XGBoost            O(n * m * T)
SVM (RBF)          O(n²) ou O(n³)
```

---

## 🎓 CHEAT SHEET FINAL

```python
# 1. PIPELINE MÍNIMO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier().fit(X_train, y_train)
score = model.score(X_test, y_test)

# 2. COM CV
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

# 3. COM TUNING
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5).fit(X, y)
best = grid.best_estimator_

# 4. COM SCALE
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. PIPELINE PROFISSIONAL
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
pipe.fit(X_train, y_train)

# 6. MÉTRICAS
print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
print(f"F1: {f1_score(y, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y, y_pred_proba):.3f}")

# 7. SALVAR
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
```

---

**Dica Final**: Copie este guia em um documento local ou Notion. Revise antes de cada novo projeto!
