# 💻 MACHINE LEARNING: EXEMPLOS DE CÓDIGO PRONTOS
## Snippets Prontos para Copy-Paste e Adaptação

---

## 📋 ÍNDICE

- [1. Projeto Completo: Classification](#1-projeto-completo-classification)
- [2. Projeto Completo: Regression](#2-projeto-completo-regression)
- [3. Clustering Exploratório](#3-clustering-exploratório)
- [4. Feature Engineering Avançado](#4-feature-engineering-avançado)
- [5. Hyperparameter Tuning Completo](#5-hyperparameter-tuning-completo)
- [6. Deploy com API](#6-deploy-com-api)
- [7. Interpretabilidade com SHAP](#7-interpretabilidade-com-shap)
- [8. Monitoramento de Performance](#8-monitoramento-de-performance)
- [9. Ensemble de Modelos](#9-ensemble-de-modelos)
- [10. Tratamento de Dados Desbalanceados](#10-tratamento-de-dados-desbalanceados)

---

## 1. PROJETO COMPLETO: CLASSIFICATION

### Exemplo Real: Previsão de Churn

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PASSO 1: CARREGAR E EXPLORAR DADOS
# ============================================================

df = pd.read_csv('customer_churn.csv')
print("Dataset shape:", df.shape)
print("\nPrimeiras linhas:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDistribuição do target:")
print(df['Churn'].value_counts(normalize=True))

# ============================================================
# PASSO 2: PREPARAR DADOS
# ============================================================

# Separar features e target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

# Identificar features numéricas e categóricas
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# Converter categóricas em numéricas
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ============================================================
# PASSO 3: SPLIT TRAIN-TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Manter proporção de classes
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")

# ============================================================
# PASSO 4: ESCALAR
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# PASSO 5: TREINAR MODELO
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ============================================================
# PASSO 6: PREVER
# ============================================================

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# PASSO 7: AVALIAR
# ============================================================

print("\n" + "="*50)
print("MÉTRICAS DE PERFORMANCE")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n" + classification_report(y_test, y_pred))

# ============================================================
# PASSO 8: VISUALIZAÇÕES
# ============================================================

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 15 Feature Importance')
plt.show()

# ============================================================
# PASSO 9: CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='f1')
print(f"\nCross-Validation F1-Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# PASSO 10: SALVAR MODELO
# ============================================================

import joblib
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModelo salvo!")
```

---

## 2. PROJETO COMPLETO: REGRESSION

### Exemplo Real: Previsão de Preço de Imóvel

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CARREGAR DADOS
# ============================================================

df = pd.read_csv('house_prices.csv')
print(f"Dataset: {df.shape}")

# ============================================================
# EDA RÁPIDA
# ============================================================

print("\nCorrelação com target (Price):")
corr = df.corr()['Price'].sort_values(ascending=False)
print(corr)

# Visualizar
plt.figure(figsize=(12, 8))
sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm')
plt.show()

# ============================================================
# PREPARAR DADOS
# ============================================================

X = df.drop('Price', axis=1)
y = df['Price']

# Lidar com missings
X = X.fillna(X.mean())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Adicionar features polinomiais para features mais importantes
X['Area_squared'] = X['Area'] ** 2
X['Rooms_squared'] = X['Rooms'] ** 2
X['Area_by_Rooms'] = X['Area'] / X['Rooms']

# ============================================================
# SPLIT E SCALE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# TREINAR MÚLTIPLOS MODELOS E COMPARAR
# ============================================================

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    print(f"\n{name}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  MAE:  {mae:,.2f}")
    print(f"  R²:   {r2:.4f}")

# ============================================================
# SELECIONAR MELHOR MODELO E DETALHES
# ============================================================

best_model = models['Gradient Boosting']
y_pred_final = best_model.predict(X_test_scaled)

# Visualizar predições
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_final, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predictions vs Actual')
plt.show()

# Residuals
residuals = y_test - y_pred_final
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_final, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# ============================================================
# CROSS-VALIDATION
# ============================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = -cross_val_score(best_model, X_train_scaled, y_train, 
                             cv=kf, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(cv_scores)

print(f"\n5-Fold CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance')
plt.show()
```

---

## 3. CLUSTERING EXPLORATÓRIO

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CARREGAR E PREPARAR
# ============================================================

df = pd.read_csv('customer_data.csv')
X = df[['Age', 'Income', 'Spending']].fillna(df.mean())

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# ENCONTRAR K ÓTIMO (ELBOW METHOD)
# ============================================================

inertias = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(2, 11), inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(range(2, 11), silhouette_scores, 'go-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score')

plt.tight_layout()
plt.show()

# ============================================================
# TREINAR COM K ÓTIMO
# ============================================================

k_optimal = 3  # Escolhido do gráfico
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, labels):.3f}")

# ============================================================
# VISUALIZAR COM PCA
# ============================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
            pca.transform(kmeans.cluster_centers_)[:, 1],
            c='red', marker='X', s=200, edgecolors='black', linewidth=2,
            label='Centroids')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('K-Means Clustering')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show()

# ============================================================
# ADICIONAR CLUSTERS AO DATAFRAME
# ============================================================

df['Cluster'] = labels

print("\nCluster Characteristics:")
print(df.groupby('Cluster')[['Age', 'Income', 'Spending']].mean())

# ============================================================
# TENTAR DBSCAN (ALTERNATIVA)
# ============================================================

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

print(f"\nDBSCAN found {len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)} clusters")
print(f"Number of noise points: {list(labels_dbscan).count(-1)}")
```

---

## 4. FEATURE ENGINEERING AVANÇADO

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# GERAR DADOS EXEMPLO
# ============================================================

np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.randint(20000, 200000, 1000),
    'score': np.random.rand(1000),
    'has_credit': np.random.randint(0, 2, 1000),
    'target': np.random.randint(0, 2, 1000)
})

X = df.drop('target', axis=1)
y = df['target']

# ============================================================
# FEATURE CREATION
# ============================================================

# Interações
X['age_income_interaction'] = X['age'] * X['income']
X['score_squared'] = X['score'] ** 2

# Transformações
X['log_income'] = np.log1p(X['income'])
X['sqrt_age'] = np.sqrt(X['age'])

# Binning
X['age_group'] = pd.cut(X['age'], bins=[0, 30, 50, 70, 100], labels=[1, 2, 3, 4])

# Ratio features
X['income_to_age'] = X['income'] / (X['age'] + 1)

print("Features criadas:")
print(X.head())

# ============================================================
# FEATURE SELECTION: SelectKBest
# ============================================================

X_numeric = X.select_dtypes(include=[np.number])

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_numeric, y)

# Ver quais features foram selecionadas
selected_features = X_numeric.columns[selector.get_support()].tolist()
print(f"\nTop 5 features (SelectKBest):")
for feat in selected_features:
    print(f"  - {feat}")

# ============================================================
# FEATURE SELECTION: Feature Importance
# ============================================================

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_numeric, y)

feature_importance = pd.DataFrame({
    'feature': X_numeric.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features (Random Forest):")
print(feature_importance.head(10))

# Usar apenas top 10
top_features = feature_importance.head(10)['feature'].tolist()
X_important = X_numeric[top_features]

# ============================================================
# FEATURE SCALING COMPARAÇÃO
# ============================================================

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

X_std = StandardScaler().fit_transform(X_numeric)
X_minmax = MinMaxScaler().fit_transform(X_numeric)
X_robust = RobustScaler().fit_transform(X_numeric)

print("\nScaling Comparison:")
print(f"StandardScaler - Mean: {X_std.mean():.3f}, Std: {X_std.std():.3f}")
print(f"MinMaxScaler - Min: {X_minmax.min():.3f}, Max: {X_minmax.max():.3f}")
print(f"RobustScaler - Median: {np.median(X_robust):.3f}, IQR: {np.percentile(X_robust, 75) - np.percentile(X_robust, 25):.3f}")

# ============================================================
# POLYNOMIAL FEATURES
# ============================================================

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_numeric)

print(f"\nOriginal features: {X_numeric.shape[1]}")
print(f"Polynomial features (degree=2): {X_poly.shape[1]}")
```

---

## 5. HYPERPARAMETER TUNING COMPLETO

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.pruners import MedianPruner
import time

# ============================================================
# PREPARAR DADOS
# ============================================================

df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MÉTODO 1: GRIDSEARCHCV (COMPLETO MAS LENTO)
# ============================================================

print("="*60)
print("MÉTODO 1: GridSearchCV")
print("="*60)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
elapsed = time.time() - start_time

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Time elapsed: {elapsed:.1f}s")

best_model_grid = grid_search.best_estimator_
test_score = best_model_grid.score(X_test_scaled, y_test)
print(f"Test score: {test_score:.4f}")

# ============================================================
# MÉTODO 2: RANDOMIZEDSEARCHCV (MAIS RÁPIDO)
# ============================================================

print("\n" + "="*60)
print("MÉTODO 2: RandomizedSearchCV")
print("="*60)

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': list(range(5, 20)),
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,  # Menos iterações que GridSearch
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
random_search.fit(X_train_scaled, y_train)
elapsed = time.time() - start_time

print(f"\nBest params: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
print(f"Time elapsed: {elapsed:.1f}s")

# ============================================================
# MÉTODO 3: OPTUNA (MAIS INTELIGENTE)
# ============================================================

print("\n" + "="*60)
print("MÉTODO 3: Optuna (Bayesian Optimization)")
print("="*60)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
    }
    
    model = xgb.XGBClassifier(**params, random_state=42)
    
    scores = []
    for train_idx, val_idx in [(list(range(int(len(y_train)*0.8))), 
                                 list(range(int(len(y_train)*0.8), len(y_train))))]:
        X_t, X_v = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=10, verbose=False)
        score = model.score(X_v, y_v)
        scores.append(score)
        
        trial.report(score, len(scores)-1)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(),
    pruner=MedianPruner()
)

start_time = time.time()
study.optimize(objective, n_trials=50, show_progress_bar=True)
elapsed = time.time() - start_time

print(f"\nBest trial: {study.best_trial.number}")
print(f"Best params: {study.best_trial.params}")
print(f"Best value: {study.best_value:.4f}")
print(f"Time elapsed: {elapsed:.1f}s")

# ============================================================
# COMPARAR MÉTODOS
# ============================================================

print("\n" + "="*60)
print("RESUMO COMPARATIVO")
print("="*60)
print(f"GridSearch best CV:     {grid_search.best_score_:.4f}")
print(f"RandomSearch best CV:   {random_search.best_score_:.4f}")
print(f"Optuna best value:      {study.best_value:.4f}")
```

---

## 6. DEPLOY COM API

### FastAPI Completa com Predição

```python
# ========== main.py ==========

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="ML Prediction API",
    description="API para predições de churn de clientes",
    version="1.0.0"
)

# Carregar modelo e recursos
try:
    MODEL = joblib.load('model.pkl')
    SCALER = joblib.load('scaler.pkl')
    FEATURE_NAMES = joblib.load('feature_names.pkl')
    logger.info("✅ Modelo carregado com sucesso")
except Exception as e:
    logger.error(f"❌ Erro ao carregar modelo: {e}")
    raise

# ============================================================
# SCHEMAS (Validação de entrada/saída)
# ============================================================

class CustomerData(BaseModel):
    """Schema para dados de cliente único"""
    age: int = Field(..., ge=18, le=100, description="Age between 18 and 100")
    income: float = Field(..., ge=0, description="Income >= 0")
    score: float = Field(..., ge=0, le=1000, description="Score between 0 and 1000")
    has_credit: int = Field(..., ge=0, le=1, description="Has credit card (0 or 1)")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 50000.0,
                "score": 750.0,
                "has_credit": 1
            }
        }

class BatchPredictionRequest(BaseModel):
    """Schema para predições em lote"""
    data: List[CustomerData]

class PredictionResponse(BaseModel):
    """Schema para resposta de predição"""
    prediction: int = Field(..., description="Prediction: 0 (No Churn) or 1 (Churn)")
    probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    confidence: float = Field(..., ge=0, le=1, description="Confidence of prediction")

class HealthResponse(BaseModel):
    """Schema para health check"""
    status: str
    timestamp: str
    model_version: str

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check do servidor e modelo"""
    return HealthResponse(
        status="OK",
        timestamp=datetime.now().isoformat(),
        model_version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData):
    """
    Fazer predição para um cliente único
    
    Example:
    ```json
    {
        "age": 35,
        "income": 50000,
        "score": 750,
        "has_credit": 1
    }
    ```
    """
    try:
        # Preparar dados
        X = np.array([[
            customer.age,
            customer.income,
            customer.score,
            customer.has_credit
        ]])
        
        # Escalar
        X_scaled = SCALER.transform(X)
        
        # Prever
        prediction = MODEL.predict(X_scaled)[0]
        probability = MODEL.predict_proba(X_scaled)[0].max()
        confidence = max(MODEL.predict_proba(X_scaled)[0])
        
        logger.info(f"Predição realizada: {prediction}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=float(confidence)
        )
    
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """Fazer predições em lote"""
    try:
        # Converter para array
        data_list = [
            [c.age, c.income, c.score, c.has_credit]
            for c in request.data
        ]
        X = np.array(data_list)
        
        # Escalar
        X_scaled = SCALER.transform(X)
        
        # Prever
        predictions = MODEL.predict(X_scaled)
        probabilities = MODEL.predict_proba(X_scaled).max(axis=1)
        
        results = [
            {
                "id": i,
                "prediction": int(pred),
                "probability": float(prob)
            }
            for i, (pred, prob) in enumerate(zip(predictions, probabilities))
        ]
        
        return {"results": results, "total": len(results)}
    
    except Exception as e:
        logger.error(f"Erro em predição em lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def info():
    """Informações sobre o modelo"""
    return {
        "model_type": str(type(MODEL).__name__),
        "n_features": SCALER.n_features_in_,
        "features": FEATURE_NAMES,
        "created_date": "2025-11-05"
    }

# ============================================================
# EXECUTAR LOCALMENTE
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )

# Rodar com: python main.py
# Docs interativa: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Streamlit App

```python
# ========== app.py ==========

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Customer Churn Prediction App")

# Carregar modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("❌ Erro ao carregar modelo")
        return None, None

model, scaler = load_model()

# ============================================================
# SIDEBAR - INPUTS
# ============================================================

st.sidebar.header("📊 Input Features")

age = st.sidebar.slider("Age", 18, 100, 35)
income = st.sidebar.number_input("Income ($)", min_value=0, value=50000)
score = st.sidebar.slider("Credit Score", 300, 850, 750)
has_credit = st.sidebar.radio("Has Credit Card?", ["No", "Yes"])
has_credit_int = 1 if has_credit == "Yes" else 0

# ============================================================
# MAIN - PREDIÇÃO
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Your Profile")
    profile_data = {
        "Age": age,
        "Income": f"${income:,}",
        "Credit Score": score,
        "Has Credit": has_credit
    }
    st.json(profile_data)

with col2:
    if st.button("🔮 Make Prediction", use_container_width=True):
        # Preparar dados
        X = np.array([[age, income, score, has_credit_int]])
        X_scaled = scaler.transform(X)
        
        # Prever
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        # Mostrar resultados
        st.subheader("Results")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Prediction",
                "WILL CHURN" if prediction == 1 else "WILL STAY",
                delta=f"{probability[prediction]*100:.1f}% confident"
            )
        
        with metric_col2:
            st.metric(
                "Churn Probability",
                f"{probability[1]*100:.1f}%"
            )
        
        with metric_col3:
            st.metric(
                "Confidence",
                f"{max(probability)*100:.1f}%"
            )
        
        # Gráfico de probabilidades
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ['Stay', 'Churn']
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(categories, probability, color=colors)
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_ylim([0, 1])
        for i, (cat, prob) in enumerate(zip(categories, probability)):
            ax.text(i, prob + 0.02, f'{prob*100:.1f}%', ha='center', fontweight='bold')
        st.pyplot(fig)

# ============================================================
# TAB - UPLOAD DE ARQUIVO
# ============================================================

st.divider()

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.info("Use the sidebar to input customer data and make predictions")

with tab2:
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.write(df.head())
        
        if st.button("Make Batch Predictions"):
            # Preparar dados
            X_batch = df[['Age', 'Income', 'Score', 'HasCredit']].values
            X_batch_scaled = scaler.transform(X_batch)
            
            # Prever
            predictions = model.predict(X_batch_scaled)
            probabilities = model.predict_proba(X_batch_scaled)
            
            # Adicionar ao dataframe
            df['Prediction'] = predictions
            df['Churn_Probability'] = probabilities[:, 1]
            df['Will_Churn'] = df['Prediction'].map({0: 'No', 1: 'Yes'})
            
            st.write("Predictions:")
            st.write(df)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

# ========== Rodar com: streamlit run app.py ==========
```

---

## 7. INTERPRETABILIDADE COM SHAP

```python
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar modelo
model = joblib.load('model.pkl')
X_test = np.load('X_test.npy')
feature_names = ['age', 'income', 'score', 'has_credit']

# ============================================================
# SHAP EXPLAINER
# ============================================================

# Para modelos tree-based
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ============================================================
# SUMMARY PLOT (Importância Global)
# ============================================================

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot - Feature Importance")
plt.tight_layout()
plt.show()

# ============================================================
# DEPENDENCE PLOT (Relação Feature-Output)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, ax in enumerate(axes.flat):
    if idx < len(feature_names):
        shap.dependence_plot(idx, shap_values, X_test, 
                           feature_names=feature_names, ax=ax, show=False)
plt.tight_layout()
plt.show()

# ============================================================
# FORCE PLOT (Predição Individual)
# ============================================================

# Para uma amostra específica
sample_idx = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_test[sample_idx],
    feature_names=feature_names,
    matplotlib=True
)
plt.show()

# ============================================================
# WATERFALL PLOT (Mais Intuitivo)
# ============================================================

explanation = shap.Explanation(
    values=shap_values[sample_idx],
    base_values=explainer.expected_value,
    data=X_test[sample_idx],
    feature_names=feature_names
)

shap.plots.waterfall(explanation)
plt.show()

# ============================================================
# SHAP VALUE ANALYSIS (Texto)
# ============================================================

print("SHAP Value Analysis for Sample 0:")
print("="*50)
print(f"Base value (model average): {explainer.expected_value:.4f}")
print()

# Ordenar por importância
shap_contrib = sorted(
    zip(feature_names, shap_values[sample_idx]),
    key=lambda x: abs(x[1]),
    reverse=True
)

for feat, contrib in shap_contrib:
    direction = "↑" if contrib > 0 else "↓"
    print(f"{feat:15} {direction} {abs(contrib):.4f}")
```

---

## 8. MONITORAMENTO DE PERFORMANCE

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# MODEL MONITOR CLASS
# ============================================================

class ModelMonitor:
    def __init__(self, baseline_metrics, threshold_drop=0.05):
        """
        baseline_metrics: dict como {'accuracy': 0.85, 'f1': 0.82}
        threshold_drop: queda máxima aceitável
        """
        self.baseline_metrics = baseline_metrics
        self.threshold_drop = threshold_drop
        self.history = []
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """Avaliar e registrar performance"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        self.history.append(metrics)
        return metrics
    
    def check_drift(self, current_metrics):
        """Verificar se performance degradou"""
        issues = []
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                drop = baseline_value - current_value
                
                if drop > self.threshold_drop:
                    issues.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'drop': drop,
                        'status': 'ALERT'
                    })
                    logger.warning(f"⚠️  {metric_name}: {baseline_value:.3f} → {current_value:.3f}")
                elif drop > self.threshold_drop / 2:
                    issues.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'drop': drop,
                        'status': 'WARNING'
                    })
                    logger.info(f"⚡ {metric_name}: {baseline_value:.3f} → {current_value:.3f}")
        
        return issues
    
    def should_retrain(self):
        """Decidir se deve re-treinar"""
        if len(self.history) < 3:
            return False
        
        # Se últimas 3 avaliações pioraram
        recent_f1 = [h['f1'] for h in self.history[-3:]]
        baseline_f1 = self.baseline_metrics.get('f1', 0.8)
        
        if all(f1 < baseline_f1 - self.threshold_drop for f1 in recent_f1):
            logger.error("🔴 Multiple degradations detected - RETRAIN RECOMMENDED")
            return True
        
        return False
    
    def get_report(self):
        """Gerar relatório"""
        if not self.history:
            return "No evaluations yet"
        
        df = pd.DataFrame(self.history)
        return df.describe()

# ============================================================
# USO
# ============================================================

# Baseline (métricas de validação)
baseline = {
    'accuracy': 0.85,
    'precision': 0.83,
    'recall': 0.82,
    'f1': 0.82
}

# Inicializar monitor
monitor = ModelMonitor(baseline, threshold_drop=0.05)

# Diariamente, avaliar novo batch
y_true_batch = np.random.randint(0, 2, 1000)
y_pred_batch = np.random.randint(0, 2, 1000)

# Avaliar
metrics = monitor.evaluate(y_true_batch, y_pred_batch)
print("Current metrics:", metrics)

# Verificar drift
issues = monitor.check_drift(metrics)
print(f"Issues found: {len(issues)}")

# Decidir re-treinamento
if monitor.should_retrain():
    logger.info("🔄 Starting re-training process...")

# Relatório
print("\n" + monitor.get_report().to_string())
```

---

## 9. ENSEMBLE DE MODELOS

```python
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# ============================================================
# VOTING CLASSIFIER
# ============================================================

voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    voting='soft'  # 'soft'=média de probabilidades, 'hard'=moda
)

voting.fit(X_train, y_train)
predictions = voting.predict(X_test)

# ============================================================
# STACKING CLASSIFIER
# ============================================================

# Level 0: Base models
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, kernel='rbf', random_state=42))
]

# Level 1: Meta-model
meta_model = LogisticRegression()

# Combinar
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5
)

stacking.fit(X_train, y_train)
predictions = stacking.predict(X_test)

# Feature importance do stacking
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': meta_model.coef_[0]
}).abs().sort_values('importance', ascending=False)

print(feature_importance)
```

---

## 10. TRATAMENTO DE DADOS DESBALANCEADOS

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# VERIFICAR DESBALANCEAMENTO
# ============================================================

print("Class distribution:")
print(y.value_counts(normalize=True))

imbalance_ratio = y.value_counts()[1] / y.value_counts()[0]
print(f"Imbalance ratio: {imbalance_ratio:.3f}")

# ============================================================
# MÉTODO 1: SMOTE (OVERSAMPLING)
# ============================================================

smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Original shape: {X_train.shape}")
print(f"SMOTE shape: {X_smote.shape}")
print(f"Class distribution:\n{pd.Series(y_smote).value_counts(normalize=True)}")

# ============================================================
# MÉTODO 2: UNDERSAMPLING
# ============================================================

rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

print(f"\nAfter Undersampling:")
print(f"Shape: {X_rus.shape}")

# ============================================================
# MÉTODO 3: COMBINED PIPELINE (SMOTE + Undersampling)
# ============================================================

pipeline = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.9)),
    ('undersampler', RandomUnderSampler(sampling_strategy=0.8)),
])

X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
print(f"\nAfter combined pipeline: {X_balanced.shape}")

# ============================================================
# MÉTODO 4: PESOS DE CLASSE
# ============================================================

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # ou passar dict
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ============================================================
# MÉTODO 5: THRESHOLD ADJUSTMENT
# ============================================================

# Treinar modelo normal
model.fit(X_train_scaled, y_train)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Ajustar threshold
threshold = 0.3  # Padrão é 0.5
y_pred_adjusted = (y_pred_proba > threshold).astype(int)

print(f"\nDefault threshold (0.5):")
print(f"Recall: {recall_score(y_test, (y_pred_proba > 0.5).astype(int)):.3f}")

print(f"\nAdjusted threshold (0.3):")
print(f"Recall: {recall_score(y_test, y_pred_adjusted):.3f}")

# ============================================================
# COMPARAR MÉTODOS
# ============================================================

methods = {
    'Original': (X_train_scaled, y_train),
    'SMOTE': (X_smote, y_smote),
    'Undersampling': (X_rus, y_rus),
    'Balanced': (X_balanced, y_balanced)
}

results = {}
for name, (X, y) in methods.items():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X_test_scaled)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)
    
    results[name] = {'F1': f1, 'Recall': recall}
    print(f"{name:20} F1: {f1:.3f}  Recall: {recall:.3f}")
```

---

## CONCLUSÃO

Este arquivo contém **10 exemplos práticos completos** que você pode adaptar para seus projetos. Cada seção é **auto-contida** e pronta para copy-paste.

**Próximos passos:**
1. Escolha um exemplo que corresponda seu problema
2. Adapte os nomes de features e paths
3. Execute e itere

Boa sorte! 🚀
