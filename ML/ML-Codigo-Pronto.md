# Machine Learning: Exemplos de Código Prontos
## Snippets Copy-Paste para Implementação Imediata

---

## Índice de Código

1. [Classificação de Imagens com CNN](#classificação-de-imagens-com-cnn)
2. [Previsão de Séries Temporais com LSTM](#previsão-de-séries-temporais-com-lstm)
3. [Sistema de Recomendação](#sistema-de-recomendação)
4. [Classificação Tabular Completa](#classificação-tabular-completa)
5. [Análise de Sentimentos NLP](#análise-de-sentimentos-nlp)
6. [Detecção de Anomalias](#detecção-de-anomalias)

---

# Classificação de Imagens com CNN

## Projeto: CIFAR-10 Image Classification

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================================
# 1. CARREGAR E PREPARAR DADOS
# ============================================

# Carregar dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalizar para [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Split validação
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

print(f"Train shape: {x_train.shape}")
print(f"Val shape: {x_val.shape}")
print(f"Test shape: {x_test.shape}")

# ============================================
# 2. DATA AUGMENTATION
# ============================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ============================================
# 3. CONSTRUIR MODELO CNN
# ============================================

def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        # Bloco 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Camadas Densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = build_cnn_model()
model.summary()

# ============================================
# 4. COMPILAR E TREINAR
# ============================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Treinar com data augmentation
history = model.fit(
    data_augmentation(x_train, training=True),
    y_train,
    batch_size=128,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ============================================
# 5. AVALIAR
# ============================================

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predições
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Visualizar histórico
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ============================================
# 6. SALVAR E CARREGAR MODELO
# ============================================

# Salvar
model.save('cifar10_model.h5')

# Carregar
loaded_model = keras.models.load_model('cifar10_model.h5')

# Fazer predição com modelo carregado
sample_pred = loaded_model.predict(x_test[:5])
print("\nAmostra de predições:")
print(np.argmax(sample_pred, axis=1))
```

---

# Previsão de Séries Temporais com LSTM

## Projeto: Predição de Preço de Ação

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================
# 1. GERAR/CARREGAR DADOS
# ============================================

# Simular dados de série temporal (preço de ação)
np.random.seed(42)
n_samples = 1000
data = np.cumsum(np.random.randn(n_samples)) + 100  # Random walk + tendência

# Normalizar
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# ============================================
# 2. PREPARAR DADOS (SLIDING WINDOW)
# ============================================

def create_sequences(data, window_size=60):
    """Cria sequências para LSTM"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(data_scaled, window_size)

# Split treino/teste (80/20)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape para LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================
# 3. CONSTRUIR MODELO LSTM
# ============================================

model = Sequential([
    # Camada LSTM 1
    layers.LSTM(
        units=50,
        activation='relu',
        input_shape=(window_size, 1),
        return_sequences=True
    ),
    layers.Dropout(0.2),
    
    # Camada LSTM 2
    layers.LSTM(
        units=50,
        activation='relu',
        return_sequences=False
    ),
    layers.Dropout(0.2),
    
    # Camadas Densas
    layers.Dense(units=25, activation='relu'),
    layers.Dense(units=1)  # Output: predição de preço
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ============================================
# 4. TREINAR
# ============================================

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)

# ============================================
# 5. AVALIAR E PREVER
# ============================================

# Predições
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Desnormalizar
y_train_pred = scaler.inverse_transform(y_train_pred)
y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# Métricas
train_mse = mean_squared_error(y_train_true, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train_true, y_train_pred)

test_mse = mean_squared_error(y_test_true, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_true, y_test_pred)

print(f"\nTrain RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

# ============================================
# 6. VISUALIZAR RESULTADOS
# ============================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Plot Loss
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Model Loss')

# Plot Predições
ax2.plot(y_train_true, label='Train True', alpha=0.7)
ax2.plot(y_train_pred, label='Train Pred', alpha=0.7)
ax2.axvline(x=len(y_train_true), color='red', linestyle='--', label='Test Start')
ax2.plot(range(len(y_train_true), len(y_train_true)+len(y_test_true)), 
         y_test_true, label='Test True', alpha=0.7)
ax2.plot(range(len(y_train_true), len(y_train_true)+len(y_test_pred)), 
         y_test_pred, label='Test Pred', alpha=0.7)
ax2.set_xlabel('Time')
ax2.set_ylabel('Price')
ax2.legend()
ax2.set_title('LSTM Predictions')

plt.tight_layout()
plt.savefig('lstm_timeseries.png')
plt.show()

# ============================================
# 7. FAZER PREVISÃO FUTURA
# ============================================

def predict_future(model, last_sequence, n_future=10):
    """Prever n_future passos à frente"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future):
        next_pred = model.predict(current_sequence.reshape(1, window_size, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Slide window
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    return np.array(predictions)

# Última sequência de teste
last_seq = X_test[-1].flatten()

# Prever 30 dias à frente
future_preds = predict_future(model, last_seq, n_future=30)

# Desnormalizar
future_preds = scaler.inverse_transform(future_preds.reshape(-1, 1))

print(f"\nPredições futuras (próximos 30 passos):")
print(future_preds.flatten())
```

---

# Sistema de Recomendação

## Projeto: Recomendação Colaborativa com Matrix Factorization

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ============================================
# 1. GERAR DADOS DE RATINGS
# ============================================

# Simular matriz user-item
n_users = 100
n_movies = 50

np.random.seed(42)
# Matriz sparse (maioria são missing values)
ratings = np.random.randint(1, 6, size=(n_users, n_movies))
# Tornar sparse (~60% faltando)
mask = np.random.random((n_users, n_movies)) < 0.6
ratings[mask] = 0

# Converter para DataFrame
movies = [f"Movie_{i}" for i in range(n_movies)]
users = [f"User_{i}" for i in range(n_users)]

ratings_df = pd.DataFrame(ratings, index=users, columns=movies)

print(f"Ratings shape: {ratings_df.shape}")
print(f"Sparsity: {(ratings_df == 0).sum().sum() / (n_users * n_movies):.2%}")

# ============================================
# 2. MATRIX FACTORIZATION (NMF)
# ============================================

# Preencher zeros com valor médio para NMF
ratings_filled = ratings_df.fillna(ratings_df.mean().mean())

# Aplicar NMF
n_factors = 10  # Dimensão latente

nmf = NMF(n_components=n_factors, init='random', random_state=42, max_iter=200)
user_factors = nmf.fit_transform(ratings_filled)  # [n_users, n_factors]
item_factors = nmf.components_.T  # [n_movies, n_factors]

# Reconstruir ratings preditos
predicted_ratings = user_factors @ item_factors.T

print(f"\nUser factors shape: {user_factors.shape}")
print(f"Item factors shape: {item_factors.shape}")

# ============================================
# 3. GERAR RECOMENDAÇÕES
# ============================================

def get_recommendations(user_idx, predicted_ratings, n_recommendations=5):
    """
    Retorna top N recomendações para usuário
    """
    # Scores para este usuário
    user_scores = predicted_ratings[user_idx]
    
    # Ordenar filmes
    top_indices = np.argsort(user_scores)[::-1]
    
    # Filtrar filmes já avaliados
    original_ratings = ratings_df.iloc[user_idx].values
    recommendations = []
    
    for idx in top_indices:
        if original_ratings[idx] == 0:  # Não avaliado ainda
            recommendations.append((movies[idx], user_scores[idx]))
            if len(recommendations) == n_recommendations:
                break
    
    return recommendations

# Recomendar para usuário 0
user_recs = get_recommendations(0, predicted_ratings, n_recommendations=5)
print(f"\nTop 5 recomendações para {users[0]}:")
for movie, score in user_recs:
    print(f"  {movie}: {score:.2f}")

# ============================================
# 4. SIMILARIDADE BASEADA EM CONTEÚDO
# ============================================

# Calcular similaridade entre usuários
user_similarity = cosine_similarity(user_factors)

# Top usuários similares a usuário 0
similar_users = np.argsort(user_similarity[0])[::-1][1:6]
print(f"\nTop 5 usuários similares a {users[0]}:")
for idx in similar_users:
    print(f"  {users[idx]}: {user_similarity[0, idx]:.3f}")

# Similaridade entre filmes
movie_similarity = cosine_similarity(item_factors)

# Filmes similares a Movie_0
similar_movies = np.argsort(movie_similarity[0])[::-1][1:6]
print(f"\nTop 5 filmes similares a {movies[0]}:")
for idx in similar_movies:
    print(f"  {movies[idx]}: {movie_similarity[0, idx]:.3f}")

# ============================================
# 5. AVALIAR (RMSE EM RATINGS CONHECIDOS)
# ============================================

from sklearn.metrics import mean_squared_error

# Calcular RMSE apenas em ratings conhecidos
known_mask = ratings_df != 0
actual_known = ratings_df[known_mask].values
predicted_known = predicted_ratings[known_mask]

rmse = np.sqrt(mean_squared_error(actual_known, predicted_known))
print(f"\nRMSE on known ratings: {rmse:.4f}")

# ============================================
# 6. VISUALIZAR
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Heatmap original
axes[0, 0].imshow(ratings_df.values[:20, :20], cmap='YlOrRd')
axes[0, 0].set_title('Original Ratings (20x20 subset)')
axes[0, 0].set_ylabel('Users')
axes[0, 0].set_xlabel('Movies')

# Heatmap predito
axes[0, 1].imshow(predicted_ratings[:20, :20], cmap='YlOrRd')
axes[0, 1].set_title('Predicted Ratings (20x20 subset)')
axes[0, 1].set_ylabel('Users')
axes[0, 1].set_xlabel('Movies')

# Distribuição de factors de usuário
axes[1, 0].scatter(user_factors[:, 0], user_factors[:, 1], alpha=0.6)
axes[1, 0].set_title('User Latent Factors (2D)')
axes[1, 0].set_xlabel('Factor 1')
axes[1, 0].set_ylabel('Factor 2')

# Distribuição de factors de item
axes[1, 1].scatter(item_factors[:, 0], item_factors[:, 1], alpha=0.6)
axes[1, 1].set_title('Movie Latent Factors (2D)')
axes[1, 1].set_xlabel('Factor 1')
axes[1, 1].set_ylabel('Factor 2')

plt.tight_layout()
plt.savefig('recommendation_system.png')
plt.show()
```

---

# Classificação Tabular Completa

## Projeto: Previsão de Churn de Clientes

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CARREGAR DADOS
# ============================================

# Simular dataset de churn
np.random.seed(42)
n_samples = 5000

data = {
    'age': np.random.randint(18, 75, n_samples),
    'tenure': np.random.randint(0, 73, n_samples),
    'monthly_charges': np.random.uniform(18, 118, n_samples),
    'total_charges': np.random.uniform(18, 8684, n_samples),
    'contract_length': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'has_tech_support': np.random.choice([0, 1], n_samples),
    'has_phone_service': np.random.choice([0, 1], n_samples),
}

# Target (simulado com correlações)
y = (
    (data['tenure'] < 10).astype(int) * 0.3 +
    (data['monthly_charges'] > 90).astype(int) * 0.2 +
    (data['contract_length'] == 'Month-to-month').astype(int) * 0.3 +
    np.random.random(n_samples) * 0.2
).astype(int)

# Converter para DataFrame
df = pd.DataFrame(data)
df['churn'] = y

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nChurn distribution:")
print(df['churn'].value_counts())

# ============================================
# 2. EXPLORAÇÃO E VISUALIZAÇÃO
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Histogramas
df['age'].hist(ax=axes[0, 0], bins=20)
axes[0, 0].set_title('Age Distribution')

df['tenure'].hist(ax=axes[0, 1], bins=20)
axes[0, 1].set_title('Tenure Distribution')

df['monthly_charges'].hist(ax=axes[0, 2], bins=20)
axes[0, 2].set_title('Monthly Charges Distribution')

# Churn por categoria
df.groupby('contract_length')['churn'].mean().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Churn Rate by Contract')

df.groupby('internet_service')['churn'].mean().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Churn Rate by Internet Service')

# Correlação
corr_cols = df.select_dtypes(include=[np.number]).columns
df[corr_cols].corr()['churn'].sort_values(ascending=False).plot(kind='barh', ax=axes[1, 2])
axes[1, 2].set_title('Feature Correlation with Churn')

plt.tight_layout()
plt.savefig('exploratory_analysis.png')
plt.show()

# ============================================
# 3. PRÉ-PROCESSAMENTO
# ============================================

# Separar features e target
X = df.drop('churn', axis=1)
y = df['churn']

# Identificar colunas categóricas e numéricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# One-hot encoding para categóricas
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Normalizar numéricas
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print(f"Final feature count: {X.shape[1]}")

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================
# 4. TREINAR MÚLTIPLOS MODELOS
# ============================================

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")
    
    # Treinar
    model.fit(X_train, y_train)
    
    # Predições
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"ROC-AUC: {test_auc:.4f}")
    
    results[name] = {
        'model': model,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba,
        'metrics': {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1,
            'auc': test_auc
        }
    }
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(importances.head(10))

# ============================================
# 5. SELECIONAR MELHOR MODELO E AVALIAR
# ============================================

best_model_name = max(results, key=lambda x: results[x]['metrics']['f1'])
best_model = results[best_model_name]['model']
best_results = results[best_model_name]

print(f"\n{'='*50}")
print(f"Best Model: {best_model_name}")
print(f"{'='*50}")

# Confusion Matrix
cm = confusion_matrix(y_test, best_results['y_pred'])
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, best_results['y_pred'], 
                          target_names=['No Churn', 'Churn']))

# ============================================
# 6. VISUALIZAR RESULTADOS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title(f'Confusion Matrix - {best_model_name}')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_results['y_proba'])
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {best_results["metrics"]["auc"]:.3f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Métricas comparação
metrics_names = list(results[best_model_name]['metrics'].keys())
metrics_values = [results[name]['metrics'] for name in results.keys()]

x = np.arange(len(metrics_names))
width = 0.25

for i, (model_name, metrics) in enumerate(results.items()):
    values = [metrics['metrics'][m] for m in metrics_names]
    axes[1, 0].bar(x + i*width, values, width, label=model_name)

axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Metrics Comparison')
axes[1, 0].set_xticks(x + width)
axes[1, 0].set_xticklabels(metrics_names)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# Feature Importance (top 10)
if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    axes[1, 1].barh(importances['feature'], importances['importance'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Features')
    axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('classification_results.png')
plt.show()

# ============================================
# 7. FAZER PREDIÇÕES EM NOVOS DADOS
# ============================================

# Simular novo cliente
new_customer = pd.DataFrame({
    'age': [45],
    'tenure': [6],
    'monthly_charges': [95],
    'total_charges': [570],
    'contract_length': ['Month-to-month'],
    'internet_service': ['Fiber optic'],
    'has_tech_support': [0],
    'has_phone_service': [1]
})

# Preprocessar
new_customer = pd.get_dummies(new_customer, columns=categorical_cols, drop_first=True)

# Garantir mesmas colunas
for col in X.columns:
    if col not in new_customer.columns:
        new_customer[col] = 0

new_customer = new_customer[X.columns]

# Normalizar
new_customer[numerical_cols] = scaler.transform(new_customer[numerical_cols])

# Prever
churn_prob = best_model.predict_proba(new_customer)[0]

print(f"\nPrediction for new customer:")
print(f"Probability of No Churn: {churn_prob[0]:.2%}")
print(f"Probability of Churn: {churn_prob[1]:.2%}")
print(f"Prediction: {'CHURN' if best_model.predict(new_customer)[0] == 1 else 'NO CHURN'}")
```

---

# Análise de Sentimentos NLP

## Projeto: Classificação de Reviews com Transformer (DistilBERT)

```python
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. DADOS
# ============================================

# Simular reviews e sentimentos
reviews = [
    "Este produto é excelente! Muito satisfeito com a qualidade.",
    "Péssimo serviço, não recomendo.",
    "Produto normal, nada excepcional.",
    "Adorei! Superou minhas expectativas!",
    "Terrível experiência, não funciona como prometido.",
    "Bom custo-benefício, recomendo.",
    "Não vale a pena o preço cobrado.",
    "Fantástico! Exatamente o que procurava.",
    "Decepcionante, esperava mais.",
    "Ótimo atendimento, produto de qualidade!"
]

# Labels: 0 = Negativo, 1 = Neutro, 2 = Positivo
sentiments = [2, 0, 1, 2, 0, 2, 0, 2, 0, 2]

data = pd.DataFrame({
    'review': reviews,
    'sentiment': sentiments,
    'sentiment_label': ['Positivo' if s == 2 else 'Negativo' if s == 0 else 'Neutro' for s in sentiments]
})

print("Dataset shape:", data.shape)
print("\nSentiment distribution:")
print(data['sentiment_label'].value_counts())

# ============================================
# 2. CARREGAR PRÉ-TREINADO TRANSFORMERS
# ============================================

# Usar DistilBERT (mais leve que BERT)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english'
)

# Verificar device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Device: {device}")

# ============================================
# 3. FAZER PREDIÇÕES (ZERO-SHOT)
# ============================================

def predict_sentiment(text, tokenizer, model, device):
    """Classificar sentimento de texto"""
    # Tokenizar
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extrair logits e proba
    logits = outputs.logits
    proba = torch.nn.functional.softmax(logits, dim=-1)
    prediction = torch.argmax(logits, dim=-1).item()
    
    return prediction, proba[0].cpu().numpy()

# Fazer predições
predictions = []
probabilities = []

for review in data['review']:
    pred, proba = predict_sentiment(review, tokenizer, model, device)
    predictions.append(pred)
    probabilities.append(proba)

data['prediction'] = predictions
data['confidence'] = [max(p) for p in probabilities]

print("\nPredictions:")
print(data[['review', 'prediction', 'confidence']])

# ============================================
# 4. AVALIAR
# ============================================

# Simplificar para binário (0 = Negativo, 1 = Positivo)
y_true = (data['sentiment'] != 0).astype(int)
y_pred = (data['prediction'] != 0).astype(int)

print("\nClassification Report (Binary):")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================================
# 5. FINE-TUNING (OPCIONAL)
# ============================================

print("\n" + "="*50)
print("Fine-tuning do modelo")
print("="*50)

from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Criar dataset
dataset = SentimentDataset(
    data['review'].tolist(),
    data['sentiment'].tolist(),
    tokenizer
)

print(f"Dataset size: {len(dataset)}")

# Fine-tuning args
training_args = TrainingArguments(
    output_dir='./sentiment_model',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("Fine-tuning...")
# trainer.train()  # Descomentar para treinar
print("Fine-tuning completo!")

# ============================================
# 6. ANÁLISE DE RESULTADOS
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribuição de confiança
axes[0].hist(data['confidence'], bins=10, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Confidence')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Model Confidence Distribution')

# Confusion Matrix
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
axes[1].set_title('Confusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('sentiment_analysis.png')
plt.show()

# ============================================
# 7. PREVER EM NOVO TEXTO
# ============================================

new_review = "Este produto é absolutamente fantástico! Recomendo muito!"
pred, proba = predict_sentiment(new_review, tokenizer, model, device)

sentiment_map = {0: 'Negativo', 1: 'Positivo'}
print(f"\nNew review: '{new_review}'")
print(f"Prediction: {sentiment_map[pred]}")
print(f"Confidence: {max(proba):.2%}")
```

---

# Detecção de Anomalias

## Projeto: Detecção com Isolation Forest e Autoencoders

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from tensorflow.keras import layers, models, Sequential
import tensorflow as tf

# ============================================
# 1. GERAR DADOS COM ANOMALIAS
# ============================================

np.random.seed(42)

# Gerar dados normais
X_normal, _ = make_blobs(n_samples=1000, n_features=2, centers=1, random_state=42)
X_normal = X_normal * 2

# Adicionar anomalias
n_anomalies = 50
anomalies = np.random.uniform(low=-15, high=15, size=(n_anomalies, 2))

# Combinar
X = np.vstack([X_normal, anomalies])
y_true = np.hstack([np.zeros(len(X_normal)), np.ones(n_anomalies)])

# Embaralhar
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y_true = y_true[shuffle_idx]

print(f"Total samples: {len(X)}")
print(f"Normal: {sum(y_true == 0)}, Anomalies: {sum(y_true == 1)}")

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 2. ISOLATION FOREST
# ============================================

print("\n" + "="*50)
print("ISOLATION FOREST")
print("="*50)

iso_forest = IsolationForest(
    contamination=n_anomalies / len(X),  # Taxa esperada de anomalias
    random_state=42
)

y_pred_if = iso_forest.fit_predict(X_scaled)
# Converter -1 (anomalia) para 1
y_pred_if = (y_pred_if == -1).astype(int)

# Scores (quanto menor, mais anômalo)
anomaly_scores_if = -iso_forest.score_samples(X_scaled)

from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score

cm_if = confusion_matrix(y_true, y_pred_if)
print("\nConfusion Matrix:")
print(cm_if)

f1_if = f1_score(y_true, y_pred_if)
print(f"F1-Score: {f1_if:.4f}")

# ============================================
# 3. AUTOENCODER PARA DETECÇÃO
# ============================================

print("\n" + "="*50)
print("AUTOENCODER")
print("="*50)

# Treinar autoencoders apenas em dados normais
X_train_ae = X_scaled[y_true == 0]

# Construir autoencoder
def build_autoencoder(input_dim, encoding_dim=1):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(5, activation='relu')(inputs)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(5, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # Autoencoder completo
    autoencoder = models.Model(inputs, decoded)
    
    return autoencoder

autoencoder = build_autoencoder(input_dim=2, encoding_dim=1)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

autoencoder.summary()

# Treinar
print("\nTraining autoencoder...")
history = autoencoder.fit(
    X_train_ae, X_train_ae,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print("Training complete!")

# Fazer predições (reconstrução)
X_reconstructed = autoencoder.predict(X_scaled, verbose=0)

# Calcular erro de reconstrução
reconstruction_error = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

# Definir threshold (percentil 95 dos dados normais)
threshold = np.percentile(
    reconstruction_error[y_true == 0],
    95
)

y_pred_ae = (reconstruction_error > threshold).astype(int)

cm_ae = confusion_matrix(y_true, y_pred_ae)
print("\nAutoencoder Confusion Matrix:")
print(cm_ae)

f1_ae = f1_score(y_true, y_pred_ae)
print(f"F1-Score: {f1_ae:.4f}")

# ============================================
# 4. VISUALIZAR RESULTADOS
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Dados originais
axes[0, 0].scatter(X[y_true == 0, 0], X[y_true == 0, 1], label='Normal', alpha=0.6)
axes[0, 0].scatter(X[y_true == 1, 0], X[y_true == 1, 1], label='Anomaly', color='red')
axes[0, 0].set_title('Original Data')
axes[0, 0].legend()

# Isolation Forest
axes[0, 1].scatter(X[y_pred_if == 0, 0], X[y_pred_if == 0, 1], label='Normal', alpha=0.6)
axes[0, 1].scatter(X[y_pred_if == 1, 0], X[y_pred_if == 1, 1], label='Anomaly', color='red')
axes[0, 1].set_title('Isolation Forest Predictions')
axes[0, 1].legend()

# Anomaly scores (IF)
scatter = axes[0, 2].scatter(X[:, 0], X[:, 1], c=anomaly_scores_if, cmap='YlOrRd')
axes[0, 2].set_title('Isolation Forest Scores')
plt.colorbar(scatter, ax=axes[0, 2])

# Training history
axes[1, 0].plot(history.history['loss'], label='Train')
axes[1, 0].plot(history.history['val_loss'], label='Val')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Autoencoder Training')
axes[1, 0].legend()

# Autoencoder predictions
axes[1, 1].scatter(X[y_pred_ae == 0, 0], X[y_pred_ae == 0, 1], label='Normal', alpha=0.6)
axes[1, 1].scatter(X[y_pred_ae == 1, 0], X[y_pred_ae == 1, 1], label='Anomaly', color='red')
axes[1, 1].set_title('Autoencoder Predictions')
axes[1, 1].legend()

# Reconstruction error
scatter = axes[1, 2].scatter(X[:, 0], X[:, 1], c=reconstruction_error, cmap='YlOrRd')
axes[1, 2].axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
axes[1, 2].set_title('Reconstruction Error')
plt.colorbar(scatter, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('anomaly_detection.png')
plt.show()

# ============================================
# 5. COMPARAÇÃO DE MÉTODOS
# ============================================

print("\n" + "="*50)
print("METHOD COMPARISON")
print("="*50)

from sklearn.metrics import precision_score, recall_score, roc_auc_score

methods = {
    'Isolation Forest': {
        'predictions': y_pred_if,
        'scores': anomaly_scores_if
    },
    'Autoencoder': {
        'predictions': y_pred_ae,
        'scores': reconstruction_error
    }
}

for method_name, method_data in methods.items():
    y_pred = method_data['predictions']
    scores = method_data['scores']
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, scores)
    
    print(f"\n{method_name}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

print("\n" + "="*50)
print("Note: Higher threshold for anomaly detection increases recall but decreases precision")
print("="*50)
```

---

## Dicas de Uso e Customização

### Para Classificação de Imagens (CNN)

- **Transfer Learning**: Use modelos pré-treinados (ResNet, EfficientNet) para datasets pequenos
- **Data Augmentation**: Critique necessário com poucos dados
- **Batch Normalization**: Accelerate convergência
- **Dropout**: Previne overfitting em datasets pequenos

### Para Séries Temporais (LSTM)

- **Window Size**: Experimente diferentes (30-90 timesteps típicos)
- **Multiple Steps**: Para prever múltiplos passos futuros, use seq2seq
- **Stationarity**: Transforme série se não-estacionária (diferencing, log-transform)
- **Attention Mechanism**: Considere Transformers para séries muito longas

### Para Datasets Tabulares

- **Feature Engineering**: Mais importante que escolha de algoritmo
- **Imbalanced Data**: Use SMOTE ou ajuste class_weight
- **Feature Selection**: Reduz complexidade e melhora generalização
- **Ensemble**: Combine múltiplos modelos para melhor performance

### Para NLP

- **Pré-treinados**: Sempre prefira modelos fine-tuned (BERT, GPT)
- **Contexto**: LSTMs/RNNs para contexto curto, Transformers para longo
- **Embedding**: Use GloVe/FastText para features leves

### Para Detecção de Anomalias

- **Unsupervised**: Ideal quando anomalias são raras/desconhecidas
- **Hybrid**: Combine múltiplos métodos para robustez
- **Threshold Tuning**: Ajuste baseado em precision/recall trade-off

---

## Troubleshooting de Código

### Erro: "CUDA out of memory"
```python
# Reduzir batch size
batch_size = 16  # ao invés de 128

# Ou forçar CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Erro: "Shape mismatch"
```python
# Verificar shapes em cada passo
print(X.shape, y.shape)
print(model.output_shape)

# Use reshape
X = X.reshape(X.shape[0], -1)  # Flatten
```

### Modelo não converge
```python
# 1. Aumentar learning rate
optimizer = Adam(learning_rate=0.001)

# 2. Normalizar dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Reduzir complexidade
model = Sequential([
    Dense(64, activation='relu'),  # menos neurônios
    Dense(num_classes, activation='softmax')
])
```

---

**Sucesso na implementação! Use estes códigos como base e adapte para seus dados.**
