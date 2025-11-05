# 🤖 GUIA PROFUNDO E ESTRUTURADO DE MACHINE LEARNING
## Uma Abordagem Rigorosa, Modular e Prática

--- 

## 📋 INTRODUÇÃO

Este guia foi desenvolvido para profissionais que desejam dominar **Machine Learning** em profundidade, combinando rigor teórico com aplicações práticas. Estruturado em **7 módulos progressivos**, aborda desde fundamentos até tópicos avançados, sempre mantendo conexão com a literatura fundamental da área (Bishop, Hastie-Friedman, Domingos).

**Público-alvo**: Desenvolvedores e Analistas com boa base em Python e estatística, buscando compreensão profunda.

**Tempo de aprendizado estimado**:
- Fundamentos sólidos: 3-6 meses com dedicação 10-15h/semana
- Proficiência aplicada: 1-2 anos com projetos práticos
- Expertise: 3-5+ anos com experiência contínua

---

## 🎯 MÓDULO 1: FUNDAMENTOS DE MACHINE LEARNING

### 1.1 O que é Machine Learning?

Machine Learning (ML) refere-se a sistemas computacionais que **aprendem padrões a partir de dados** em vez de serem explicitamente programados. Fundamentalmente, o objetivo é criar modelos que generalizem bem para dados nunca vistos.

#### Diferenciação: Programação Tradicional vs Machine Learning

```
PROGRAMAÇÃO TRADICIONAL:
Dados + Regras → Computador → Resultado

MACHINE LEARNING:
Dados + Resultados Esperados → Computador → Regras (Modelo)
```

**Comparação Técnica:**

|        Aspecto        |     Programação Tradicional     |            Machine Learning             |
|-----------------------|---------------------------------|-----------------------------------------|
| **Abordagem**         | Rule-based, determinística      | Data-driven, probabilística             |
| **Adaptabilidade**    | Requer recodificação            | Adapta com novos dados                  |
| **Complexidade**      | Aumenta com casos especiais     | Aumenta com dados                       |
| **Explicitabilidade** | Código é a especificação        | Padrões latentes                        |
| **Escalabilidade**    | Limitada por lógica             | Escalável com dados                     |
| **Exemplo Prático**   | `if temperatura > 30: alerta()` | Modelo aprende relação multidimensional |

### 1.2 Ciclo de Vida de um Projeto de ML (8 Fases)

Todo projeto segue um ciclo bem definido:

```
1. DEFINIÇÃO DO PROBLEMA
   ↓
2. COLETA E EXPLORAÇÃO DE DADOS
   ↓
3. ANÁLISE EXPLORATÓRIA (EDA)
   ↓
4. PRÉ-PROCESSAMENTO E LIMPEZA
   ↓
5. FEATURE ENGINEERING E SELEÇÃO
   ↓
6. SELEÇÃO DE MODELO
   ↓
7. TREINAMENTO, VALIDAÇÃO E AJUSTE
   ↓
8. DEPLOY E MONITORAMENTO CONTÍNUO
```

**Detalhamento:**

- **Fase 1-2**: Entender problema de negócio, coletar dados relevantes (60% do tempo!)
- **Fase 3**: Visualizações, estatísticas, identificação de padrões
- **Fase 4**: Tratamento de missing values, outliers, normalização
- **Fase 5**: Criar features que captem informação relevante, seleção
- **Fase 6**: Comparar famílias de algoritmos (supervised, unsupervised, etc.)
- **Fase 7**: Cross-validation, hyperparameter tuning, seleção final
- **Fase 8**: Deploy em produção, monitorar performance, re-treinar conforme drift

### 1.3 Paradigmas de Aprendizado

#### 1.3.1 Aprendizado Supervisionado

Dados possuem **labels conhecidos** (y). O modelo aprende mapeamento X → y.

**Subtipo: Classificação**
- **Objetivo**: Prever categoria discreta
- **Exemplos**: Diagnóstico médico (sim/não), Classificação de iris (setosa/versicolor/virginica)
- **Métrica típica**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Algoritmos**: Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost

**Subtipo: Regressão**
- **Objetivo**: Prever valor contínuo
- **Exemplos**: Preço de imóvel, Temperatura, Vendas futuras
- **Métrica típica**: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), R² Score
- **Algoritmos**: Linear Regression, Decision Trees, Random Forest, Gradient Boosting

#### 1.3.2 Aprendizado Não-Supervisionado

Dados **sem labels**. Objetivo é descobrir estrutura, padrões, agrupamentos inerentes.

**Subtipo: Clustering (Agrupamento)**
- Particionar dados em grupos similares
- **Algoritmos**: K-means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Models
- **Métricas**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz

**Subtipo: Redução de Dimensionalidade**
- Reduzir número de features mantendo informação
- **Aplicações**: Visualização, compressão, remoção de ruído
- **Algoritmos**: PCA (linear), t-SNE, UMAP, Autoencoders

**Subtipo: Association Rules (Rule Mining)**
- Descobrir relações entre variáveis
- **Exemplo clássico**: Market basket ("clientes que compram pão e leite também compram manteiga")
- **Métricas**: Support, Confidence, Lift

#### 1.3.3 Aprendizado por Reforço

**Agent** interage com **environment**, recebe **rewards** por ações boas, **penalties** por ruins.

- **Componentes**: State, Action, Reward, Policy, Value Function
- **Algoritmos**: Q-Learning, Policy Gradient (PPO, A3C, REINFORCE), Actor-Critic
- **Aplicações**: Robótica, Jogos (AlphaGo, Chess), Recomendação
- **Característica**: Aprende através de trial-and-error

**Nota**: Diferente dos anteriores por ter loop de feedback.

#### 1.3.4 Aprendizado Semi-Supervisionado

Combina dados **labeled e unlabeled**. Útil quando labeling é caro.

- **Técnicas**: Self-training, Co-training, Pseudo-labeling
- **Aplicação**: NLP (muitos textos, alguns anotados)

### 1.4 Conceitos-Chave Fundamentais

#### Bias-Variance Tradeoff (Conceito Central)

Decomposição do erro esperado:

```
Erro Total = Bias² + Variance + Ruído Irreduzível
```

- **High Bias**: Modelo muito simples, underfitting
  - Sintomas: Train e test accuracy ambos baixos
  - Solução: Modelo mais complexo, mais features

- **High Variance**: Modelo muito complexo, overfitting
  - Sintomas: Train accuracy alto, test accuracy baixo
  - Solução: Regularização, mais dados, early stopping

- **Objetivo**: Encontrar sweet spot que minimize Bias² + Variance

#### Overfitting vs Underfitting

```
UNDERFITTING (High Bias):     BALANCED (Sweet Spot):     OVERFITTING (High Variance):
   \    /Train                    \  /Train                 \    Train
    \  /                           \/                         \___
     \/Test                        /\Test                        /\___Test
Test (baixo)                   Test (ótimo)                  Test (piora)
```

**Técnicas para Combater Overfitting:**
1. Regularização (L1/L2)
2. Cross-validation
3. Early stopping (em neural networks)
4. Dropout (redes neurais)
5. Mais dados (quando possível)
6. Feature selection (features menos relevantes)

#### Generalização

Capacidade do modelo de fazer boas predições em dados **nunca vistos** durante treinamento. É o verdadeiro objetivo do ML!

### 1.5 Aplicações Práticas Reais do ML

|     Domínio     |                                Aplicação                          |             Tipo           |                Desafio Principal               |
|-----------------|-------------------------------------------------------------------|----------------------------|------------------------------------------------|
| **Healthcare**  | Diagnóstico de doenças, descoberta de drogas, análise genômica    | Classificação/Segmentação  | Poucos dados, interpretabilidade crítica       |
| **Finanças**    | Detecção de fraude, previsão de mercado, credit scoring           | Classificação/Regressão    | Dados desbalanceados, conformidade regulatória |
| **Varejo**      | Recomendação de produtos, previsão de churn, demand forecasting   | Clustering/Regressão       | Dados faltantes, popularidade dinâmica         |
| **Transportes** | Veículos autônomos, reconhecimento de placas, otimização de rotas | Computer Vision/Otimização | Volume de dados, latência baixa                |
| **NLP**         | Tradução automática, chatbots, análise de sentimentos             | NLP/Classificação          | Contexto, nuances linguísticas                 |
| **Imagem**      | Detecção de objetos, segmentação semântica, reconhecimento facial | Computer Vision            | Robustez, viés de dados                        |
| **Energia**     | Previsão de demanda, manutenção preditiva                         | Regressão/Anomalia         | Série temporal, sazonalidade                   |

---

## 🔧 MÓDULO 2: PRÉ-PROCESSAMENTO E ENGENHARIA DE DADOS

### 2.1 A Realidade dos Dados: "Garbage In, Garbage Out"

**Estatística impactante**: 60-80% do tempo em projetos de ML é gasto em preparação de dados!

Dados do mundo real são **sujos**: incompletos, ruidosos, inconsistentes. Um pipeline robusto de pré-processamento é **essencial**.

### 2.2 Tratamento de Missing Values

#### Estratégia 1: Remoção
```python
# Se % missing < 5%, pode remover
df_clean = df.dropna()

# Se missing em features específicas
df_clean = df.dropna(subset=['important_feature'])
```

**Quando usar**: Quando realmente poucos dados faltam

#### Estratégia 2: Imputação com Estatística Simples
```python
# Imputação Mean/Median/Mode
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')  # Robusto a outliers
X_imputed = imputer.fit_transform(X)
```

**Quando usar**: Dados MCAR (Missing Completely At Random), features numéricas

#### Estratégia 3: KNN Imputation
```python
from sklearn.impute import KNNImputer

knn_imp = KNNImputer(n_neighbors=5)
X_imputed = knn_imp.fit_transform(X)
```

**Quando usar**: Dados com estrutura local, relações entre features

#### Estratégia 4: Imputação Preditiva
```python
# Treina modelo para prever missing values
from sklearn.ensemble import RandomForestRegressor

def impute_with_model(X, feature_with_missing):
    train_idx = X[feature_with_missing].notna()
    X_train = X.loc[train_idx, other_features]
    y_train = X.loc[train_idx, feature_with_missing]
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    X_missing = X.loc[~train_idx, other_features]
    X.loc[~train_idx, feature_with_missing] = model.predict(X_missing)
    return X
```

**Quando usar**: Missing pattern complexo, muitas features relacionadas

#### Estratégia 5: Indicator Variable
```python
# Cria feature binária indicando missing
X['feature_was_missing'] = X['feature'].isna()
X['feature'].fillna(X['feature'].median(), inplace=True)
```

**Quando usar**: Missing pode ser informativo (ex: "não respondeu")

### 2.3 Tratamento de Outliers

#### Detecção: Método IQR
```python
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1

outliers = (df['feature'] < Q1 - 1.5*IQR) | (df['feature'] > Q3 + 1.5*IQR)
df_clean = df[~outliers]
```

#### Detecção: Z-score
```python
from scipy import stats

z_scores = np.abs(stats.zscore(X))
outliers = (z_scores > 3).all(axis=1)
X_clean = X[~outliers]
```

#### Detecção: Isolation Forest (Algoritmo ML)
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05, random_state=42)
labels = iso_forest.fit_predict(X)  # -1 = outlier, 1 = normal
X_clean = X[labels == 1]
```

**Vantagem**: Detecta outliers multidimensionais (anomalias em combinação de features)

#### Tratamento: Opções
1. **Remoção**: Se claramente erro de medição
2. **Cap/Floor**: Limitar a máximo/mínimo razoável
3. **Transformação**: log(x), boxcox(x)
4. **Manter**: Se podem ser insights válidos

### 2.4 Feature Scaling (Normalização)

**Por que escalar?**
- Algoritmos baseados em distância (KNN, KMeans, SVM) são sensíveis
- Regularização (L1/L2) assume features na mesma escala
- Convergência mais rápida em gradient descent

#### StandardScaler (Z-score normalization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Resultado: média=0, std=1
```

**Fórmula**: (x - μ) / σ

#### MinMaxScaler (Normalização [0,1])
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Resultado: [0, 1]
```

**Fórmula**: (x - min) / (max - min)

#### RobustScaler (Robusto a outliers)
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
# Usa mediana e IQR
```

**Fórmula**: (x - mediana) / IQR

**Quando NÃO escalar**: Decision Trees, Random Forest (invariantes a escala)

### 2.5 Encoding de Variáveis Categóricas

#### One-Hot Encoding (para nominais)
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = ohe.fit_transform(X_categorical)
# Cria coluna binária para cada categoria
```

**Exemplo**: Color=['Red', 'Blue', 'Green'] →
```
color_Red  color_Blue  color_Green
1          0           0
0          1           0
0          0           1
```

#### Label Encoding (para ordinais)
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_encoded = le.fit_transform(X_categorical)
# Color=['Red', 'Blue', 'Green'] → [2, 0, 1]
```

**Cuidado**: Pode introduzir ordem indesejada em árvores!

#### Target Encoding (baseado em média do target)
```python
target_encoding = df.groupby('category')['target'].mean()
X_encoded = X_categorical.map(target_encoding)
```

**Risco**: Data leakage! Use apenas com validação cruzada apropriada.

### 2.6 Balanceamento de Classes (Imbalanced Data)

**Problema**: 99% classe 0, 1% classe 1 → modelo aprende a prever sempre 0

#### Técnica 1: Oversampling (SMOTE)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.8)
X_balanced, y_balanced = smote.fit_resample(X, y)
# Gera amostras sintéticas da classe minoritária
```

#### Técnica 2: Undersampling
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=0.8)
X_balanced, y_balanced = rus.fit_resample(X, y)
# Remove amostras da classe maioritária
```

#### Técnica 3: Pesos nas Classes
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y), 
                                     y=y)
model = LogisticRegression(class_weight='balanced')
```

#### Técnica 4: Threshold Adjustment
```python
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.3).astype(int)  # Threshold customizado
```

### 2.7 Pipeline Completo de Pré-processamento

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Definir processadores por tipo de feature
numeric_features = ['age', 'income']
categorical_features = ['city', 'gender']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline completo: Preprocessing + Modelo
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Usar assim:
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
```

**Vantagem**: Aplicar transformações automaticamente, evitar data leakage

---

## 🎓 MÓDULO 3: ALGORITMOS SUPERVISIONADOS - REGRESSÃO E CLASSIFICAÇÃO

### 3.1 Regressão Linear

**Modelo Matemático:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Onde:
- y: variável dependente (target)
- x_i: features independentes
- β_i: coeficientes (pesos)
- ε: erro (ruído)
```

**Objetivo**: Minimizar soma de erros ao quadrado (SSE, Least Squares)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")  # 0-1, maior é melhor
print(f"Coeficientes: {model.coef_}")
```

**Interpretação:**
- β₁ = 2.5 significa: aumentar x₁ em 1 unidade → y aumenta 2.5 em média

**Vantagens**:
- Rápido, escalável
- Interpretável
- Bom baseline

**Desvantagens**:
- Assume relação linear
- Sensível a outliers
- Sem feature interactions

### 3.2 Regressão Logística (para Classificação Binária)

**Nota Importante**: Apesar do nome, é para **CLASSIFICAÇÃO**, não regressão!

**Modelo:**
```
P(y=1|x) = 1 / (1 + e^(-z))
Onde z = β₀ + β₁x₁ + ... + βₙxₙ

Isso é a função sigmoide: output ∈ [0, 1] (probabilidade)
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # 0 ou 1
y_pred_proba = model.predict_proba(X_test)  # Probabilidades

print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.3f}")
```

**Características**:
- Probabilística (outputs são probabilidades)
- Linear (no espaço log-odds)
- Interpretável
- Bom para dados balanceados

### 3.3 Árvores de Decisão

**Ideia**: Dividir recursivamente o espaço de features usando regras if-then

```
             [idade < 35]
            /            \
       NÃO /              \ SIM
          /                \
    [renda > 50k]      [score > 600]
```

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(
    max_depth=5,              # Profundidade máxima
    min_samples_split=10,     # Amostras mínimas para dividir
    min_samples_leaf=5,       # Amostras mínimas em folha
    random_state=42
)
model.fit(X_train, y_train)

# Visualizar
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=feature_names, filled=True, rounded=True)
plt.show()
```

**Vantagens**:
- Interpretável (simula pensamento humano)
- Não requer scaling
- Captura não-linearidades
- Rápido para predição

**Desvantagens**:
- Propenso a overfitting
- Instável (pequenas mudanças → árvore completamente diferente)
- Necessário regularização via max_depth, min_samples_split

### 3.4 Ensembles: Combinando Força de Múltiplos Modelos

**Conceito Base**: "Sabedoria das multidões" - múltiplos modelos combinados superam indivíduo

#### 3.4.1 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,      # Número de árvores
    max_depth=10,
    random_state=42,
    n_jobs=-1              # Usar todos os cores
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Feature importance
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df)
```

**Funcionamento**:
1. Cria N árvores em subamostras aleatórias dos dados
2. Em cada árvore, usa subset aleatório de features
3. Cada árvore "vota": classificação = moda, regressão = média
4. Resultado final = votação agregada

**Por que funciona**: Decorrelação entre árvores → reduz variance

**Vantagens**:
- Muito melhor que Decision Tree individual
- Robusto a outliers
- Feature importance inteligível
- Parallelizável

**Desvantagens**:
- Black-box comparado a árvore individual
- Lento para datasets gigantescos

#### 3.4.2 Gradient Boosting (XGBoost, LightGBM)

**Ideia Fundamental**: Ao contrário de RF que treina em paralelo, Boosting treina **sequencialmente**. Cada árvore tenta **corrigir erros da anterior**.

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,      # Taxa de aprendizado (shrinkage)
    subsample=0.8,          # % das amostras por árvore
    colsample_bytree=0.8,   # % das features por árvore
    random_state=42
)

# Early stopping para evitar overfitting
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=10,
              verbose=False)

y_pred = xgb_model.predict(X_test)
```

**LightGBM** (mais rápido, menos memória):
```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    num_leaves=31,           # LightGBM usa leaves não depth
    random_state=42
)
lgb_model.fit(X_train, y_train)
```

**Performance**: XGBoost/LightGBM > Random Forest >> Decision Tree

**Por que tão bom:**
- Seqüencial refinamento de erros
- Regularização automática (learning_rate, max_depth)
- Suporta early stopping
- Muito competitivo em Kaggle

**Desvantagens**:
- Mais lento para treinar
- Hiperparâmetros sensíveis

#### 3.4.3 Support Vector Machines (SVM)

**Ideia**: Encontrar hiperplano que **maximiza margem** entre classes

```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# IMPORTANTE: SVM requer dados escalados!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(
    kernel='rbf',           # 'linear', 'rbf', 'poly'
    C=1.0,                  # Regularização (menor = mais suave)
    gamma='scale',          # Influência de um ponto
    probability=True        # Para predict_proba
)

svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
```

**Kernels (Truques para dados não-lineares)**:
- **Linear**: Dados linearmente separáveis
- **RBF (Radial Basis Function)**: Padrão, bom geral
- **Polynomial**: Grau customizável
- **Sigmoid**: Menos comum

**Matematicamente**: Transforma dados em dimensão mais alta onde são lineares

**Vantagens**:
- Eficaz em alta dimensão
- Memória eficiente
- Interpretação geométrica clara

**Desvantagens**:
- Lento em datasets grandes
- Requer scaling
- Ajuste de hiperparâmetros não trivial

### 3.5 K-Nearest Neighbors (KNN)

**Princípio**: "Você é amigo de seus vizinhos". Para prever novo ponto, usa classe de seus K vizinhos mais próximos.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Escalar dados (distância é sensível!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(
    n_neighbors=5,          # Quantos vizinhos considerar
    weights='distance',     # 'uniform' ou 'distance'
    metric='euclidean'      # 'euclidean', 'manhattan', etc
)

knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
```

**Características**:
- **Non-parametric**: Não assume distribuição
- **Lazy learner**: Não treina, busca em tempo de predição
- **Sensível** a features irrelevantes

**Escolha de K**:
- K pequeno: Overfitting (ruído influencia)
- K grande: Underfitting (perda de detalhes)
- Regra: K ≈ √n ou verificar com validação cruzada

**Vantagens**:
- Implementação simples
- Não linear
- Bom para small data

**Desvantagens**:
- Lento em tempo de predição (compara com todos)
- Sensível a escala
- Afetado por features irrelevantes

### 3.6 Validação Cruzada (Cross-Validation)

**Problema**: Se treinar e testar no mesmo conjunto, vazamos informação (data leakage)

**Solução**: Estratificar dados

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model, X, y, 
    cv=skf, 
    scoring='accuracy',
    n_jobs=-1
)

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Processo:**
```
Dataset dividido em 5 folds:
[fold1] [fold2] [fold3] [fold4] [fold5]

Iteração 1: Treina em [2,3,4,5], testa em [1]
Iteração 2: Treina em [1,3,4,5], testa em [2]
...
Iteração 5: Treina em [1,2,3,4], testa em [5]

Score final = média dos 5 scores
```

**StratifiedKFold**: Mantém proporção de classes em cada fold (importante para classificação!)

### 3.7 Hyperparameter Tuning (Busca de Hiperparâmetros)

**Grid Search** (brute force):
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'learning_rate': [0.01, 0.1, 0.5]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
```

**Otuna** (inteligente com Bayesian Optimization):
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
    }
    
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    return model.score(X_val, y_val)  # Otimizar métrica

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best trial: {study.best_trial.params}")
```

**Vantagens Optuna sobre GridSearch**:
- Mais eficiente (não testa todas combinações)
- Usa histórico para guiar busca
- Suporta pruning (para early stopping)

### 3.8 Métricas de Classificação (Binária)

```
Matriz de Confusão:

                Predicted
              Pos      Neg
Actual Pos    TP       FN     (P)
       Neg    FP       TN     (N)
           (PP)      (PN)
```

#### Definições

**Accuracy (Taxa de Acerto)**
```
Accuracy = (TP + TN) / Total
```
- Percentagem geral de acertos
- **Problema**: Enganosa em dados desbalanceados

**Precision (Precisão)**
```
Precision = TP / (TP + FP)
```
- De POSITIVOS preditos, quantos eram realmente?
- "Qual a qualidade do meu sistema de alertas?"
- Minimiza false alarms

**Recall (Cobertura)**
```
Recall = TP / (TP + FN)
```
- De POSITIVOS reais, quantos detectei?
- "Quantas fraudes realmente peguei?"
- Minimiza missed cases

**F1-Score (Harmônico)**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Balanceia precision e recall
- Usa quando FP e FN têm custos iguais

**ROC-AUC (Area Under ROC Curve)**
```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.legend()
plt.show()
```

- Independente de threshold
- AUC = 1: Perfeito, AUC = 0.5: Random

```python
# Exemplo Completo
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.3f}")
print("\n" + classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

**Quando usar cada uma:**
- **Accuracy**: Dados balanceados, custo igual para FP e FN
- **Precision**: Importante minimizar false positives (ex: spam detection)
- **Recall**: Importante minimizar false negatives (ex: detecção de fraude)
- **F1**: Quando FP e FN têm custos iguais mas dados desbalanceados
- **ROC-AUC**: Comparação robusta entre modelos

### 3.9 Métricas de Regressão

**MAE (Mean Absolute Error)**
```python
mae = mean_absolute_error(y_test, y_pred)
```
- Erro médio absoluto
- Mesma unidade que y
- Robusto a outliers

**RMSE (Root Mean Squared Error)**
```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```
- Penaliza erros grandes
- Mesma unidade que y
- Sensível a outliers

**R² Score**
```python
r2 = r2_score(y_test, y_pred)
```
- Proporção de variância explicada
- Range: 0 a 1 (maior melhor)
- R² = 1: Perfeito, R² = 0: Modelo igual à média

---

## 🔍 MÓDULO 4: MODELOS NÃO-SUPERVISIONADOS E REDUÇÃO DE DIMENSIONALIDADE

### 4.1 Clustering: K-Means

**Objetivo**: Particionar dados em K grupos (clusters) de objetos similares

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,              # Múltiplas inicializações
    max_iter=300
)

labels = kmeans.fit_predict(X)  # Cluster de cada ponto

# Visualizar
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.show()
```

**Algoritmo**:
1. Inicializar K centroids aleatoriamente
2. Atribuir cada ponto ao centroid mais próximo
3. Recalcular centroids como média dos pontos
4. Repetir 2-3 até convergência

**Encontrar K ótimo (Elbow Method)**:
```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
# Procurar por "cotovelo" (mudança de inclinação)
```

**Vantagens**:
- Simples e rápido
- Escalável
- Interpretável

**Desvantagens**:
- Assume clusters esféricos
- Sensível à inicialização (usar n_init > 1)
- Necessário especificar K
- Sensível a escala

### 4.2 DBSCAN: Clustering Baseado em Densidade

**Ideia**: Agrupa pontos densamente conectados, marca outliers como ruído

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Encontrar eps ótimo (k-distance graph)
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
distances = np.sort(distances[:, 4], axis=0)

plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('5th Nearest Neighbor Distance')
plt.title('K-distance Graph')
plt.show()
# Procurar por "joelho" para definir eps

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# labels = -1 para pontos de ruído
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters: {n_clusters}, Ruído: {n_noise}")
```

**Parâmetros**:
- **eps**: Raio máximo de vizinhança
- **min_samples**: Quantos pontos devem estar em eps para ser core point

**Vantagens**:
- Não precisa especificar K
- Detecta ruído
- Encontra clusters de forma arbitrária
- Teoricamente bem motivado

**Desvantagens**:
- Sensível a eps (difícil de ajustar)
- Problematário em dados heterogêneos (densidades variáveis)
- Computacionalmente caro em alta dimensão

### 4.3 Gaussian Mixture Models (GMM)

**Ideia**: Dados são gerados por mistura de distribuições gaussianas

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,
    random_state=42,
    covariance_type='full'  # 'full', 'tied', 'diag', 'spherical'
)

labels = gmm.fit_predict(X)  # Hard assignment
probabilities = gmm.predict_proba(X)  # Soft assignment (probabilidades)

# Selecionar n_components com BIC
bic_scores = []
for n in range(1, 11):
    gmm_temp = GaussianMixture(n_components=n, random_state=42)
    gmm_temp.fit(X)
    bic_scores.append(gmm_temp.bic(X))

plt.plot(range(1, 11), bic_scores)
plt.xlabel('n_components')
plt.ylabel('BIC')
plt.show()
```

**Diferenças**:
- **K-Means**: Hard assignment (cada ponto pertence exatamente a um cluster)
- **GMM**: Soft assignment (probabilidade de pertencer a cada cluster)

**Vantagens**:
- Probabilístico (outputs são probabilidades)
- Flexible (diferentes covariance types)
- BIC para seleção de modelo

### 4.4 Redução de Dimensionalidade

#### 4.4.1 PCA (Principal Component Analysis)

**Objetivo**: Encontrar direções de máxima variância, projetar dados nelas

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

# Escolher componentes para 95% variância
n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"N componentes para 95%: {n_components_95}")

# Visualizar
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.show()
```

**Matematicamente**:
- Encontra autovetores da matriz de covariância
- PCs são combinações lineares das features originais
- Ordenadas por variância (PC1 > PC2 > ...)

**Características**:
- Linear
- Determinístico (sem aleatoriedade)
- Rápido
- Interpretável (cada PC é combinação das features)

**Limitações**:
- Não otimizado para visualização
- Perde estrutura local
- Assume variância = informação

#### 4.4.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Objetivo**: Visualizar dados em 2D/3D preservando **estrutura local**

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,          # Balanço entre local e global (default 5-50)
    n_iter=1000,
    random_state=42,
    n_jobs=-1
)

X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE Visualization')
plt.show()
```

**Características**:
- Não-linear
- Excelente para visualização
- Preserva vizinhança local

**Desvantagens**:
- Lento em dados grandes
- Perplexity é sensível
- Não pode transformar novos dados
- Estrutura global pode ser distorcida

#### 4.4.3 UMAP (Uniform Manifold Approximation and Projection)

**Objetivo**: Semelhante a t-SNE mas mais rápido e preserva global structure

```python
import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,         # Equivalente a perplexity do t-SNE
    min_dist=0.1,          # Compacidade local
    metric='euclidean',
    random_state=42
)

X_umap = reducer.fit_transform(X)

# Pode transformar novos dados!
X_new_umap = reducer.transform(X_new)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
plt.title('UMAP Visualization')
plt.show()
```

**Comparação: PCA vs t-SNE vs UMAP**

|          Critério      |    PCA    | t-SNE |  UMAP  |
|------------------------|-----------|-------|--------|
| **Velocidade**         | Rápido    | Lento | Rápido |
| **Interpretabilidade** | Alta      | Baixa | Baixa  |
| **Preserva Local**     | Não       | Sim   | Sim    |
| **Preserva Global**    | Sim       | Não   | Sim    |
| **Transformar novos**  | Sim       | Não   | Sim    |
| **Escalabilidade**     | Excelente | Pobre | Boa    |

**Recomendação**: 
- **Exploração rápida**: PCA
- **Visualização fina clusters**: t-SNE
- **Produção**: UMAP

### 4.5 Métricas de Avaliação de Clustering

**Silhouette Score** (-1 a 1, maior melhor)
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
# 1: Clusters bem definidos
# 0: Clusters se sobrepõem
# -1: Pontos em clusters errados
```

**Davies-Bouldin Index** (menor melhor)
```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X, labels)
# Razão entre separação e compacidade
```

**Calinski-Harabasz Index** (maior melhor)
```python
from sklearn.metrics import calinski_harabasz_score

score = calinski_harabasz_score(X, labels)
# Razão entre separação e dispersão intra-cluster
```

---

## 🚀 MÓDULO 5: PIPELINE COMPLETO E DEPLOY

### 5.1 Estrutura Profissional de Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# 1. Definir features por tipo
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'gender', 'category']

# 2. Criar transformadores específicos
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# 3. Combinar transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # Descartar features não listadas
)

# 4. Pipeline completo: Preprocessing + Modelo
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

# 5. Usar
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
score = full_pipeline.score(X_test, y_test)
```

**Vantagem**: Evita data leakage! Transformações aplicadas corretamente.

### 5.2 Persistência de Modelos

#### Joblib (Recomendado para scikit-learn)
```python
import joblib

# Salvar
joblib.dump(full_pipeline, 'model_pipeline.pkl')

# Carregar
model = joblib.load('model_pipeline.pkl')
y_pred = model.predict(X_test)
```

#### MLflow (Rastreamento completo)
```python
import mlflow
import mlflow.sklearn

mlflow.start_run()

# Log de parâmetros
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)

# Train model
model.fit(X_train, y_train)

# Log de métricas
accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)

# Registrar modelo
mlflow.sklearn.log_model(model, "random_forest_model")

mlflow.end_run()

# Visualizar: mlflow ui (browser na porta 5000)
```

**Rastreia**: Parâmetros, métricas, artefatos, versões

### 5.3 API com FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="ML Prediction API", version="1.0")

# Carregar modelo na inicialização
model = joblib.load('model_pipeline.pkl')

# Definir schema de entrada
class PredictionRequest(BaseModel):
    age: int
    income: float
    score: float
    city: str
    gender: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Fazer predição para novo ponto"""
    
    # Preparar dados
    input_data = {
        'age': [request.age],
        'income': [request.income],
        'score': [request.score],
        'city': [request.city],
        'gender': [request.gender]
    }
    df = pd.DataFrame(input_data)
    
    # Predição
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].max()
    
    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability)
    )

@app.get("/health")
def health_check():
    """Health check do servidor"""
    return {"status": "OK"}

# Rodar: uvicorn main:app --reload
# URL: http://localhost:8000/docs
```

### 5.4 Interface com Streamlit

```python
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="ML Predictor", layout="wide")

st.title("🤖 Machine Learning Prediction App")

# Carregar modelo (cache para performance)
@st.cache_resource
def load_model():
    return joblib.load('model_pipeline.pkl')

model = load_model()

# Sidebar para inputs
st.sidebar.header("Input Features")

age = st.sidebar.slider('Age', 18, 100, 30)
income = st.sidebar.number_input('Income ($)', min_value=0, value=50000)
score = st.sidebar.slider('Score', 0.0, 1.0, 0.5)
city = st.sidebar.selectbox('City', ['New York', 'LA', 'Chicago'])
gender = st.sidebar.radio('Gender', ['Male', 'Female'])

# Preparar dados
input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'score': [score],
    'city': [city],
    'gender': [gender]
})

# Predição
if st.sidebar.button('🔮 Make Prediction'):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediction", "Class 1" if prediction == 1 else "Class 0")
    
    with col2:
        st.metric("Confidence", f"{max(probability):.1%}")
    
    # Mostrar probabilidades
    st.bar_chart(pd.DataFrame({
        'Class 0': [probability[0]],
        'Class 1': [probability[1]]
    }))

# Rodar: streamlit run app.py
```

### 5.5 Monitoramento e Re-treinamento

```python
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, baseline_accuracy=0.85, threshold_drop=0.05):
        self.baseline_accuracy = baseline_accuracy
        self.threshold_drop = threshold_drop
        self.history = []
    
    def check_drift(self, y_true, y_pred):
        """Detecta degradação de performance"""
        from sklearn.metrics import accuracy_score
        
        current_accuracy = accuracy_score(y_true, y_pred)
        
        self.history.append({
            'timestamp': datetime.now(),
            'accuracy': current_accuracy
        })
        
        # Verificar se accuracy caiu abaixo do threshold
        if current_accuracy < self.baseline_accuracy - self.threshold_drop:
            logger.warning(f"⚠️ MODEL DRIFT DETECTED!")
            logger.warning(f"   Current accuracy: {current_accuracy:.2%}")
            logger.warning(f"   Baseline: {self.baseline_accuracy:.2%}")
            return True  # Sinal para re-treinar
        
        return False
    
    def should_retrain(self):
        """Lógica para decidir se deve re-treinar"""
        if len(self.history) < 10:
            return False
        
        # Se últimas 3 medições caíram, re-treinar
        recent = [h['accuracy'] for h in self.history[-3:]]
        return all(a < self.baseline_accuracy for a in recent)

# Uso em pipeline de produção
monitor = ModelMonitor()

# Diariamente, avaliar em novo dataset
new_predictions = model.predict(new_data)
needs_retrain = monitor.check_drift(y_new, new_predictions)

if needs_retrain:
    logger.info("🔄 Re-training model...")
    # Re-treinar com dados recentes
    model.fit(X_recent, y_recent)
    joblib.dump(model, f'model_backup_{datetime.now().strftime("%Y%m%d")}.pkl')
```

---

## 🎯 MÓDULO 6: INTERPRETABILIDADE E EXPLICABILIDADE (XAI)

### 6.1 Por Que Explicabilidade é Crítica?

1. **Conformidade Regulatória**: GDPR, Fair Lending laws exigem explicações
2. **Confiança**: Usuários confiam em decisões que entendem
3. **Debug**: Identificar problemas no modelo
4. **Detecção de Bias**: Garantir justiça algorítmica

### 6.2 SHAP (SHapley Additive exPlanations)

**Fundamento Teórico**: Valores de Shapley da teoria dos jogos

```python
import shap
import matplotlib.pyplot as plt

# Para modelo tree-based
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (importância global)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Dependence plot (relação entre feature e output)
shap.dependence_plot('age', shap_values, X_test, feature_names=feature_names)

# Force plot (predição individual)
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0:1])

# Waterfall plot (mais intuitivo)
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
    feature_names=feature_names
))
```

**Interpretação**:
- SHAP value = quanto uma feature contribuiu para distanciar a predição da baseline
- Valor positivo = contribuiu para output positivo
- Valor negativo = contribuiu para output negativo

**Vantagens SHAP**:
- Teoricamente fundamentado
- Global e local
- Comparável entre features
- Funciona com qualquer modelo

### 6.3 LIME (Local Interpretable Model-agnostic Explanations)

**Ideia**: Criar modelo simples (linear) que aproxima modelo complexo localmente

```python
import lime
import lime.tabular

explainer = lime.tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Negative', 'Positive'],
    mode='classification'
)

# Explicar predição individual
exp = explainer.explain_instance(X_test[0], model.predict_proba)

# Visualizar
exp.show_in_notebook()

# Obtém features e seus pesos
exp.as_list()  # Retorna lista de (feature, weight) tuples
```

**Interpretação**:
- Perturba features ao redor da instância
- Treina modelo linear nos resultados
- Coeficientes do linear = importância local

**Comparação: SHAP vs LIME**

|          Aspecto       |            SHAP           |              LIME               |
|------------------------|---------------------------|---------------------------------|
| **Teórico**            | Rigoroso (Shapley values) | Heurístico                      |
| **Escopo**             | Global + local            | Local                           |
| **Estabilidade**       | Consistente               | Variável (amostragem aleatória) |
| **Velocidade**         | Mais lento                | Rápido                          |
| **Interpretabilidade** | Maior                     | Menor                           |

**Recomendação**: Use SHAP quando possível, LIME para modelos que SHAP não suporta

### 6.4 Permutation Feature Importance

```python
from sklearn.inspection import permutation_importance

importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance.importances_mean,
    'std': importance.importances_std
}).sort_values('importance', ascending=False)

print(importance_df)
```

**Ideia**: Embaralha cada feature, mede queda na métrica
- Grande queda = feature importante
- Pequena queda = feature não importante

### 6.5 Ética e Detecção de Viés

```python
from fairness_metrics import calculate_disparate_impact

# Verificar bias em subgrupos
def check_disparate_impact(X, y, y_pred, sensitive_feature, threshold=1.25):
    """
    Disparate impact ratio (4/5 rule):
    Taxa de seleção minoritário / taxa maioritário
    Se < 0.8 ou > 1.25 → possível discriminação
    """
    
    results = {}
    
    for group in X[sensitive_feature].unique():
        mask = X[sensitive_feature] == group
        group_rate = y_pred[mask].mean()
        results[group] = group_rate
    
    rates = list(results.values())
    di_ratio = min(rates) / max(rates)
    
    print(f"Disparate Impact Ratio: {di_ratio:.3f}")
    print(f"Suspeito de bias: {di_ratio < 0.8 or di_ratio > 1.25}")
    
    return results

# Exemplo
check_disparate_impact(X_test, y_test, y_pred, 'gender')
```

---

## 🚀 MÓDULO 7: TÓPICOS AVANÇADOS

### 7.1 Ensembles Avançados

#### Stacking
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Nível 0: Modelos base
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Nível 1: Meta-modelo
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_clf.fit(X_train, y_train)
```

**Funcionamento**:
1. Treina N modelos base no fold training
2. Modelos base fazem predições no fold validation (meta-features)
3. Meta-modelo treina nas meta-features
4. Final: predições dos base → meta-modelo

**Por que funciona**: Meta-modelo aprende quando confiar em cada base model

#### Voting
```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100)),
        ('svm', SVC(probability=True))
    ],
    voting='soft'  # 'soft'=média probabilities, 'hard'=moda classes
)

voting.fit(X_train, y_train)
```

### 7.2 AutoML e Hyperparameter Optimization

#### Optuna com Pruning
```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

def objective(trial):
    # Sugerer hiperparâmetros
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    
    model = XGBClassifier(**params, random_state=42)
    
    # Cross-validation com early stopping
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_t, y_t,
                  eval_set=[(X_v, y_v)],
                  early_stopping_rounds=20,
                  verbose=False)
        
        score = model.score(X_v, y_v)
        scores.append(score)
        
        # Pruning: parar se performance ruim
        trial.report(score, len(scores)-1)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

# Estudar
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(),  # Bayesian optimization
    pruner=MedianPruner()  # Prune trials ruins cedo
)

study.optimize(objective, n_trials=100, show_progress_bar=True)

# Resultados
print(f"Best trial: {study.best_trial.number}")
print(f"Best params: {study.best_trial.params}")
print(f"Best value: {study.best_value:.3f}")
```

#### Ray Tune (Distribuído)
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import Stopper

def train_model(config):
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)
    accuracy = model.score(X_val, y_val)
    tune.report(accuracy=accuracy)

scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='accuracy',
    mode='max',
    max_t=10,           # Iterações máximas
    grace_period=1      # Iterações antes de considerar pruning
)

result = tune.run(
    train_model,
    name='rf_tuning',
    config={
        'n_estimators': tune.randint(50, 500),
        'max_depth': tune.randint(3, 30),
        'min_samples_split': tune.randint(2, 20)
    },
    num_samples=20,     # Número de trials paralelos
    scheduler=scheduler,
    progress_reporter=tune.CLIReporter(metric_columns=['accuracy']),
    verbose=1
)

best_config = result.get_best_config(metric='accuracy', mode='max')
```

### 7.3 Transfer Learning

**Conceito**: Usar conhecimento aprendido em tarefa grande → adaptar para tarefa pequena

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model
import tensorflow as tf

# Carregar modelo pré-treinado (ImageNet weights)
base_model = VGG16(
    weights='imagenet',
    include_top=False,      # Remove classificador final
    input_shape=(224, 224, 3)
)

# Congelar pesos da base
base_model.trainable = False

# Construir modelo
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # Usar batch norm training=False
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar com congelamento
print("Training with frozen base...")
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# Fine-tuning: descongelar últimas camadas
print("Fine-tuning...")
base_model.trainable = True

# Recongelar primeiras camadas
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Compilar com learning rate menor
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

**Estratégias**:
1. **Feature Extraction**: Congelar base, treinar apenas head
2. **Fine-tuning Parcial**: Descongelar últimas camadas
3. **Fine-tuning Total**: Descongelar tudo (aprendizado menor)

### 7.4 Few-Shot Learning (Aprender com Poucos Exemplos)

**Paradigma**: Meta-learning - aprender a aprender

```python
from prototypical_networks import PrototypicalNetworks

# Treinar meta-modelo em muitas tarefas
meta_model = PrototypicalNetworks(embedding_dim=64)

# Tarefa = 5-way, 1-shot (5 classes, 1 exemplo cada)
meta_model.meta_fit(
    train_tasks,        # Muitas tasks de treino
    val_tasks,
    num_tasks=1000,
    ways=5,
    shots=1
)

# Adaptar a novo problema (5 exemplos novos por classe)
few_shot_data = load_few_shot_task()  # 5 samples × 5 classes
meta_model.adapt(few_shot_data)

# Predizer em novo domínio
predictions = meta_model.predict(query_data)
```

**Vantagem**: Pode aprender novo conceito com muito poucos exemplos

### 7.5 Machine Learning em Larga Escala

#### Spark MLlib
```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# Inicializar Spark
spark = SparkSession.builder \
    .appName("ML") \
    .getOrCreate()

# Carregar dados
df = spark.read.csv('large_data.csv', header=True, inferSchema=True)

# Pipeline Spark
feature_cols = ['age', 'income', 'score']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=10,
    labelCol='target',
    featuresCol='features'
)

pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(df)

# Predições
predictions = model.transform(df)
predictions.select('target', 'prediction').show(5)
```

**Vantagens**:
- Distributed processing
- Integração com Hadoop/YARN
- MLlib com muitos algoritmos

**Desvantagens**:
- Overhead de distribuição
- API diferente de scikit-learn
- Mais lento para small data

#### Dask (Python distribuído)
```python
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.ensemble import RandomForestClassifier

# Carregar dados grandes
ddf = dd.read_csv('large_data.csv')

X = ddf[feature_cols]
y = ddf['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar com dask
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Predições
y_pred = model.predict(X_test)
```

**Vantagens**:
- Usa familiar Scikit-learn API
- Integração com NumPy/Pandas
- Mais intuitivo
- Suporta GPU com CuPy

---

## 📚 RECURSOS RECOMENDADOS

### Livros Fundamentais (Leitura Essencial)

1. **"Pattern Recognition and Machine Learning"** - Christopher M. Bishop (2006)
   - Fundação teórica completa
   - Métodos Bayesianos em profundidade
   - Graphical models
   - Avançado, mas de ouro

2. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman (2009)
   - Perspectiva estatística rigorosa
   - Cobertura de regressão, classificação, árvores
   - Matemática exigente
   - Referência padrão na indústria

3. **"Hands-On Machine Learning"** - Aurélien Géron (2019)
   - Prático desde o início
   - ML tradicional + Deep Learning
   - Código em Python/TensorFlow
   - Excelente para aprender fazendo

4. **"A Few Useful Things to Know About Machine Learning"** - Pedro Domingos (2012)
   - Paper leitura rápida (12 páginas)
   - Insights sobre bias-variance, feature engineering
   - Fundamental para intuição

### Cursos Online

- **Fast.ai**: Prático, "top-down", excelente Deep Learning
- **Coursera - Machine Learning** (Andrew Ng): Clássico
- **Udacity Nanodegrees**: Especializados e certificados
- **DataCamp**: Interativo, muita prática hands-on

### Bibliotecas Python Essenciais

|     Biblioteca   |            Função          |           Instalação          |
|------------------|----------------------------|-------------------------------|
| **NumPy**        | Computação numérica arrays | `pip install numpy`           |
| **Pandas**       | Manipulação de dados       | `pip install pandas`          |
| **Scikit-learn** | ML tradicional             | `pip install scikit-learn`    |
| **Matplotlib**   | Visualização estática      | `pip install matplotlib`      |
| **Seaborn**      | Visualização estatística   | `pip install seaborn`         |
| **Plotly**       | Visualização interativa    | `pip install plotly`          |
| **TensorFlow**   | Deep Learning              | `pip install tensorflow`      |
| **PyTorch**      | Deep Learning (alternativa)| `pip install torch`           |
| **XGBoost**      | Gradient Boosting          | `pip install xgboost`         |
| **LightGBM**     | Gradient Boosting rápido   | `pip install lightgbm`        |
| **Optuna**       | Hyperparameter tuning      | `pip install optuna`          |
| **SHAP**         | Explicabilidade            | `pip install shap`            |
| **LIME**         | Explicabilidade            | `pip install lime`            |
| **MLflow**       | Experiment tracking        | `pip install mlflow`          |
| **FastAPI**      | APIs                       | `pip install fastapi uvicorn` |
| **Streamlit**    | Web apps rápidas           | `pip install streamlit`       |
| **Dask**         | Computação distribuída     | `pip install dask[complete]`  |

### Datasets Públicos para Praticar

- **Kaggle**: https://kaggle.com/datasets (milhares de datasets)
- **UCI ML Repository**: https://archive.ics.uci.edu (datasets clássicos)
- **Google Dataset Search**: https://datasetsearch.research.google.com
- **Awesome Public Datasets**: https://github.com/awesomedata/awesome-public-datasets

### Papers Seminais

1. "Random Forests" - Breiman (2001)
2. "XGBoost: A Scalable Tree Boosting System" - Chen & Guestrin (2016)
3. "A Unified Approach to Interpreting Model Predictions" - Lundberg & Lee (2017) [SHAP]
4. "'Why Should I Trust You?': Explaining the Predictions of Any Classifier" - Ribeiro et al. (2016) [LIME]
5. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" - Hinton et al. (2012)
6. "A Few Useful Things to Know About Machine Learning" - Domingos (2012)

---

## 🎓 CONCLUSÃO E PRÓXIMAS PASSOS

### Timeline de Desenvolvimento

**Meses 1-2: Fundamentos**
- Python data science stack (NumPy, Pandas, Matplotlib)
- Conceitos: tipos de aprendizado, métricas, validação
- Projetos: Iris classification, Boston housing regression

**Meses 3-4: Algoritmos Core**
- Regressão linear/logística
- Árvores de decisão, Random Forest, SVM
- Feature engineering prático
- Projetos: Titanic, Credit card fraud

**Meses 5-6: Avançado**
- XGBoost, LightGBM
- Hyperparameter tuning
- Cross-validation estratégico
- Projetos: Competições Kaggle

**Meses 7-12: Especialização**
- Clustering e redução de dimensionalidade
- Interpretabilidade (SHAP, LIME)
- Deploy (FastAPI, Streamlit)
- MLOps basics
- Projetos próprios em produção

**Ano 2+: Expertise**
- Transfer Learning e Deep Learning
- AutoML e meta-learning
- ML em escala (Spark, Dask)
- Pesquisa em tópicos específicos

### Estratégia de Aprendizado Recomendada

1. **Escolher Projeto Real**: Use dados públicos (Kaggle)
2. **Implementar do Zero**: Sem notebooks prontos
3. **Experimentar 5+ Modelos**: Nunca confiar em um único
4. **Interpretar Resultados**: Use SHAP/LIME
5. **Deploy e Feedback**: Colocar em produção
6. **Iterar Continuamente**: Machine Learning é iterativo

### Desafios Comuns e Como Evitar

|         Erro        |             Causa         |                Solução              |
|---------------------|---------------------------|-------------------------------------|
| Overfitting         | Modelo muito complexo     | Regularização, CV, early stopping   |
| Data leakage        | Features de test no train | Split antes de preprocessar         |
| Desbalanceamento    | Classes desiguais         | SMOTE, pesos, métricas apropriadas  |
| Poor generalization | Dados train ≠ test        | Validação robusta, monitoramento    |
| Falta de baseline   | Sem comparação            | Sempre implementar baseline simples |

### Ressources Finais

- **Comunidade**: Kaggle forums, Reddit r/MachineLearning
- **Blogs**: Towards Data Science, Analytics Vidhya
- **Conferências**: NeurIPS, ICML, ICLR
- **GitHub**: Explore implementações open-source

---

**Última atualização**: Novembro 2025

**Próximas revisões**: Novos papers, bibliotecas, tendências de mercado

Este guia é living document e deve evoluir conforme o campo avança.
