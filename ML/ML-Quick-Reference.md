# Machine Learning: Resumo Executivo & Quick Reference
## Referência Rápida para Algoritmos e Implementação

---

## Índice de Referência Rápida

1. [Selecionador de Algoritmo](#selecionador-de-algoritmo)
2. [Tabelas Comparativas](#tabelas-comparativas)
3. [Fórmulas Essenciais](#fórmulas-essenciais)
4. [Checklist do Pipeline](#checklist-do-pipeline)
5. [Quando Usar Cada Método](#quando-usar-cada-método)
6. [Troubleshooting Comum](#troubleshooting-comum)

---

## Selecionador de Algoritmo

### Você tem labels para os dados?

#### SIM → Aprendizado Supervisionado

**Predir número (valor contínuo)?**

- **SIM** → **REGRESSÃO**
  - Simples: Linear Regression
  - Complexo: Random Forest, Gradient Boosting (XGBoost)
  - Não-linear: Polynomial Regression, SVM (kernel RBF)

- **NÃO** → **CLASSIFICAÇÃO**
  
  **Dados estruturados (tabelas)?**
  
  - **SIM**:
    - Simples/Interpretável: Logistic Regression, Decision Trees
    - Robusto: Random Forest, LightGBM
    - Extremo: XGBoost, Gradient Boosting
    - Sem tuning: SVM
  
  - **NÃO** (imagens/texto/audio):
    - Imagens: CNN (ResNet, EfficientNet)
    - Texto: RNN/LSTM, Transformers (BERT, GPT)
    - Audio: CNN 1D, MelSpectrograms + CNN

#### NÃO → Aprendizado Não-Supervisionado

**Encontrar grupos nos dados?**

- **SIM** → **CLUSTERING**
  - Spherical clusters: K-Means
  - Arbitrary shapes: DBSCAN, Hierarchical
  - Densidade variável: DBSCAN

- **NÃO** → **REDUÇÃO DE DIMENSIONALIDADE**
  - Linear: PCA
  - Não-linear: t-SNE, UMAP
  - Features: Autoencoders

---

## Tabelas Comparativas

### Classificação: Algoritmos Supervisionados

| Algoritmo | Interpretabilidade | Velocidade | Dados Faltantes | Escalabilidade | Tuning | Melhor Para |
|-----------|-------------------|-----------|-----------------|----------------|--------|------------|
| Logistic Regression | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Não | ⭐⭐⭐⭐⭐ | Fácil | Linear, baseline |
| Decision Tree | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✓ | ⭐⭐⭐ | Médio | Rápido, interpretável |
| Random Forest | ⭐⭐⭐ | ⭐⭐⭐ | ✓ | ⭐⭐⭐⭐ | Médio | Robusto, geral |
| XGBoost | ⭐⭐ | ⭐⭐ | ✓ | ⭐⭐⭐ | Complexo | Competições, máxima performance |
| LightGBM | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✓ | ⭐⭐⭐⭐⭐ | Complexo | Big Data, velocidade |
| SVM | ⭐ | ⭐⭐ | Não | ⭐⭐⭐ | Difícil | Média dimensionalidade |
| KNN | ⭐⭐⭐⭐⭐ | ⭐ | ✓ | ⭐ | Fácil | Sem tuning, baseline |
| Naive Bayes | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✓ | ⭐⭐⭐⭐⭐ | Muito Fácil | NLP, streaming |
| Neural Network | ⭐ | ⭐⭐ | ✓ | ⭐⭐⭐⭐ | Difícil | Dados complexos, imagens |

### Regressão: Comparação

| Algoritmo | Linear | Interpretável | Rápido | Outliers | Escalável | Melhor Para |
|-----------|--------|--------------|--------|----------|-----------|------------|
| Linear Regression | ✓ | ✓✓✓ | ✓ | ✗ (sensível) | ✓ | Baseline, inferência |
| Ridge/Lasso | ✓ | ✓✓✓ | ✓ | ✗ | ✓ | Multicolinearidade |
| Polynomial | Não | ✓✓ | ✓ | ✗ | ✓ | Relação não-linear |
| Decision Tree | ✗ | ✓✓✓ | ✓ | ✓ | ✓ | Interpretabilidade |
| Random Forest | ✗ | ✓ | ✓ | ✓ | ✓✓ | Robusto, geral |
| Gradient Boosting | ✗ | ✗ | ✗ | ✓ | ✓✓ | Máxima accuracy |
| SVM | ✗ | ✗ | ✗ | ✓✓ | ✓ | Kernel, não-linear |
| Neural Network | ✗ | ✗ | ✗ | ✓ | ✓✓ | Estruturas complexas |

### Clustering: Comparação

| Algoritmo | Clusters | Escalabilidade | Forma | Outliers | Hiperparâmetros | Tuning |
|-----------|----------|----------------|-------|----------|-----------------|--------|
| K-Means | Esféricos | ⭐⭐⭐⭐⭐ | Esfera | ✗ | K (número clusters) | Fácil |
| DBSCAN | Arbitrária | ⭐⭐⭐ | Qualquer | ✓ | ε, MinPts | Difícil |
| Hierarchical | Arbitrária | ⭐⭐ | Qualquer | ✗ | Linkage criterion | Fácil |
| Gaussian Mixture | Esféricos | ⭐⭐⭐⭐ | Elipse | ✗ | K, covariance | Médio |
| Spectral | Arbitrária | ⭐⭐ | Qualquer | ✗ | K, kernel | Difícil |

---

## Fórmulas Essenciais

### Otimização

**Gradient Descent Básico**:
\[w := w - \eta \nabla L(w)\]

**Momentum**:
\[v := \beta v - \eta \nabla L(w) \quad ; \quad w := w + v\]

**Adam**:
\[m := \beta_1 m + (1-\beta_1)\nabla L \quad ; \quad v := \beta_2 v + (1-\beta_2)(\nabla L)^2\]
\[w := w - \eta \frac{m}{\sqrt{v} + \epsilon}\]

### Regularização

**L2 (Ridge)**:
\[L_{total} = L + \frac{\lambda}{2n}||w||_2^2\]

**L1 (Lasso)**:
\[L_{total} = L + \frac{\lambda}{n}||w||_1\]

### Regressão Linear

**Equação Normal**:
\[w = (X^T X)^{-1}X^T y\]

**MSE**:
\[MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2\]

**R²**:
\[R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}\]

### Regressão Logística

**Sigmoid**:
\[\sigma(z) = \frac{1}{1 + e^{-z}}\]

**Cross-Entropy Loss**:
\[L = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]\]

### Métricas de Classificação

**Accuracy**:
\[\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}\]

**Precision**:
\[\text{Prec} = \frac{TP}{TP + FP}\]

**Recall (Sensitivity)**:
\[\text{Rec} = \frac{TP}{TP + FN}\]

**F1-Score**:
\[\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\]

**ROC-AUC**: Área sob curva (FPR vs TPR)

### Métricas de Regressão

**MAE** (Robusto):
\[\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|\]

**RMSE** (Diferenças grandes penalizadas):
\[\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}\]

**MAPE** (Interpretável em %):
\[\text{MAPE} = \frac{100}{n}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right|\]

### Clustering

**K-Means Intra-cluster**:
\[J = \sum_{k=1}^{K}\sum_{x \in C_k}||x - \mu_k||^2\]

**Silhouette Score**:
\[s = \frac{b - a}{\max(a,b)}\]

Onde a = distância intra-cluster, b = distância para cluster mais próximo

---

## Checklist do Pipeline

### ✅ Pré-processamento

- [ ] Dados carregados e explorados
- [ ] Valores faltantes tratados (imputação ou remoção)
- [ ] Outliers identificados e tratados
- [ ] Variáveis categóricas encodadas (one-hot, label encoding)
- [ ] Features escaladas/normalizadas (StandardScaler, MinMaxScaler)
- [ ] Features com muito missing (>50%) removidas
- [ ] Features altamente correlacionadas removidas
- [ ] Dados divididos em treino/validação/teste (ex: 70/15/15)

### ✅ Feature Engineering

- [ ] Novas features criadas se necessário
- [ ] Interações entre features exploradas
- [ ] Transformações polinomiais aplicadas (se apropriado)
- [ ] Seleção de features realizada (RFE, SelectKBest)
- [ ] Class imbalance tratado (oversampling, undersampling, SMOTE)

### ✅ Modelagem

- [ ] Baseline estabelecido (modelo simples)
- [ ] Múltiplos algoritmos testados
- [ ] Validação cruzada implementada
- [ ] Hiperparâmetros tuned (GridSearch, RandomSearch, Optuna)
- [ ] Overfitting verificado (treino vs validação)
- [ ] Ensemble methods testados (bagging, boosting)

### ✅ Avaliação

- [ ] Métricas apropriadas escolhidas
- [ ] Matriz de confusão analisada
- [ ] Falsos positivos vs falsos negativos entendidos
- [ ] ROC-AUC calculado
- [ ] Importância de features analisada
- [ ] Performance em dados de teste reportada

### ✅ Produção

- [ ] Modelo versionado (MLflow, DVC)
- [ ] Pipeline reprodutível (seeds, versões de libs)
- [ ] Intervalo de confiança/incerteza estimado
- [ ] Modelo serializado (pickle, ONNX, SavedModel)
- [ ] API criada se necessário (FastAPI, Flask)
- [ ] Monitoramento configurado (drift detection, performance)

---

## Quando Usar Cada Método

### Você deve usar Logistic Regression Se:
- ✓ Dados linearmente separáveis (ou aproximadamente)
- ✓ Interpretabilidade é crítica
- ✓ Muitas amostras (converge rápido)
- ✓ Baseline necessário
- ✗ Relação altamente não-linear

### Você deve usar Decision Trees Se:
- ✓ Precisa interpretabilidade visual
- ✓ Dados com valores faltantes
- ✓ Features mistas (numéricas + categóricas)
- ✓ Interações complexas
- ✗ Dados muito ruidosos (overfitting)
- ✗ Estrutura muito profunda (lento)

### Você deve usar Random Forest Se:
- ✓ Boa performance sem muito tuning
- ✓ Feature importance necessário
- ✓ Features mistas
- ✓ Parallelizável
- ✗ Precisa model muito interpretável
- ✗ Deploy com latência baixa crítica

### Você deve usar XGBoost/LightGBM Se:
- ✓ Máxima performance é objetivo
- ✓ Dados estruturados/tabular
- ✓ Disposição para tuning complexo
- ✓ Tempos de treino relevantes (LightGBM)
- ✗ Interpretabilidade crítica
- ✗ Dados com muito ruído

### Você deve usar SVM Se:
- ✓ Dados linearmente separáveis em espaço transformado
- ✓ Dimensionalidade moderada
- ✓ Pequeno dataset (< 100k amostras)
- ✓ Kernel específico reduz dimensionalidade
- ✗ Muitas amostras (escalabilidade)
- ✗ Muito tuning necessário

### Você deve usar KNN Se:
- ✓ Baseline muito rápido necessário
- ✓ Sem tempo para tuning
- ✓ Dados com padrões locais
- ✗ Alto custo computacional aceitável
- ✗ Muitos features (curse of dimensionality)

### Você deve usar Redes Neurais Se:
- ✓ Dados não-estruturados (imagens, texto, audio)
- ✓ Muito dados disponível (> 100k)
- ✓ Recursos computacionais (GPU)
- ✓ Tempo para tuning
- ✗ Precisa interpretabilidade
- ✗ Muito poucos dados

### Você deve usar CNN Se:
- ✓ Dados são imagens
- ✓ Estrutura espacial importante
- ✓ Muitos dados (transferência aprendizado se poucos)
- ✓ GPU disponível

### Você deve usar LSTM/RNN Se:
- ✓ Dados sequenciais (séries temporais, texto)
- ✓ Contexto histórico importante
- ✓ Sequências de comprimento variável
- ✓ GPU para treinamento

### Você deve usar Transformers Se:
- ✓ Texto (BERT, GPT, T5)
- ✓ Sequências muito longas
- ✓ Transfer learning crítico
- ✓ Recursos (GPU/TPU) disponível

### Você deve usar K-Means Se:
- ✓ Clusters aproximadamente esféricos
- ✓ Número de clusters conhecido
- ✓ Escalabilidade é prioridade
- ✗ Clusters densidade variável
- ✗ Muitos outliers

### Você deve usar DBSCAN Se:
- ✓ Clusters forma arbitrária
- ✓ Deseja descobrir número de clusters
- ✓ Outliers devem ser ignorados
- ✗ Parâmetros sensíveis (ε, MinPts)
- ✗ Escalabilidade crítica

### Você deve usar PCA Se:
- ✓ Reduzir dimensionalidade lineares
- ✓ Visualização com redução
- ✓ Features correlacionadas
- ✗ Relações não-lineares importantes
- ✗ Interpretabilidade das componentes

### Você deve usar t-SNE Se:
- ✓ Visualização 2D/3D
- ✓ Relações não-lineares
- ✓ Qualidade visual é prioridade
- ✗ Reproducibilidade exata
- ✗ Grandes datasets (lento)

---

## Troubleshooting Comum

### Problema: Overfitting (train_acc >> val_acc)

**Causas**:
- Modelo muito complexo para dados
- Regularização insuficiente
- Dados de treino muito pequeno

**Soluções** (em ordem):
1. Aumentar L1/L2 regularization (λ)
2. Aumentar Dropout rate
3. Early stopping
4. Reduzir complexidade (menos camadas, max_depth)
5. Mais dados de treino
6. Data augmentation

### Problema: Underfitting (train_acc ≈ val_acc, ambos baixos)

**Causas**:
- Modelo muito simples
- Features inadequadas
- Não convergiu ainda

**Soluções**:
1. Aumentar complexidade (mais camadas, max_depth)
2. Feature engineering (mais features, transformações)
3. Reduzir regularização
4. Treinar mais epochs
5. Aumentar learning rate (teste incrementalmente)

### Problema: Modelo muito lento para treinar

**Causas**:
- Dataset muito grande
- Modelo muito complexo
- Learning rate muito pequeno

**Soluções**:
1. Mini-batch training
2. Reduzir número de features
3. Aumentar learning rate
4. Usar algoritmo mais rápido (LightGBM vs XGBoost)
5. GPU acceleration

### Problema: Modelo muito lento em predição (latência)

**Causas**:
- Modelo muito grande
- Features computacionalmente caras
- Sem caching

**Soluções**:
1. Quantização (float32 → float16 ou int8)
2. Model pruning (remover neurônios/features)
3. Reduzir número de features
4. Batch prediction
5. Caching de predições comuns

### Problema: Desempenho ruim em dados novos (drift)

**Causas**:
- Data drift (distribuição das features mudou)
- Concept drift (relação X→y mudou)
- Modelo não captura padrões

**Soluções**:
1. Monitorar drift
2. Retraining automático periódico
3. Feature importance → revisar features
4. Mais dados de treino
5. Modelo mais complexo/adequado

### Problema: Classe desbalanceada (ex: 99% negativos)

**Sintomas**:
- Acurácia alta mas precision/recall baixa
- Modelo prediz sempre classe majoritária

**Soluções**:
1. **Métrica**: Usar F1, precision-recall, ROC-AUC (não accuracy)
2. **Resampling**: 
   - Oversampling: SMOTE, random oversampling
   - Undersampling: random undersampling
3. **Pesos**: `class_weight='balanced'` em sklearn
4. **Threshold**: Ajustar threshold de decisão
5. **Ensemble**: XGBoost com `scale_pos_weight`

```python
# Exemplo com class_weight
model = RandomForestClassifier(class_weight='balanced')

# Ou customizado
class_weight = {0: 1, 1: 10}  # Penalizar menos classe minoritária
model = LogisticRegression(class_weight=class_weight)
```

### Problema: Valores faltantes demais (>50%)

**Decisão**:
- Se aleatório: Remover coluna
- Se padrão: Feature importante → Manter + imputar

**Métodos de Imputação**:
1. Média/Mediana (simples, pode perder info)
2. KNN Imputer (usa vizinhança)
3. Multiple Imputation (cria múltiplas versões)
4. Model-based (treina modelo para imputar)
5. Forward fill (séries temporais)

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simples
imputer = SimpleImputer(strategy='mean')

# KNN
imputer = KNNImputer(n_neighbors=5)

X_imputed = imputer.fit_transform(X)
```

### Problema: Features em escalas muito diferentes

**Sintoma**: Modelo não converge bem, especialmente com regularização

**Soluções**:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# [0,1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Log transform para features altamente skewed
X['feature'] = np.log1p(X['feature'])
```

---

## Dicas Pro

1. **Sempre versione seus modelos** - Use MLflow ou DVC
2. **Valide em dados realmente novos** - Não reuse conjunto teste
3. **Baselines são importantes** - Simples modelo primeiro
4. **Métricas importam mais que acurácia** - Escolha métrica do negócio
5. **Explainabilidade > accuracy** - Se modelo deve ser confiável
6. **Dados > algoritmo** - Qualidade dados > ajustes finos
7. **Reproduzibilidade é essencial** - Sempre seed determinístico
8. **Monitor em produção** - Drift happens, retrain quando necessário

