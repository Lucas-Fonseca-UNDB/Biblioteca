# 📋 Deep Learning: Resumo Executivo & Quick Reference Guide
## Referência Rápida para Implementação Prática

---

## 🚀 QUICK START - TEMPLATE UNIVERSAL

### PyTorch Pipeline (3 minutos)
```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

# 1. Dados
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Modelo
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
).to('cuda')

# 3. Treinar
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for X, y in train_loader:
        X, y = X.to('cuda'), y.to('cuda')
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### TensorFlow Pipeline (3 minutos)
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Dados - já em TensorFlow

# 2. Modelo
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compilar e Treinar
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

---

## 📊 DECISÃO: QUAL ARQUITETURA USAR?

```
Seu Problema?
    │
    ├─ Imagens → CNN (ResNet/EfficientNet)
    ├─ Texto → Transformer (BERT/GPT)
    ├─ Séries Temporais → LSTM/GRU/Transformer
    ├─ Geração → GAN/Diffusion/VAE
    ├─ Dados Tabulares → MLP/XGBoost
    └─ Multimodal → CLIP/LLaVA
```

---

## 🔧 HIPERPARÂMETROS RECOMENDADOS (2025)

| Parâmetro | Recomendação | Intervalo |
|-----------|-------------|-----------|
| **Learning Rate** | 0.001 (Adam) | 1e-5 a 0.1 |
| **Batch Size** | 32 | 8-256 |
| **Epochs** | 50-100 | 10-1000 |
| **Dropout** | 0.2-0.5 | 0.1-0.7 |
| **Optimizer** | **Adam** | SGD, AdamW, RMSprop |
| **Weight Init** | He (ReLU) | Xavier (Tanh) |
| **Activation** | **ReLU** | GELU, Swish, Leaky-ReLU |
| **Normalization** | Batch Norm (CNN), Layer Norm (Transformer) | - |
| **Early Stop Patience** | 5-10 épocas | 3-20 |

---

## 💾 DADOS & PREPROCESSING

### Normalização
```python
# Standard (z-score)
data = (data - data.mean()) / data.std()

# Min-Max [0,1]
data = (data - data.min()) / (data.max() - data.min())

# ImageNet padrão
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
```

### Data Augmentation (Imagens)
```python
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
```

### Train/Val/Test Split
```python
# 70/15/15 típico
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train, val, test = random_split(dataset, [train_size, val_size, test_size])
```

---

## 🧠 CAMADAS COMUNS

| Camada | Função | Input/Output | Uso |
|--------|--------|-------------|-----|
| **Dense/Linear** | Multiplica matriz pesos | (batch, n_in) → (batch, n_out) | MLP, output |
| **Conv2d** | Convolução 2D | (batch, C, H, W) → (batch, Filters, H', W') | CNNs |
| **MaxPool2d** | Max pooling | (batch, C, H, W) → (batch, C, H/2, W/2) | Reduz dimensão |
| **Dropout** | Desativa neurônios | x → x (stochastic) | Regularização |
| **BatchNorm** | Normaliza por batch | x → (x-μ)/σ | Acelera treino |
| **LayerNorm** | Normaliza por features | x → (x-μ)/σ | Transformers |
| **LSTM** | Memória longa | (batch, seq, features) → (batch, hidden) | Sequências |
| **Embedding** | Mapeia índices a vetores | (batch, seq) → (batch, seq, emb_dim) | NLP |
| **Attention** | Focagem ponderada | (batch, seq, dim) → (batch, seq, dim) | Transformers |

---

## 🎯 FUNÇÕES DE ATIVAÇÃO RÁPIDA

```python
import torch.nn.functional as F

# ReLU (padrão)
x = F.relu(x)

# Sigmoid (output 0-1, classificação binária)
x = torch.sigmoid(x)

# Softmax (distribuição probabilidade, multiclass)
x = F.softmax(x, dim=1)

# Tanh (output -1 a 1)
x = torch.tanh(x)

# GELU (Transformers modernos)
x = F.gelu(x)

# Leaky ReLU (evita dying ReLU)
x = F.leaky_relu(x, negative_slope=0.01)
```

---

## 📉 LOSS FUNCTIONS

| Tarefa | Loss | Código PyTorch |
|--------|------|----------|
| **Classificação Binária** | Binary Cross-Entropy | `nn.BCELoss()` |
| **Classificação Multiclass** | Cross-Entropy | `nn.CrossEntropyLoss()` |
| **Regressão** | MSE/MAE | `nn.MSELoss()` ou `nn.L1Loss()` |
| **Similaridade** | Contrastive/Triplet | `nn.TripletMarginLoss()` |
| **Ranking** | Margin Ranking | `nn.MarginRankingLoss()` |

---

## ⚡ OTIMIZADORES

```python
# Adam (MELHOR para 90% dos casos)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# SGD com Momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# RMSprop
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# AdamW (weight decay desacoplado - melhor regularização)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

## 📍 LEARNING RATE SCHEDULING

```python
# Step Decay (reduz a cada N épocas)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Exponential Decay
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine Annealing (recomendado)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau (reduz se loss para de melhorar)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# No loop de treino
for epoch in range(100):
    train()
    scheduler.step()  # ou scheduler.step(val_loss) para ReduceLROnPlateau
```

---

## 🛑 REGULARIZAÇÃO (EVITA OVERFITTING)

```python
# 1. L1/L2 Regularization (Weight Decay)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# 2. Dropout
nn.Dropout(p=0.5)  # 50% dos neurônios desativados

# 3. Batch Normalization
nn.BatchNorm2d(num_features)

# 4. Early Stopping
if val_loss < best_loss:
    best_loss = val_loss
    torch.save(model.state_dict(), 'best.pth')
else:
    patience -= 1
    if patience == 0: break

# 5. Data Augmentation
# Ver seção acima

# 6. Layer Normalization (Transformers)
nn.LayerNorm(features)
```

---

## 📊 MÉTRICAS ESSENCIAIS (Classificação)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Calcula automaticamente
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
auc_roc = roc_auc_score(y_true, y_probs[:, 1])

# Quando usar qual
# Dados balanceados → Accuracy
# Recall importante (não perder positivos) → Recall
# Precision importante (minimizar falsos positivos) → Precision
# Balanceado → F1-Score
# Variando threshold → AUC-ROC ou PR-AUC
```

---

## 🐛 DEBUGGING & TROUBLESHOOTING

| Sintoma | Causa Provável | Solução |
|---------|------------|---------|
| **Loss NaN/Inf** | Gradiente explodindo | Clip gradients, reduz LR, verifica dados |
| **Loss não muda** | LR muito baixo | Aumenta learning rate |
| **Diverge (loss ↑)** | LR muito alto | Reduz learning rate |
| **Train ↑, Val ↓** | Overfitting | Dropout, L2, Early stop, mais dados |
| **Train ↓, Val ↓** | Underfitting | Modelo maior, mais épocas, features |
| **GPU Memory Exceeded** | Batch muito grande | Reduz batch_size ou usa gradient accumulation |
| **Treino muito lento** | GPU não sendo usada | Verifica device, aumenta batch_size |
| **Reprodutibilidade** | Seeds não fixadas | `torch.manual_seed(42)` |

---

## ✅ CHECKLIST PRÉ-PRODUÇÃO

- [ ] Dados preprocessados e normalizados
- [ ] Train/Val/Test splits (70/15/15)
- [ ] Data augmentation apropriada
- [ ] Modelo testado em small batch (overfitting esperado)
- [ ] Learning rate schedule definido
- [ ] Early stopping implementado
- [ ] Métricas corretas para tarefa
- [ ] Baseline comparado (chance level, modelo simples)
- [ ] Validação cruzada considerada
- [ ] Hiperparâmetros documentados
- [ ] Melhor modelo salvo
- [ ] Análise de erros feita
- [ ] Modelo quantizado/otimizado para deployment
- [ ] Testes em dados nunca vistos
- [ ] Documentação completa

---

## 🎯 COMPARAÇÃO FRAMEWORKS (2025)

| Framework | Pesquisa | Prototipagem | Produção | Velocidade | Comunidade |
|-----------|----------|-------------|----------|-----------|-----------|
| **PyTorch** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Excelente |
| **TensorFlow** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Forte |
| **JAX** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Crescente |

**Recomendação**: PyTorch para aprender, TensorFlow para produção.

---

## 📚 RECURSOS RECOMENDADOS

### Papers Fundamentais
- AlexNet (2012): ImageNet Classification
- ResNet (2015): Deep Residual Learning
- Transformer (2017): Attention Is All You Need
- BERT (2018): Bidirectional Encoder Representations
- Vision Transformer (2021): Image is Worth 16x16 Words

### Datasets Populares
- **MNIST**: Dígitos manuscritos (entrada fácil)
- **CIFAR-10/100**: Imagens 32×32 (visão básica)
- **ImageNet**: 1.2M imagens, 1000 classes (visão avançada)
- **COCO**: Detecção, segmentação (multimodal)
- **SQuAD**: Question Answering (NLP)

### Bibliotecas Essenciais
- **PyTorch** / **TensorFlow**: Frameworks principais
- **Hugging Face Transformers**: Modelos pré-treinados
- **scikit-learn**: Métricas, preprocessing
- **OpenCV**: Processamento imagens
- **NumPy / Pandas**: Manipulação dados
- **Matplotlib / Seaborn**: Visualização

### Cursos Online
- Fast.ai: Prático, top-down
- Stanford CS231N: CNNs (visão)
- Stanford CS224N: NLP
- MIT 6.S191: Introdução IA

---

## 🚀 PRÓXIMOS PASSOS

1. **Semana 1**: Setup, MLP simples, MNIST
2. **Semana 2**: CNNs, CIFAR-10, transfer learning
3. **Semana 3**: RNNs/LSTMs, séries temporais
4. **Semana 4**: Transformers, BERT fine-tuning
5. **Semana 5**: Modelos generativos (VAE/GAN)
6. **Semana 6**: Projeto completo end-to-end

---

**Última atualização**: Novembro 2025  
**Framework principal**: PyTorch 2.0+  
**TensorFlow**: 2.13+
