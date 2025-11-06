# 🔧 Deep Learning: Exemplos de Código Prontos
## Snippets Prontos para Copy-Paste e Adaptação (PyTorch & TensorFlow)

---

# PARTE 1: CONFIGURAÇÃO E IMPORTS

## 1.1 PyTorch Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageNet, MNIST
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Verificar GPU
print(f"GPU disponível: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 1.2 TensorFlow/Keras Setup

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Verificar GPU
print(f"GPU disponível: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Version: {tf.__version__}")

# Habilitar memory growth (não aloca toda GPU de uma vez)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

# PARTE 2: REDES FEEDFORWARD (MLP)

## 2.1 PyTorch - MLP Simples

```python
# Definir arquitetura
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Instanciar modelo
model = SimpleMLP().to(device)

# Loss e optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinar loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Validar
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Pipeline completo
num_epochs = 20
best_val_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("  ✓ Modelo salvo!")
```

## 2.2 TensorFlow/Keras - MLP

```python
# Definir modelo
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
)

# Avaliar
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

# PARTE 3: REDES CONVOLUCIONAIS (CNN)

## 3.1 PyTorch - CNN Customizada

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # Bloco 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloco 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloco 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Assume input 32x32
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Bloco 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)  # 32 → 16
        
        # Bloco 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # 16 → 8
        
        # Bloco 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # 8 → 4
        
        # FC
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CustomCNN(num_classes=10).to(device)
```

## 3.2 PyTorch - ResNet Transfer Learning

```python
import torchvision.models as models

# Carregar ResNet pré-treinado
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Congelar todos os parâmetros
for param in model.parameters():
    param.requires_grad = False

# Substituir última camada FC para nosso dataset
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 classes para CIFAR-10

model = model.to(device)

# Apenas última camada será treinada (feature extraction)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Depois de convergência, descongelar e fine-tune:
# for param in model.parameters():
#     param.requires_grad = True
# optimizer = optim.Adam(model.parameters(), lr=0.0001)  # LR menor
```

## 3.3 TensorFlow/Keras - Transfer Learning

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Carregar ResNet50 pré-treinado
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar base
base_model.trainable = False

# Construir modelo
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar (apenas FC layers)
model.fit(train_data, epochs=10)

# Fine-tuning: descongelar últimas camadas
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Congela primeiras 50
    layer.trainable = False

# Recompilar com LR menor
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=10)
```

---

# PARTE 4: REDES RECORRENTES (LSTM/GRU)

## 4.1 PyTorch - LSTM para Séries Temporais

```python
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Usar último hidden state
        out = self.fc(lstm_out[:, -1, :])  # (batch, hidden_size) → (batch, output)
        return out

# Exemplo: Previsão de estoque
model = LSTMNet(input_size=5, hidden_size=64, num_layers=2, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinar (similar ao MLP)
for epoch in range(50):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 4.2 PyTorch - Bidirectional LSTM para NLP

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, output_size):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        # Bidirectional → saída é 2*hidden_size
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, seq_len) com índices de palavras
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, 2*hidden)
        # Usar último hidden state (forward + backward concatenado)
        last_hidden = lstm_out[:, -1, :]  # (batch, 2*hidden)
        out = self.fc(self.dropout(last_hidden))  # (batch, output)
        return out

# Uso
model = BiLSTMClassifier(vocab_size=10000, embedding_dim=100, 
                        hidden_size=128, num_layers=2, output_size=2).to(device)
```

## 4.3 TensorFlow/Keras - LSTM

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=100),
    layers.LSTM(64, return_sequences=True, dropout=0.2),
    layers.LSTM(32, dropout=0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

---

# PARTE 5: TRANSFORMERS

## 5.1 PyTorch - Atenção Simples (Scaled Dot-Product)

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Linear transformations and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        return output, attn_weights
```

## 5.2 Usar BERT Pré-treinado com PyTorch

```python
from transformers import AutoTokenizer, AutoModel

# Carregar tokenizer e modelo BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Exemplo: embeddings de frase
text = "Deep learning is amazing!"
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)
    cls_token = last_hidden_state[:, 0, :]  # (1, 768) - classificação

# Fine-tune para classificação
class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(cls_token)
        return logits

model = BertClassifier("bert-base-uncased", num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
```

## 5.3 TensorFlow/Keras com Transformers

```python
from transformers import TFAutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = TFAutoModel.from_pretrained(model_name)

# Construir modelo com BERT
def build_model():
    input_ids = keras.Input(shape=(128,), dtype=tf.int32)
    attention_mask = keras.Input(shape=(128,), dtype=tf.int32)
    
    outputs = bert_model(input_ids, attention_mask=attention_mask)
    cls_token = outputs[0][:, 0, :]  # CLS token
    
    x = layers.Dense(256, activation='relu')(cls_token)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs=[input_ids, attention_mask], outputs=predictions)
    return model

model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

# PARTE 6: AUTOENCODERS E REDES GENERATIVAS

## 6.1 PyTorch - Autoencoder

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output [0, 1]
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Usar para anomalia detection
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinar
for epoch in range(50):
    total_loss = 0
    for X_batch, _ in train_loader:
        X_batch = X_batch.view(-1, 784).to(device)
        
        reconstruction, latent = model(X_batch)
        loss = criterion(reconstruction, X_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# Detectar anomalias: reconstrução ruim = anomalia
reconstruction_errors = []
for X_batch, _ in test_loader:
    X_batch = X_batch.view(-1, 784).to(device)
    with torch.no_grad():
        reconstruction, _ = model(X_batch)
        error = torch.mean((X_batch - reconstruction)**2, dim=1)
        reconstruction_errors.append(error.cpu().numpy())

reconstruction_errors = np.concatenate(reconstruction_errors)
threshold = np.percentile(reconstruction_errors, 95)  # Top 5% como anomalias
anomalies = reconstruction_errors > threshold
```

## 6.2 PyTorch - Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)
    
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss VAE
def vae_loss(reconstruction, x, mu, logvar):
    MSE = nn.functional.mse_loss(reconstruction, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        reconstruction, mu, logvar = model(X_batch)
        loss = vae_loss(reconstruction, X_batch, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

# PARTE 7: MÉTRICAS E AVALIAÇÃO

## 7.1 PyTorch - Métricas de Classificação

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt

def evaluate_classifier(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusão:")
    print(cm)
    
    return accuracy, precision, recall, f1

evaluate_classifier(model, test_loader, device)
```

## 7.2 TensorFlow/Keras - Métricas

```python
# Callback para salvar melhor modelo
best_model_callback = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Callback para parar cedo
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Treinar
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[best_model_callback, early_stopping]
)

# Avaliar
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Plotar histórico
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()
```

---

# PARTE 8: PIPELINES COMPLETOS

## 8.1 Pipeline Completo - Classificação CIFAR-10 com PyTorch

```python
# 1. DADOS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 2. MODELO
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# 3. TREINAMENTO
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_val_acc = 0
patience = 5
patience_counter = 0

for epoch in range(50):
    # Train
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_acc += (preds == labels).sum().item()
    
    val_acc /= len(test_dataset)
    scheduler.step()
    
    print(f"Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

# 4. TESTE FINAL
model.load_state_dict(torch.load('best_model.pth'))
evaluate_classifier(model, test_loader, device)
```

---

**Fim do arquivo de código. Veja próxima seção para Quick Reference.**
