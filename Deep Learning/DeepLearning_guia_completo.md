# 📚 Guia Completo de Deep Learning
## Um Curso Estruturado sobre Redes Neurais Profundas e IA Moderna

---

# MÓDULO 1: FUNDAMENTOS DE DEEP LEARNING

## 1.1 O que é Deep Learning?

Deep Learning é um subcampo do Machine Learning que utiliza **redes neurais com múltiplas camadas (profundas)** para aprender representações hierárquicas dos dados. Diferencia-se de ML tradicional por **não requerer engenharia manual de features**.

### 1.1.1 Evolução Histórica

**1943** - McCulloch & Pitts: Primeiro modelo de neurônio artificial  
**1958** - Rosenblatt: Perceptron (primeira NN treinável)  
**1974-1980** - Invernos da IA: Limitações computacionais  
**1986** - Rumelhart, Hinton, Williams: Backpropagation revoluciona treinamento  
**2006** - Hinton: Deep Learning breaks through (redes profundas são viáveis)  
**2012** - AlexNet: ImageNet ILSVRC, GPU aceleração, Deep Learning explode  
**2014** - VGG, GoogLeNet, Batch Normalization consolidam CNNs  
**2015** - ResNet (152 camadas!), problemas de vanishing gradient resolvidos  
**2017** - Transformers ("Attention Is All You Need"), revoluciona NLP  
**2018** - BERT, GPT pré-treina em larga escala  
**2020-2025** - Modelos multimodais (CLIP), Diffusion Models, Large Language Models

### 1.1.2 Diferença: Machine Learning Tradicional vs Deep Learning

| Aspecto | ML Tradicional | Deep Learning |
|--------|--------|-------------|
| **Features** | Engenharia manual | Aprendidas automaticamente |
| **Dados** | Funciona com poucos | Requer muito volume |
| **Interpretabilidade** | Alta (árvores, regressão) | Baixa (black box) |
| **Computação** | CPU geralmente ok | GPU/TPU necessária |
| **Flexibilidade** | Limitada | Altamente flexível |
| **Custo treino** | Baixo | Alto (dados + compute) |
| **Performance limite** | Teto mais cedo | Escala com dados |

## 1.2 Por que Redes Profundas Funcionam?

### 1.2.1 Aprendizado Hierárquico

Redes profundas aprendem **representações em camadas**:

```
Camada 1 (imagem): Pixels
  ↓
Camada 2: Bordas, texturas
  ↓
Camada 3: Formas, padrões
  ↓
Camada 4: Partes de objetos (olhos, bocas)
  ↓
Camada 5+: Conceitos abstratos (rostos, animais, cenas)
```

Essa hierarquia permite que o modelo capture **abstrações cada vez mais sofisticadas**.

### 1.2.2 Universalidade e Aproximação

**Teorema da Aproximação Universal**: Qualquer função contínua pode ser aproximada por uma rede neural com uma camada oculta suficientemente larga.

**Porém**: Uma camada oculta pode precisar de bilhões de neurônios. **Redes profundas são mais eficientes**, traduzindo-se em menos parâmetros necessários.

### 1.2.3 Representação Distribuída

Cada neurônio em uma camada profunda representa uma "característica abstrata" que é **combinada de formas exponencialmente mais ricas** nas camadas seguintes.

**Exemplo**: Para classificar imagens em 10 bilhões de conceitos possíveis, uma rede profunda com 1 bilhão de parâmetros é mais eficiente que uma rasa.

## 1.3 Perceptron, MLP e Conceito de Camadas

### 1.3.1 Perceptron (Neurônio Simples)

```
         w₁
    x₁ ─→ ⊕
    x₂ ─→ ⊕ → σ(·) → ŷ
    ...  ⊕
    xₙ ─→ ⊕
           ↑
          bias
```

**Equação**:
\[ \hat{y} = \sigma(w^T x + b) = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) \]

Onde:
- \( w_i \): pesos
- \( b \): bias
- \( \sigma(\cdot) \): função de ativação (Sigmoid originalmente)
- \( \hat{y} \): predição

**Limitação**: Perceptron linear **só pode aprender funções linearmente separáveis** (XOR problem).

### 1.3.2 Multi-Layer Perceptron (MLP)

MLP é um Perceptron com **múltiplas camadas ocultas**:

```
INPUT → [Hidden Layer 1] → [Hidden Layer 2] → ... → OUTPUT
(n_in)   (n_h1 neurons)     (n_h2 neurons)        (n_out)
```

Cada camada aprende **transformações não-lineares** que permitem modelar funções arbitrariamente complexas.

### 1.3.3 Anatomia de uma Camada

Cada camada implementa:
\[ z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \]
\[ a^{(l)} = \sigma(z^{(l)}) \]

Onde:
- \( z^{(l)} \): ativação pré-transformação (logits)
- \( a^{(l)} \): ativação pós-transformação
- \( W^{(l)} \): matriz de pesos (peso × anterior)
- \( \sigma(\cdot) \): função de ativação

**Profundidade**: Número de camadas ocultas (não inclui input/output).

## 1.4 Benefícios e Limitações do Deep Learning

### Benefícios

✅ **Aprendizado automático de features**: Sem engenharia manual  
✅ **Escalabilidade**: Performance melhora com mais dados  
✅ **Flexibilidade**: Aplicável a muitos domínios (visão, NLP, áudio)  
✅ **Performance SOTA**: Bate métodos tradicionais em muitos benchmarks  
✅ **End-to-end**: Treina pipeline completo  
✅ **Transfer Learning**: Reutiliza conhecimento de outras tarefas  

### Limitações

⚠️ **Requer muitos dados**: Overfitting em datasets pequenos  
⚠️ **Computacionalmente caro**: GPUs, TPUs caras  
⚠️ **Interpretabilidade baixa**: "Black box" difícil explicar  
⚠️ **Hiperparâmetros sensíveis**: Tuning crucial  
⚠️ **Pode aprender correlações espúrias**: Em dados biased  
⚠️ **Convergência não garantida**: Treino pode falhar  
⚠️ **Lentidão inicial**: Setup complexo vs ML clássico  

---

# MÓDULO 2: FUNDAMENTOS MATEMÁTICOS E COMPUTACIONAIS

## 2.1 Álgebra Linear Aplicada

### 2.1.1 Escalares, Vetores, Matrizes, Tensores

- **Escalar**: Número único → \( x \in \mathbb{R} \)
- **Vetor**: Lista de números → \( \mathbf{x} \in \mathbb{R}^{n} \)
- **Matriz**: Grade 2D → \( X \in \mathbb{R}^{m \times n} \)
- **Tensor**: Generalização N-dimensional → \( X \in \mathbb{R}^{d_1 \times d_2 \times ... \times d_n} \)

**Exemplo em Deep Learning**:
- Imagem: Tensor 4D (batch_size, height, width, channels)
- Sequência de palavras: Tensor 3D (batch_size, seq_length, embedding_dim)

### 2.1.2 Operações Essenciais

**Produto Matriz-Vetor**:
\[ y = Ax \]
Onde A é m×n e x é n×1, resultado é m×1.

**Produto Hadamard (elemento a elemento)**:
\[ (A \odot B)_{ij} = A_{ij} B_{ij} \]

**Traço e Determinante**:
- Traço: \( \text{tr}(A) = \sum_{i} A_{ii} \)
- Det: Mede se matriz é invertível

**Normas**:
- L1: \( \|x\|_1 = \sum_{i} |x_i| \)
- L2: \( \|x\|_2 = \sqrt{\sum_{i} x_i^2} \)

### 2.1.3 Eigenvalues e Eigenvectors

Para matriz A:
\[ A v = \lambda v \]

Onde λ são eigenvalues e v são eigenvectors. **Intuição**: Direções onde A age apenas como escala.

**Aplicação**: Análise de convergência, estabilidade de redes.

## 2.2 Cálculo Diferencial para Deep Learning

### 2.2.1 Derivadas Parciais

Para função \( f(x_1, x_2, ..., x_n) \):
\[ \frac{\partial f}{\partial x_i} \]

Mede taxa de mudança de f em relação a \( x_i \).

### 2.2.2 Gradiente

Vetor de todas as derivadas parciais:
\[ \nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n} \right]^T \]

**Propriedade**: Aponta em direção de maior aumento de f.
**Uso em DL**: Gradiente descendente move na direção \( -\nabla L \) para minimizar loss L.

### 2.2.3 Regra da Cadeia (Chain Rule)

Para composição de funções:
\[ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} \]

**Exemplo**: Se \( y = (x^2 + 1)^3 \), então:
\[ \frac{dy}{dx} = 3(x^2 + 1)^2 \cdot 2x \]

**Em DL**: Backpropagation é aplicação eficiente da chain rule através de grafos de computação.

## 2.3 Backpropagation: Intuição e Matemática

### 2.3.1 Intuição Visual

```
Forward Pass:
x → [Layer1] → [Layer2] → [Layer3] → ŷ → Loss = L

Backward Pass:
∂L/∂w₃ ← ∂L/∂a₃ ← ∂L/∂a₂ ← ∂L/∂a₁ ← ∂L/∂x ← x
```

**Fluxo reverso**: Propaga gradientes de trás para frente, calculando \( \frac{\partial L}{\partial w} \) para cada peso.

### 2.3.2 Algoritmo Formal

Para cada camada l (de trás para frente):

```
1. Calcular δ^(l) = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^(l))   # Erro em z^(l)
2. Calcular ∂L/∂W^(l) = δ^(l) (a^(l-1))^T + λW^(l)   # Gradiente dos pesos
3. Calcular ∂L/∂b^(l) = δ^(l)                         # Gradiente do bias
```

Onde ⊙ é operação Hadamard e λ é regularização.

### 2.3.3 Complexidade

- **Forward pass**: O(n) onde n é número de pesos
- **Backward pass**: Aprox. O(2n) (similar a forward)
- **Vantagem**: Eficiente mesmo com milhões de parâmetros

## 2.4 Gradiente Descendente e Variações

### 2.4.1 Gradiente Descendente Vanilla

```
w ← w - α ∇L(w)
```

Onde α é learning rate.

**Problemas**:
- Convergência lenta em plateaus
- Sensível a learning rate
- Pode oscilar perto ótimo

### 2.4.2 SGD (Stochastic Gradient Descent)

```
w ← w - α ∇L_mini_batch(w)
```

Usa mini-batch ao invés de dataset completo.

**Vantagens**: Convergência mais rápida, menos memória, escapa locais mínimos  
**Desvantagens**: Ruidoso, pode divergir

### 2.4.3 Momentum

```
v ← β v + (1-β) ∇L(w)
w ← w - α v
```

Acumula gradientes passados, acelera convergência.

**β típico**: 0.9

### 2.4.4 Adam (Adaptive Moment Estimation) - Recomendado 2025

```
m ← β₁ m + (1-β₁) ∇L(w)           # 1º momento (média)
v ← β₂ v + (1-β₂) (∇L(w))²        # 2º momento (variância)
m̂ ← m / (1 - β₁^t)                # Bias correction
v̂ ← v / (1 - β₂^t)                # Bias correction
w ← w - α m̂ / (√v̂ + ε)
```

**Parâmetros padrão**: β₁=0.9, β₂=0.999, α=0.001  
**Vantagens**: Adaptativo por parâmetro, converge rápido, robusto  
**Adotado por**: 90% dos papers 2023-2025

## 2.5 Computação em GPU e Otimizações

### 2.5.1 Por que GPUs?

| Aspecto | CPU | GPU |
|--------|-----|-----|
| **Cores** | ~8-16 | ~1000-10000 |
| **Throughput** | Alto por core | Moderado, mas massivamente paralelo |
| **Latência** | Baixa | Alta (para single thread) |
| **Ideal para** | Sequencial | Paralelo (multiplicas de matriz) |

**Deep Learning é Matrix-Heavy**: Multiply-Accumulate (MAC) é perfeito para GPUs.

**Speedup típico**: 50-100× em training com GPU vs CPU.

### 2.5.2 Arquiteturas Populares

- **NVIDIA (CUDA)**: A100, H100, RTX 4090 - SOTA 2025
- **Google (TPU)**: Tensor Processing Unit - otimizado para ML
- **AMD (ROCm)**: Crescendo em adoção

### 2.5.3 Otimizações Práticas

1. **Batch Size**: Maior = melhor utilização GPU, mas menos frequente atualiza
2. **Mixed Precision**: FP32 (precisão) + FP16 (velocidade) → ~2× speedup
3. **Gradient Accumulation**: Simula batch maior com GPU menor
4. **Model Parallelism**: Rede distribuída entre múltiplas GPUs
5. **Quantização**: INT8 em vez FP32 → menor footprint, mais rápido

---

# MÓDULO 3: ARQUITETURAS DE REDES NEURAIS PROFUNDAS

## 3.1 Feedforward Networks (Dense/MLP)

### Estrutura

```
Input → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
(784)        ↓                   ↓                  ↓                 ↓
        150K params        32K params         4K params          640 params
```

### Características

- **Simples**: Apenas multiplicações e ativações
- **Versátil**: Pode ser "token" de muitas arquiteturas
- **Problema**: Não explora estrutura espacial (ineficiente para imagens)

### Quando Usar

✅ Dados tabulares, estruturados  
✅ Regressão simples  
✅ Classificação com features pré-processadas  
❌ Imagens (ineficiente, muitos parâmetros)  
❌ Sequências (sem memória)  

## 3.2 Redes Convolucionais (CNNs)

### 3.2.1 Motivação

Exploram **localidade espacial**: Neurônios vizinhos devem se comunicar.

**Intuição biológica**: Córtex visual tem campos receptivos locais.

### 3.2.2 Camada Convolucional

```
      [Feature Map]   [Filter: 3×3]   [Output]
    ┌───────────────┐  ┌────────────┐ ┌──────────┐
    │ W W W W W W W │  │ w w w      │ │ z z z z  │
    │ W W W W W W W │  │ w w w      │ │ z z z z  │
    │ W W W W W W W │  │ w w w      │ │ z z z z  │
    │ W W W W W W W │  │            │ │ z z z z  │
    │ W W W W W W W │  └────────────┘ │          │
    │ W W W W W W W │                 │          │
    │ W W W W W W W │                 │          │
    └───────────────┘                 └──────────┘
```

**Equação**:
\[ z[i,j] = \sigma\left( \sum_{h} \sum_{w} W[h,w] \cdot X[i+h, j+w] + b \right) \]

Onde W é o filtro (kernel).

### 3.2.3 Parâmetros Importantes

- **Kernel Size**: Tipicamente 3×3 ou 5×5
- **Stride**: Deslocamento do filtro (1 ou 2)
- **Padding**: Adiciona zeros nas bordas (same ou valid)
- **Num Filters**: Número de kernels (aumenta com profundidade)

### 3.2.4 Pooling

Reduz dimensionalidade preservando features:

```
Max Pooling 2×2:
Input  [3 1]    Output [3]
       [2 4]           

Average Pooling 2×2:
Input  [3 1]    Output [2.5]
       [2 4]           
```

**Usado**: Após blocos de conv para reduzir memória e parâmetros.

### 3.2.5 Arquiteturas Clássicas

#### **LeNet (1998)** - Pioneira
```
Input(32×32) → Conv(6,5×5) → Pool → Conv(16,5×5) → Pool → FC → Output
```

#### **AlexNet (2012)** - Revolucionária
```
Input(224×224×3) 
  → Conv(96, 11×11, stride 4) 
  → ReLU → Pool
  → Conv(256, 5×5, pad 2) 
  → ReLU → Pool
  → Conv(384, 3×3) → ReLU
  → Conv(384, 3×3) → ReLU
  → Conv(256, 3×3) → ReLU → Pool
  → FC(4096) → ReLU → Dropout
  → FC(4096) → ReLU → Dropout
  → FC(1000) → Softmax
```

**Inovações**: GPU training, ReLU, Dropout  
**Resultado**: 15.3% top-5 error (vs 26.2% antes)

#### **VGG-16 (2014)** - Simplicidade
```
64 → 64 → Pool → 128 → 128 → Pool → 256 → 256 → 256 → Pool → 
512 → 512 → 512 → Pool → 512 → 512 → 512 → Pool → FC(4096) × 2 → FC(1000)
```

**Insight**: Múltiplos 3×3 filtros = melhor que um 5×5 ou 7×7  
**92.7% top-5 accuracy ImageNet**

#### **ResNet-50 (2015)** - Skip Connections
```
[64 filters]
  ↓
Residual Block: identity → Conv → BN → ReLU → Conv → BN → +input → ReLU
  (Principais blocos: 3 / 4 / 6 / 3 do seu tipo)
  ↓
[256, 512, 1024, 2048 filters]
  ↓
Global Average Pool → FC(1000)
```

**Breakthrough**: 152 camadas! Permite treinar redes muito profundas.

**Skip connection**:
\[ x^{(l+1)} = F(x^{(l)}) + x^{(l)} \]

Permite gradientes fluírem, resolvendo vanishing gradient.

#### **Inception-v3 (2015)** - Multi-Escala
```
[1×1 Conv]    [3×3 Conv]    [5×5 Conv]    [Max Pool]
    ↓             ↓             ↓             ↓
  Concat
```

Múltiplas resoluções simultaneamente = captura features em várias escalas.

#### **EfficientNet (2019)** - Escalamento Composto
```
Fórmula: EfficientNet-B(d, w, r)
- d: profundidade (número de blocos)
- w: largura (número de canais)
- r: resolução (tamanho imagem)

Optimal scaling: aumenta todos 3 de forma balanceada
```

**Resultado**: Melhor accuracy-latency tradeoff (2019-2025)

### 3.2.6 Quando Usar CNNs

✅ Visão computacional (classificação, detecção)  
✅ Processamento de imagens  
✅ Detecção de padrões espaciais locais  
✅ Sinais 1D/2D/3D com estrutura local  
❌ Dados muito abstratos sem localidade  

## 3.3 Redes Recorrentes (RNN, LSTM, GRU)

### 3.3.1 RNN Vanilla

Para sequências, processa elementos um por um:

```
h^(t) = σ(W_h h^(t-1) + W_x x^(t) + b)
y^(t) = W_y h^(t) + b_y
```

Onde h é hidden state (memória).

**Problema**: Vanishing/Exploding Gradient em sequências longas.

### 3.3.2 LSTM (Long Short-Term Memory)

Adiciona "célula de memória" com gates para controlar fluxo:

```
[Input Gate] ──→ ×  
[Forget Gate] → × (cell state) → × ──→ [Output Gate] → hidden state
[Candidate] ──→ +
```

**Equações**:
\[ f_t = \sigma(W_f h_{t-1} + W_f x_t + b_f) \] (Forget gate)
\[ i_t = \sigma(W_i h_{t-1} + W_i x_t + b_i) \] (Input gate)
\[ \tilde{C}_t = \tanh(W_C h_{t-1} + W_C x_t + b_C) \] (Candidate)
\[ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \] (Cell state)
\[ o_t = \sigma(W_o h_{t-1} + W_o x_t + b_o) \] (Output gate)
\[ h_t = o_t \odot \tanh(C_t) \] (Hidden state)

**Vantagem**: Controla fluxo de gradientes, captura dependências longas.

### 3.3.3 GRU (Gated Recurrent Unit)

Versão simplificada de LSTM:

```
[Reset Gate] ⊙ hidden_state → × 
[Update Gate] ⊙ (candidate) → +
```

**Equações**:
\[ r_t = \sigma(W_r x_t + U_r h_{t-1}) \] (Reset)
\[ z_t = \sigma(W_z x_t + U_z h_{t-1}) \] (Update)
\[ \tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1})) \] (Candidate)
\[ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \] (Output)

**vs LSTM**: Menos parâmetros, geralmente treina mais rápido, performance similar.

### 3.3.4 Bidirecional (BiLSTM)

Processa sequência em ambas direções:

```
Forward:  x₁ → x₂ → x₃ → x₄
Backward: x₁ ← x₂ ← x₃ ← x₄
          |    |    |    |
Output:  [h_f,h_b] (concatenado)
```

**Vantagem**: Contexto em ambas direções → melhor para NLP.

### 3.3.5 Quando Usar RNNs

✅ Séries temporais  
✅ Processamento de sequências  
✅ NLP (tradução, sumarização, antes de Transformers)  
✅ Dados onde ordem importa  
❌ Sequências muito longas (Transformers são melhores)  
❌ Paralelo massivo (recorrência é sequencial)  

## 3.4 Transformers e Mecanismo de Atenção

### 3.4.1 Intuição de Atenção

Em tradução "O gato estava sentado":

Tradução para espanhol: "El gato estaba sentado"

Ao gerar cada palavra, modelo deve "focar" em partes relevantes da entrada:
- "El" → foca em "O"
- "gato" → foca em "gato"
- "estaba" → foca em "estava"
- "sentado" → foca em "sentado"

### 3.4.2 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Onde:
- Q (Query): "O que procuro?" (dimensão d_k)
- K (Key): "Onde procuro?" (dimensão d_k)
- V (Value): "O que retorno?" (dimensão d_v)

**Processo**:
1. Compute similarity: \( QK^T \) (batch_size, seq_len, seq_len)
2. Scale: dividir por \( \sqrt{d_k} \) (estabiliza gradientes)
3. Softmax: converte em pesos de probabilidade
4. Multiply valores: agregação ponderada

### 3.4.3 Multi-Head Attention

Não apenas 1 atenção, mas múltiplas em paralelo:

```
Input X
  ├→ Linear → Q₁, K₁, V₁ → Attention₁ → Z₁
  ├→ Linear → Q₂, K₂, V₂ → Attention₂ → Z₂
  ├→ Linear → Q₃, K₃, V₃ → Attention₃ → Z₃
  └→ Linear → Q₈, K₈, V₈ → Attention₈ → Z₈
              ↓
            Concat(Z₁...Z₈) → Linear → Output
```

**Vantagem**: Diferentes cabeças focam em diferentes relações (sintaxe, semântica, coreference, etc.)

### 3.4.4 Transformer Completo

```
┌─────────────────────────────────────┐
│ Encoder                             │
├─────────────────────────────────────┤
│ [Input Embedding + Positional Enc]  │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Multi-Head Attention            │ │ 6× blocos
│ ├─────────────────────────────────┤ │
│ │ Feed-Forward (2 Linear layers)  │ │
│ │ (Residual + Layer Norm em cada) │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Decoder                             │
├─────────────────────────────────────┤
│ [Output Embedding + Positional Enc] │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Masked Multi-Head Attention     │ │ 6× blocos
│ ├─────────────────────────────────┤ │
│ │ Cross-Attention (com Encoder)   │ │
│ ├─────────────────────────────────┤ │
│ │ Feed-Forward                    │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
            ↓
        [Output Linear]
            ↓
        [Softmax]
```

### 3.4.5 Positional Encoding

Transformers processam em paralelo, não sequencial. Como saber ordem?

**Solução**: Adiciona vetor posicional a cada embedding:

\[ PE(pos, 2i) = \sin(pos / 10000^{2i/d}) \]
\[ PE(pos, 2i+1) = \cos(pos / 10000^{2i/d}) \]

Onde pos é posição na sequência, i é dimensão.

### 3.4.6 Quando Usar Transformers

✅ NLP (tradução, sumarização, QA) - MELHOR para 2025  
✅ Visão (Vision Transformer, detecção)  
✅ Multimodal (CLIP, LLaVA, Gemini)  
✅ Sequências longas (não há dependência recorrente)  
✅ Paralelo massivo  
✅ Pré-treinamento em larga escala  
❌ Memória limitada (atenção é O(seq_len²))  
❌ Dados muito pequenos (requer pré-treinamento)  

## 3.5 Autoencoders e Redes Generativas

### 3.5.1 Autoencoders

Comprimem dados em representação latente:

```
Input → [Encoder] → Latent (comprimido) → [Decoder] → Reconstructed
(784)    256→64      (Z: 10-50 dim)       64→256      (784)
         ↓                                  ↓
      ReLU                               ReLU/Sigmoid
```

**Loss**: Reconstrução MSE entre input e output.

**Usos**:
- Anomaly detection (reconstructions ruins → anomalia)
- Data compression
- Feature extraction
- Denoising (treinado com dados ruidosos)

### 3.5.2 Variational Autoencoders (VAEs)

Autoencoders com distribuição latente probabilística:

```
Encoder: X → Mean (μ), Std (σ) → Sample z ~ N(μ, σ²)
         ↓
      z + noise
         ↓
Decoder: z → Reconstructed X
```

**Loss**:
\[ L = ||X - X̂||² + KL(N(μ, σ²) || N(0, 1)) \]

Primeiro termo: reconstrução  
Segundo termo: regularização (latent deve ser N(0,1) para suavidade)

**Vantagem**: Latent space é contínuo, interpolação funciona, geração nova data.

### 3.5.3 Generative Adversarial Networks (GANs)

Duas redes competem:

```
Generator: Random noise z → Fake image X̂
           ↓
        [Discriminator: Real or Fake?]
           ↓
Discriminator: Real image X / Fake X̂ → Real? (0-1)
```

**Treinamento**:
- Discriminator: Maximize log D(X) + log(1 - D(G(z)))
- Generator: Maximize log D(G(z))

**Resultado**: Generator aprende a gerar imagens realistas.

**Desafio**: Instabilidade, mode collapse, convergência difícil.

### 3.5.4 Diffusion Models (SOTA 2024-2025)

Processo de denoising iterativo:

```
Forward (adiciona ruído):
X₀ → X₁ → X₂ → ... → Xₜ (puro ruído)

Reverse (remove ruído, treinado):
Xₜ → Xₜ₋₁ → ... → X₁ → X₀ (imagem limpa)
```

**Treinamento**: Rede prevê ruído que foi adicionado.

**Vantagem**: Treino estável, melhor qualidade que GANs.

**Desvantagem**: Geração lenta (muitas iterações).

### 3.5.5 Quando Usar Cada Uma

| Tipo | Vantagem | Desvantagem | Uso |
|------|----------|-----------|-----|
| **AE** | Simples, rápido | Reconstrução inferior | Compressão, anomalia |
| **VAE** | Interpolação smooth | Menos fidelidade | Geração controlada |
| **GAN** | Imagens realistas | Instável, mode collapse | Síntese, estilo transfer |
| **Diffusion** | SOTA qualidade | Lento | Texto→imagem, super-res |

---

# MÓDULO 4: TÉCNICAS DE TREINAMENTO E REGULARIZAÇÃO

## 4.1 Inicialização de Pesos

Inicialização pobre → gradientes ruins → treino falha.

### 4.1.1 Inicialização Uniforme

```
W ~ Uniform(-a, a)
```

Problema: Não considera tamanho da camada anterior.

### 4.1.2 Xavier (Glorot) Initialization

```
W ~ Uniform[-√(6 / (n_in + n_out)), √(6 / (n_in + n_out))]
```

Ideal para **Sigmoid, Tanh**.

**Intuição**: Variância de ativações é uniforme entre camadas.

### 4.1.3 He Initialization

```
W ~ Normal(0, √(2 / n_in))
```

Ideal para **ReLU** e variantes.

**Por quê ReLU requer diferente**: ReLU mata 50% das ativações (negativos), então precisa maior variância.

### 4.1.4 Comparação

| Método | Para | Vantagem |
|--------|------|----------|
| Uniform | Histórico | Simples |
| Xavier | Sigmoid/Tanh | Balanceado |
| He | ReLU | Mantém variância |

**Recomendação 2025**: He para ReLU/Leaky-ReLU, Xavier para Tanh (raramente usado).

## 4.2 Funções de Ativação

### 4.2.1 Sigmoid

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

**Intervalo**: (0, 1)  
**Derivada**: \( \sigma'(z) = \sigma(z)(1 - \sigma(z)) \)  
**Problema**: Vanishing gradient (derivada máx 0.25), output layer principalmente.

### 4.2.2 Tanh

\[ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \]

**Intervalo**: (-1, 1)  
**Derivada**: \( \tanh'(z) = 1 - \tanh^2(z) \)  
**Melhora**: Centrado em 0, gradiente máx 1.0.  
**Uso**: RNNs, sequências (antes de Transformers).

### 4.2.3 ReLU (Rectified Linear Unit) - RECOMENDADO

\[ ReLU(z) = \max(0, z) \]

**Intervalo**: [0, ∞)  
**Derivada**: 1 se z > 0, 0 caso contrário  
**Vantagem**: Simples, computacionalmente eficiente, sem vanishing gradient  
**Desvantagem**: "Dying ReLU" (muitos 0s se learning rate alto)  
**Uso**: Camadas ocultas, praticamente universal 2025.

### 4.2.4 Leaky ReLU

\[ \text{Leaky-ReLU}(z) = \begin{cases} z & \text{se } z > 0 \\ \alpha z & \text{se } z \leq 0 \end{cases} \]

Onde α ≈ 0.01 (permite gradiente negativo).

**Vantagem**: Evita "dying ReLU".

### 4.2.5 GELU (Gaussian Error Linear Unit)

\[ GELU(z) = z \cdot \Phi(z) \]

Onde Φ é CDF da distribuição normal.

**Propriedade**: "Suave" transição, usada em Transformers modernos (BERT, GPT).

### 4.2.6 Swish (SiLU)

\[ \text{Swish}(z) = z \cdot \sigma(\beta z) \]

**Vantagem**: Melhor performance que ReLU em alguns casos.  
**Uso**: EfficientNet, modelos recentes.

### 4.2.7 Quando Usar Qual

- **Camadas Ocultas**: ReLU / Leaky-ReLU / GELU / Swish
- **Output (Classificação Binária)**: Sigmoid
- **Output (Classificação Multiclass)**: Softmax
- **Output (Regressão)**: Linear (ou ReLU se y ≥ 0)

## 4.3 Batch Normalization, Layer Norm, Group Norm

### 4.3.1 Batch Normalization

Normaliza ativações por mini-batch:

```
μ_batch = (1/m) Σ zᵢ              # Média do batch
σ_batch = sqrt((1/m) Σ (zᵢ - μ)²) # Std do batch
ẑ = (z - μ_batch) / σ_batch        # Normaliza
z_norm = γ ẑ + β                   # Escala/shift aprendido
```

**Vantagem**:
- Reduz internal covariate shift
- Permite learning rate maior
- Efeito regularizador (reduz overfitting)
- Acelera treinamento ~2-3×

**Desvantagem**:
- Diferente comportamento train vs test (usa média/std acumulada)
- Requer batch size moderado (não bom para tiny batches)

### 4.3.2 Layer Normalization

Normaliza por features (não por batch):

```
μ_layer = (1/d) Σ zⱼ               # Média das features
σ_layer = sqrt((1/d) Σ (zⱼ - μ)²)  # Std das features
ẑ = (z - μ_layer) / σ_layer        # Normaliza
z_norm = γ ẑ + β
```

**Vantagem**:
- Independente de batch size
- Mesmo comportamento train/test
- Padrão em Transformers

### 4.3.3 Group Normalization

Meio termo entre Batch Norm e Layer Norm:

Divide features em grupos, normaliza por grupo.

**Vantagem**: Bom para CNNs com batch size pequeno.

### Comparação

| Método | Eixo Normalizado | Train/Test | Batch Dep | Uso |
|--------|-----------------|-----------|----------|-----|
| Batch Norm | Batch | Diferente | Sim | CNNs clássico |
| Layer Norm | Features | Igual | Não | Transformers |
| Group Norm | Features em grupos | Igual | Não | Small batch CNNs |

## 4.4 Dropout

Desativa aleatoriamente neurônios durante treinamento:

```
Durante treino (com probabilidade p=0.5):
z_dropped = z ⊙ mask   (onde mask ~ Bernoulli(1-p))
z_scaled = z_dropped / (1-p)   # Scaling

Durante teste: Sem dropout, usa todas ativações
```

**Intuição**: Treina ensemble de sub-redes, força co-adaptações.

**Efeito**: Regularização forte, reduz overfitting.

**Típicos p**: 0.1-0.5 (maior nas FC layers).

## 4.5 Data Augmentation

Cria variações dos dados durante treinamento:

**Imagens**:
- Rotação, Flip, Zoom, Crop
- Color jitter, Gaussian blur
- Mixup (combina 2 imagens): \( x_{aug} = \lambda x_1 + (1-\lambda) x_2 \)
- CutMix (copia patches)
- RandAugment (aplica random ops)

**Text**:
- Backtranslation
- Synonym replacement
- Random insertion/deletion

**Vantagem**: Aumenta dataset efetivamente, reduz overfitting.

## 4.6 Early Stopping

Para treinamento quando validation loss para de melhorar:

```
best_val_loss = ∞
patience = 10  # épocas sem melhora

for epoch in range(max_epochs):
    train_model()
    val_loss = evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 10
        save_checkpoint()
    else:
        patience -= 1
        if patience == 0:
            break  # Para aqui
```

**Benefício**: Evita overfitting automático.

## 4.7 Learning Rate Scheduling

Reduz learning rate ao longo do tempo:

### Step Decay
```
α(epoch) = α₀ × 0.1^(epoch // 10)
```

### Exponential Decay
```
α(epoch) = α₀ × e^(-k × epoch)
```

### Cosine Annealing
```
α(epoch) = α_min + (α_max - α_min) × (1 + cos(π × epoch/total))/2
```

**Vantagem**: Converge melhor, evita oscilar perto ótimo.

---

# MÓDULO 5: IMPLEMENTAÇÃO PRÁTICA

[Continuação no arquivo DeepLearning-Codigo-Pronto.md - os códigos práticos são extensos e continuarão lá]

---

# MÓDULO 6: AVALIAÇÃO E MÉTRICAS

## 6.1 Classificação

### Matriz de Confusão

```
         Predito Positivo | Predito Negativo
Positivo Real:  TP         |      FN
Negativo Real:  FP         |      TN
```

### Métricas Derivadas

**Accuracy**: \( \frac{TP + TN}{TP + TN + FP + FN} \) - Use apenas dados balanceados  
**Precision**: \( \frac{TP}{TP + FP} \) - Taxa de falsos positivos  
**Recall (Sensitivity)**: \( \frac{TP}{TP + FN} \) - Taxa de falsos negativos  
**F1-Score**: \( 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \) - Balanceado  
**AUC-ROC**: Área sob curva ROC (trade-off True Positive Rate vs False Positive Rate)  
**PR-AUC**: Melhor para dados muito desbalanceados (classe rara importante)

### Quando Usar Qual

- **Accuracy**: Dados balanceados
- **F1**: Balanço precision-recall
- **AUC-ROC**: Comparação de modelos, dados balanceados
- **PR-AUC**: Dados muito desbalanceados, anomalia
- **Precision**: Minimizar falsos positivos (e.g., diagnóstico)
- **Recall**: Minimizar falsos negativos (e.g., detecção câncer)

## 6.2 Regressão

**MAE (Mean Absolute Error)**:
\[ MAE = \frac{1}{n} \sum |y_i - \hat{y}_i| \]

**MSE (Mean Squared Error)**:
\[ MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \]

**RMSE**:
\[ RMSE = \sqrt{MSE} \]

**R² (Coeficiente de Determinação)**:
\[ R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} \]

## 6.3 Análise de Overfitting/Underfitting

Overfitting: Train ↑, Val ↓  
Underfitting: Ambos baixos  
Bem-ajustado: Train ≈ Val, ambos altos

## 6.4 Validação Cruzada

Divide dados em k folds:

```
Fold 1: Train [2,3,4,5], Test [1]
Fold 2: Train [1,3,4,5], Test [2]
Fold 3: Train [1,2,4,5], Test [3]
Fold 4: Train [1,2,3,5], Test [4]
Fold 5: Train [1,2,3,4], Test [5]

Final score = média dos 5 testes
```

---

# MÓDULO 7: APLICAÇÕES PRÁTICAS

## 7.1 Visão Computacional

### Classificação de Imagens
- **Entrada**: Imagem
- **Saída**: Classe
- **Arquitetura**: CNN (ResNet, EfficientNet, ViT)
- **Example**: ImageNet classification

### Detecção de Objetos
- **Entrada**: Imagem
- **Saída**: Bounding boxes + classes
- **Arquitetura**: YOLO, Faster R-CNN, SSD
- **Example**: Detecção de pedestres, placas

### Segmentação Semântica
- **Entrada**: Imagem
- **Saída**: Máscara pixel-level
- **Arquitetura**: U-Net, FCN, Transformers
- **Example**: Segmentação médica, cena

### Detecção de Poses
- **Entrada**: Imagem/Vídeo
- **Saída**: Articulações (x,y)
- **Arquitetura**: OpenPose, MediaPipe
- **Example**: Fitness, análise de movimento

## 7.2 Processamento de Linguagem Natural (NLP)

### Análise de Sentimentos
- **Entrada**: Texto
- **Saída**: Sentimento (positivo/negativo/neutro)
- **Arquitetura**: BERT, RoBERTa
- **Example**: Reviews, redes sociais

### Tradução Automática
- **Entrada**: Texto em idioma A
- **Saída**: Texto em idioma B
- **Arquitetura**: Transformer Seq-to-Seq
- **Example**: Google Translate

### Sumarização de Texto
- **Entrada**: Longo texto
- **Saída**: Resumo conciso
- **Arquitetura**: BART, T5
- **Example**: News, documentos legais

### Reconhecimento de Entidades (NER)
- **Entrada**: Texto
- **Saída**: Entidades + tipos
- **Arquitetura**: BERT + CRF
- **Example**: Extração de nomes, organizações

### Question Answering
- **Entrada**: Pergunta + contexto
- **Saída**: Resposta (span ou gerada)
- **Arquitetura**: BERT, RoBERTa
- **Example**: SQuAD, Jeopardy

## 7.3 Séries Temporais e Previsão

### Previsão de Estoque
- **Entrada**: Histórico de preços
- **Saída**: Preço futuro
- **Arquitetura**: LSTM, GRU, TCN, Transformer
- **Métrica**: RMSE, MAE

### Previsão de Carga Elétrica
- **Entrada**: Consumo histórico + features (hora, dia)
- **Saída**: Carga prevista
- **Arquitetura**: LSTM/GRU outperforms RNN vanilla

### Detecção de Anomalias
- **Entrada**: Série temporal
- **Saída**: Scores de anomalia
- **Arquitetura**: Autoencoder, Isolation Forest + NN
- **Example**: Detecção de fraude, falhas de equipamento

## 7.4 Sistemas de Recomendação

### Collaborative Filtering
- **Entrada**: User-item interactions
- **Saída**: Items recomendados
- **Arquitetura**: Embeddings + Neural Network
- **Example**: Netflix, Amazon

### Content-Based
- **Entrada**: Features de item
- **Saída**: Items similares
- **Arquitetura**: Siamese Networks, Metric Learning

### Hybrid
- Combina collaborative + content-based

## 7.5 Aplicações em Saúde

### Diagnóstico de Doenças
- **Entrada**: Imagens médicas (X-ray, CT, MRI)
- **Saída**: Diagnóstico + confiança
- **Arquitetura**: ResNet, DenseNet, Transformers
- **Métrica**: Accuracy, AUC-ROC (sensibilidade > especificidade)
- **Example**: Detecção de câncer com 95%+ accuracy

### Previsão de Mortalidade
- **Entrada**: Dados do paciente (age, vitals, labs)
- **Saída**: Risco de morte
- **Arquitetura**: Dense networks, Transformers
- **Example**: ICU prediction

## 7.6 Comparação: Deep Learning vs Transfer Learning vs Fine-Tuning

| Abordagem | Dados Req. | Tempo | Performance | Quando Usar |
|-----------|----------|--------|-----------|-----------|
| **DL from scratch** | Muito (>100K) | Alto | Excelente | Dados únicos abundantes |
| **Transfer Learning** | Moderado (1K-10K) | Baixo | Bom | Domínio similar |
| **Fine-tuning** | Baixo (<1K) | Muito Baixo | Muito Bom | Dados especializados |

---

**Continua em DeepLearning_guia_completo_PARTE2.md**
