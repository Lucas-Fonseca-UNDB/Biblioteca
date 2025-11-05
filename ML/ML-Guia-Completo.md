# Guia Completo de Machine Learning
## Curso Estruturado da Teoria à Prática em IA

---

## Índice Geral

1. [Módulo 1: Fundamentos de Machine Learning](#módulo-1-fundamentos-de-machine-learning)
2. [Módulo 2: Matemática e Estatística para ML](#módulo-2-matemática-e-estatística-para-ml)
3. [Módulo 3: Algoritmos Clássicos](#módulo-3-algoritmos-clássicos)
4. [Módulo 4: Deep Learning](#módulo-4-deep-learning)
5. [Módulo 5: Implementação Prática](#módulo-5-implementação-prática)
6. [Módulo 6: Avaliação e Métricas](#módulo-6-avaliação-e-métricas)
7. [Módulo 7: Deploy e Produção](#módulo-7-deploy-e-produção)
8. [Extensões Avançadas](#extensões-avançadas)

---

# Módulo 1: Fundamentos de Machine Learning

## 1.1 Definição e História do ML

**Machine Learning** é o campo de estudo que permite aos computadores aprenderem com dados sem serem explicitamente programados para cada tarefa (Arthur Samuel, 1959).

### Contexto Histórico

- **1943**: McCulloch e Pitts propõem o primeiro neurônio artificial
- **1958**: Frank Rosenblatt inventa o Perceptron
- **1974-1980**: Primeiro "AI Winter" - limitações computacionais
- **1980-1987**: Ressurgimento com sistemas especialistas
- **1987-1993**: Segundo "AI Winter"
- **1997**: Deep Blue derrota Kasparov em xadrez
- **2011-Presente**: Era do Deep Learning e Big Data
- **2017**: Vaswani et al. introduzem Transformers
- **2022+**: Modelos de linguagem de larga escala (LLMs)

### Definição Formal

Segundo Tom Mitchell (1997), um programa aprende com experiência **E** em relação a uma classe de tarefas **T** e medida de desempenho **P**, se seu desempenho em **T**, medido por **P**, melhora com a experiência **E**.

## 1.2 Tipos de Aprendizado

### Aprendizado Supervisionado

**Definição**: O modelo aprende com dados rotulados (features + labels).

**Tipos**:

1. **Classificação**: Predizer categorias discretas
   - Exemplo: Detecção de spam (spam/não-spam)
   - Saída: Categorias finitas

2. **Regressão**: Predizer valores contínuos
   - Exemplo: Preço de um imóvel
   - Saída: Valores numéricos contínuos

### Aprendizado Não-Supervisionado

**Definição**: O modelo encontra padrões em dados sem rótulos.

**Tipos**:

1. **Clustering**: Agrupar dados similares
   - Exemplo: Segmentação de clientes
   - Métodos: K-Means, DBSCAN, Hierarchical Clustering

2. **Redução de Dimensionalidade**: Reduzir número de features
   - Exemplo: Visualização de dados de alta dimensão
   - Métodos: PCA, t-SNE, Autoencoders

3. **Detecção de Anomalias**: Identificar outliers
   - Exemplo: Detecção de fraude
   - Métodos: Isolation Forest, Local Outlier Factor

### Aprendizado por Reforço

**Definição**: Um agente aprende interagindo com um ambiente, recebendo recompensas/punições.

**Componentes**:
- **Agente**: Toma ações
- **Ambiente**: Responde às ações
- **Recompensa**: Sinal numérico de qualidade da ação
- **Política**: Estratégia do agente

**Aplicações**: Jogos (AlphaGo), Robótica, Otimização de recursos

### Aprendizado Semi-Supervisionado

**Definição**: Combina dados rotulados (pequeno) com não-rotulados (grande).

**Técnicas**:
- Self-training
- Co-training
- Pseudo-labeling
- Expectation-Maximization (EM)

**Vantagem**: Reduz custo de anotação manual

## 1.3 Paradigmas Principais

### Classificação

**Objetivo**: Predizer classe de uma amostra

**Tipos de Problemas**:
- **Binária**: 2 classes (sim/não, positivo/negativo)
- **Multiclasse**: > 2 classes mutuamente exclusivas
- **Multilabel**: Múltiplas labels por amostra

**Exemplo Matemático**:

Dado um conjunto de dados \(\{(x_1, y_1), ..., (x_n, y_n)\}\) onde \(x_i \in \mathbb{R}^d\) e \(y_i \in \{1, 2, ..., K\}\), encontrar função \(f: \mathbb{R}^d \rightarrow \{1, ..., K\}\) que minimize:

\[L = \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(f(x_i), y_i)\]

### Regressão

**Objetivo**: Predizer valor contínuo

**Exemplo Matemático**:

Para dados \(\{(x_1, y_1), ..., (x_n, y_n)\}\) onde \(x_i \in \mathbb{R}^d\) e \(y_i \in \mathbb{R}\), encontrar \(f: \mathbb{R}^d \rightarrow \mathbb{R}\) que minimize erro quadrático:

\[MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - f(x_i))^2\]

### Clustering

**Objetivo**: Agrupar dados similares sem rótulos

**K-Means** exemplo:
- Particionar dados em K clusters
- Minimizar variância dentro de cada cluster
- Função objetivo: \(J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2\)

### Redução de Dimensionalidade

**Objetivo**: Reduzir número de features mantendo informação

**PCA** (Principal Component Analysis):
- Encontrar direções de máxima variância
- Projetar dados em subespaço de menor dimensão
- Preserva estrutura essencial

## 1.4 Overfitting e Underfitting

### Definições

**Overfitting**: Modelo memoriza padrões específicos do treinamento, generalizando mal em dados novos.

**Underfitting**: Modelo é muito simples, não captura padrões principais.

**Bias-Variance Tradeoff**: Balanço entre viés (underfitting) e variância (overfitting).

### Análise Matemática

Erro total = Bias² + Variância + Erro Irreduzível

\[E[(f(x) - y)^2] = \text{Bias}^2[f(x)] + \text{Var}[f(x)] + \sigma^2\]

Onde:
- **Bias²**: Erro esperado de um modelo simples
- **Variância**: Sensibilidade a flutuações nos dados
- **σ²**: Ruído inerente aos dados

### Visualização Conceitual

```
            Erro Total
                 |
         ___    /\    ___
        /   \  /  \  /
Erro  /      \/    \/
      |  Underfitting | Optimal | Overfitting |
      |     (High      |         |   (Low
      |      Bias)     |         |    Bias)
      ---------------------------------------->
               Complexidade do Modelo
```

### Estratégias de Prevenção

1. **Validação Cruzada**: Avaliar em múltiplos subconjuntos
2. **Regularização**: Penalizar complexidade (L1, L2)
3. **Early Stopping**: Parar treinamento quando val_loss aumenta
4. **Data Augmentation**: Aumentar dados de treinamento
5. **Dropout**: Desativar neurônios aleatoriamente
6. **Redução de features**: Usar menos variáveis

---

# Módulo 2: Matemática e Estatística para ML

## 2.1 Álgebra Linear

### Conceitos Fundamentais

**Escalar**: Número único \(x \in \mathbb{R}\)

**Vetor**: Array de números \(\mathbf{v} = [v_1, v_2, ..., v_n]^T \in \mathbb{R}^n\)

**Matriz**: Array 2D \(\mathbf{A} \in \mathbb{R}^{m \times n}\)

**Tensor**: Array n-dimensional (generalização)

### Operações Essenciais

**Produto Escalar**:
\[\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \mathbf{u}^T\mathbf{v}\]

**Norma (L2)**:
\[||\mathbf{v}||_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{\mathbf{v}^T\mathbf{v}}\]

**Norma (L1)**:
\[||\mathbf{v}||_1 = \sum_{i=1}^{n} |v_i|\]

**Multiplicação de Matrizes**:
\[(\mathbf{A}\mathbf{B})_{ij} = \sum_{k=1}^{p} A_{ik}B_{kj}\]

**Transposta**:
\[(\mathbf{A}^T)_{ij} = A_{ji}\]

**Propriedades Importantes**:
- \((\mathbf{A}\mathbf{B})^T = \mathbf{B}^T\mathbf{A}^T\)
- \((\mathbf{A}^{-1})^T = (\mathbf{A}^T)^{-1}\)

### Decomposição de Matrizes

**Determinante**: Mede invertibilidade e volume

\[\det(\mathbf{A}) = 0 \Rightarrow \mathbf{A} \text{ singular (não invertível)}\]

**Eigenvalores e Eigenvectores**:

\[\mathbf{A}\mathbf{v} = \lambda \mathbf{v}\]

- \(\mathbf{v}\): eigenvector
- \(\lambda\): eigenvalue
- Encontrado resolvendo: \(\det(\mathbf{A} - \lambda\mathbf{I}) = 0\)

**Singular Value Decomposition (SVD)**:

\[\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T\]

- \(\mathbf{U}, \mathbf{V}\): matrizes ortogonais
- \(\mathbf{\Sigma}\): valores singulares (diagonais)
- Fundamental para: PCA, redução de dimensionalidade, compressão

## 2.2 Cálculo e Otimização

### Derivadas e Gradientes

**Derivada Parcial**:
\[\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_n)}{h}\]

**Gradiente** (vetor de derivadas parciais):
\[\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right]^T\]

**Interpretação**: Aponta na direção de maior aumento

### Chain Rule

Para composição de funções \(z = f(g(\mathbf{x}))\):

\[\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial f}{\partial g_j} \frac{\partial g_j}{\partial x_i}\]

**Forma Vetorial**:
\[\nabla_{\mathbf{x}} z = \left(\frac{\partial \mathbf{g}}{\partial \mathbf{x}}\right)^T \nabla_{\mathbf{g}} f\]

Fundamental para **backpropagation** em redes neurais.

### Otimização: Gradient Descent

**Ideia**: Caminhar na direção oposta do gradiente

**Algoritmo**:
```
1. Inicializar w aleatoriamente
2. Para cada epoch:
   3. Calcular gradiente: ∇L(w)
   4. Atualizar: w := w - η∇L(w)
   5. Se convergeu: parar
```

Onde \(\eta\) é a taxa de aprendizado (learning rate).

**Convergência**: Garante mínimo local para funções convexas

### Variações do Gradient Descent

**Batch Gradient Descent** (BGD):
\[\mathbf{w} := \mathbf{w} - \eta \nabla L(\mathbf{w})\]
- Usa todos os dados (lento, mas estável)

**Stochastic Gradient Descent** (SGD):
\[\mathbf{w} := \mathbf{w} - \eta \nabla L(\mathbf{w}; x_i, y_i)\]
- Usa 1 amostra por vez (rápido, ruidoso)

**Mini-batch Gradient Descent**:
\[\mathbf{w} := \mathbf{w} - \eta \frac{1}{B} \sum_{i \in B} \nabla L(\mathbf{w}; x_i, y_i)\]
- Usa B amostras (balanço entre ambos)

**Momentum**:
\[\mathbf{v} := \beta \mathbf{v} - \eta \nabla L(\mathbf{w})\]
\[\mathbf{w} := \mathbf{w} + \mathbf{v}\]
- Acelera convergência com "inércia"

**Adam** (Adaptive Moment Estimation):
\[\mathbf{m} := \beta_1 \mathbf{m} + (1-\beta_1)\nabla L\]
\[\mathbf{v} := \beta_2 \mathbf{v} + (1-\beta_2)(\nabla L)^2\]
\[\mathbf{w} := \mathbf{w} - \eta \frac{\mathbf{m}}{\sqrt{\mathbf{v}} + \epsilon}\]
- Adapta taxa para cada parâmetro
- Mais eficiente em prática

## 2.3 Probabilidade e Estatística

### Distribuições de Probabilidade

**Distribuição Normal** (Gaussiana):

\[f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}\]

- \(\mu\): média
- \(\sigma^2\): variância
- Ubíqua em ML: muitas variáveis naturais são gaussianas

**Distribuição de Bernoulli**:

\[P(X=k) = p^k(1-p)^{1-k}, \quad k \in \{0,1\}\]

- Modela eventos binários
- Base para regressão logística

**Distribuição Multinomial**:

\[P(X_1=k_1, ..., X_m=k_m) = \frac{n!}{k_1!...k_m!} p_1^{k_1}...p_m^{k_m}\]

- Generaliza Bernoulli para múltiplas categorias

### Teorema de Bayes

\[P(A|B) = \frac{P(B|A)P(A)}{P(B)}\]

**Interpretação em ML**:
- \(P(A|B)\): Posterior (probabilidade do modelo dado dados)
- \(P(B|A)\): Likelihood (prob. dos dados dado modelo)
- \(P(A)\): Prior (crença antes de ver dados)
- \(P(B)\): Evidence (normalizador)

**Aplicação**: Classificação Naive Bayes, Bayesian Inference

### Funções de Verossimilhança

**Likelihood** é probabilidade dos dados observados dado parâmetros:

\[\mathcal{L}(\theta | \mathbf{X}) = P(\mathbf{X} | \theta)\]

**Maximum Likelihood Estimation** (MLE):

\[\hat{\theta} = \arg\max_{\theta} \mathcal{L}(\theta | \mathbf{X})\]

**Exemplo - Regressão Linear**:

Assumir \(y_i \sim \mathcal{N}(\mathbf{w}^T\mathbf{x}_i, \sigma^2)\)

\[\mathcal{L} = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}}\]

MLE leva a minimizar MSE (erro quadrático médio)

### Inferência Estatística

**Estimadores Pontuais**:
- **Média Amostral**: \(\bar{x} = \frac{1}{n}\sum x_i\)
- **Variância Amostral**: \(s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2\)

**Intervalos de Confiança**:

Para média com desvio padrão desconhecido (t-student):

\[\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}\]

**p-valor**: Probabilidade de observar resultado tão extremo sob hipótese nula

---

# Módulo 3: Algoritmos Clássicos

## 3.1 Regressão Linear e Logística

### Regressão Linear

**Modelo**:
\[\hat{y} = \mathbf{w}^T\mathbf{x} + b\]

**Objetivo**: Minimizar MSE

\[L = \frac{1}{n}\sum_{i=1}^{n}(y_i - (\mathbf{w}^T\mathbf{x}_i + b))^2\]

**Solução Analítica** (Normal Equation):

\[\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\]

**Vantagens**:
- Interpretabilidade (coeficientes têm significado)
- Computacionalmente eficiente
- Fundamentação teórica sólida

**Limitações**:
- Assume relação linear
- Sensível a outliers
- Requer inversão de matriz (ineficiente para n >> d)

### Regressão Logística

**Modelo**: Para classificação binária

\[P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}\]

Onde \(\sigma\) é a **função sigmoid**.

**Função de Perda** (Cross-Entropy):

\[L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]\]

**Otimização**: Via gradient descent (não tem solução analítica)

**Extensão Multiclasse** (Softmax):

\[P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K}e^{\mathbf{w}_j^T\mathbf{x}}}\]

## 3.2 Árvores de Decisão e Random Forest

### Árvores de Decisão

**Conceito**: Particionar espaço de features recursivamente

**Algoritmo ID3/C4.5**:
1. Selecionar feature que melhor divide dados (máxima informação)
2. Criar branch para cada valor
3. Recursivamente repetir para subconjuntos
4. Parar quando classe é pura ou critério é atendido

**Critério de Divisão - Entropy**:

\[H(S) = -\sum_{c} p_c \log_2(p_c)\]

**Information Gain**:

\[IG = H(\text{parent}) - \sum \frac{|S_i|}{|S|} H(S_i)\]

**Vantagens**:
- Interpretabilidade visual
- Captura não-linearidades
- Sem normalização necessária

**Limitações**:
- Tendência a overfitting
- Instável (pequenas mudanças causam grandes mudanças)

### Random Forest

**Ideia**: Ensemble de múltiplas árvores, cada uma em subset aleatório

**Algoritmo**:
1. Para b = 1 até B:
   - Amostrar B' amostras com reposição (bootstrap)
   - Treinar árvore T_b em B' com feature subset aleatório
2. Predição final: Média (regressão) ou Votação (classificação)

**Matemática Formal**:

\[\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})\]

**Vantagens sobre árvore única**:
- Reduz variância via averaging
- Decorrelação entre árvores (random subsets)
- Paralelizável
- Feature importance estimável

**Importância de Feature**:

\[Imp(f) = \frac{1}{B}\sum_{b=1}^{B} \sum_{t \in T_b} \mathbb{1}(\text{split em } f) \cdot \Delta IG_t\]

## 3.3 Support Vector Machines (SVM)

**Conceito**: Encontrar hiperplano ótimo que maximize margem entre classes

### SVM Linear

**Objetivo**:

Maximizar margem = \(\frac{2}{||\mathbf{w}||}\) sujeito a:

\[y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad \forall i\]

**Formulação Dual**:

\[\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j\]

Sujeito a: \(0 \leq \alpha_i \leq C\) e \(\sum_i \alpha_i y_i = 0\)

Onde \(\alpha_i\) são multiplicadores de Lagrange e C controla regularização.

### SVM com Kernel

**Ideia**: Mapear para espaço de maior dimensionalidade onde dados são linearmente separáveis

**Truque do Kernel**: 

\[\mathbf{x}_i^T\mathbf{x}_j \Rightarrow K(\mathbf{x}_i, \mathbf{x}_j)\]

Sem calcular mapeamento explícito!

**Kernels Comuns**:

- Linear: \(K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j\)
- Polinomial: \(K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T\mathbf{x}_j + r)^d\)
- RBF (Gaussian): \(K(\mathbf{x}_i, \mathbf{x}_j) = e^{-\gamma||\mathbf{x}_i - \mathbf{x}_j||^2}\)

## 3.4 K-Nearest Neighbors (KNN)

**Princípio**: "Diga-me seus vizinhos e direi quem você é"

**Algoritmo**:
1. Dado novo ponto \(\mathbf{x}\)
2. Encontrar K vizinhos mais próximos no conjunto treinamento
3. Classificação: Votação entre K vizinhos
4. Regressão: Média dos K vizinhos

**Predição**:

\[\hat{y}(\mathbf{x}) = \frac{1}{K}\sum_{i \in \text{K-NN}} y_i\]

**Métrica de Distância** (padrão - Euclidiana):

\[d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{d=1}^{D} (x_{id} - x_{jd})^2}\]

**Vantagens**:
- Extremamente simples
- Sem fase de treinamento (lazy learner)
- Funciona bem com dados não-lineares

**Limitações**:
- Computacionalmente caro em predição (O(n))
- Sensível a features não-normalizadas
- "Curse of dimensionality": espaço fica esparso em altas dimensões

## 3.5 Naive Bayes

**Assunção Fundamental**: Features são condicionalmente independentes dado a classe

\[P(\mathbf{x}|y) = \prod_{d=1}^{D} P(x_d|y)\]

**Classificador**:

\[\hat{y} = \arg\max_y P(y) \prod_{d=1}^{D} P(x_d|y)\]

**Estimação de Probabilidades**:

\[P(x_d|y) = \frac{\text{count}(x_d, y)}{\text{count}(y)}\]

Com suavização Laplace para evitar zeros:

\[P(x_d|y) = \frac{\text{count}(x_d, y) + 1}{\text{count}(y) + K}\]

Onde K é número de categorias.

**Variantes**:
- Multinomial Naive Bayes: Contagem de palavras (NLP)
- Gaussian Naive Bayes: Assume distribuição normal
- Bernoulli Naive Bayes: Features binárias

**Vantagens**:
- Muito rápido
- Funciona bem em alta dimensão (NLP)
- Poucos dados necessários

**Limitações**:
- Assunção de independência frequentemente violada

## 3.6 Clustering

### K-Means

**Objetivo**: Particionar dados em K clusters minimizando variância

**Algoritmo**:
1. Inicializar K centroides aleatoriamente
2. Atribuir cada ponto ao centroide mais próximo
3. Atualizar centroides como média dos pontos
4. Repetir 2-3 até convergência

**Função Objetivo**:

\[J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} ||\mathbf{x} - \mu_k||^2\]

Onde \(\mu_k\) é centroide do cluster k.

**Complexidade**: O(nkd) por iteração

**Limitações**:
- Requer K pré-definido
- Sensível a inicialização
- Assume clusters esféricos

### DBSCAN

**Abordagem Baseada em Densidade**: Clusters são regiões de alta densidade separadas por baixa densidade

**Parâmetros**:
- \(\epsilon\): Raio de vizinhança
- \(MinPts\): Número mínimo de pontos em vizinhança

**Definições**:
- **Core Point**: Tem ≥ MinPts pontos em raio ε
- **Border Point**: Não é core mas próximo de core
- **Outlier**: Nem core nem border

**Vantagens**:
- Descobre número de clusters automaticamente
- Identifica outliers
- Clusters de forma arbitrária

**Limitação**: Sensível a escolha de ε e MinPts

### Hierarchical Clustering

**Ideia**: Construir hierarquia de clusters (dendrograma)

**Aglomerativo** (bottom-up):
1. Começar com cada ponto como cluster
2. Repetidamente mesclar 2 clusters mais próximos
3. Parar quando critério é atendido

**Linkage Criteria**:
- Complete: Máxima distância entre clusters
- Single: Mínima distância (encadeia)
- Average: Distância média
- Ward: Minimiza variância (similar a K-Means)

**Vantagem**: Hierarquia oferece flexibilidade de granularidade

---

# Módulo 4: Deep Learning

## 4.1 Redes Neurais: Fundamentos

### Neurônio Artificial (Perceptron)

**Modelo**:

\[a = \sigma\left(\sum_{i} w_i x_i + b\right) = \sigma(\mathbf{w}^T\mathbf{x} + b)\]

Onde:
- \(\mathbf{x}\): inputs
- \(\mathbf{w}\): weights
- \(b\): bias
- \(\sigma\): função de ativação

**Neurônio Original** (McCulloch-Pitts):

\[y = \begin{cases} 1 & \text{se } \sum_i w_i x_i + b > 0 \\ 0 & \text{caso contrário} \end{cases}\]

Função step não diferenciável → Problema para treinamento!

### Funções de Ativação

**Sigmoid**:

\[\sigma(z) = \frac{1}{1 + e^{-z}}\]

- Saída em (0,1)
- Derivada: \(\sigma'(z) = \sigma(z)(1-\sigma(z))\)
- Problema: Vanishing gradients em extremos

**Tanh**:

\[\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\]

- Saída em (-1,1)
- Centrada em 0 (melhor que sigmoid)
- Mesma problema de vanishing gradients

**ReLU** (Rectified Linear Unit):

\[\text{ReLU}(z) = \max(0, z)\]

- Simples e eficiente computacionalmente
- Derivada: 1 se z > 0, 0 caso contrário
- Problema: Dead ReLU (neurônios que nunca ativam)

**Leaky ReLU**:

\[\text{Leaky ReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{case contrário} \end{cases}\]

Onde \(\alpha\) é pequeno (0.01)

**ELU** (Exponential Linear Unit):

\[\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{case contrário} \end{cases}\]

**GELU** (Gaussian Error Linear Unit):

\[\text{GELU}(z) = z \cdot \Phi(z)\]

Onde \(\Phi\) é CDF da distribuição normal. Usado em Transformers modernos.

## 4.2 Backpropagation

**Conceito**: Calcular gradientes eficientemente via chain rule

**Rede Simples**:

\[z^{(l)} = \mathbf{w}^{(l)} a^{(l-1)} + b^{(l)}\]
\[a^{(l)} = \sigma(z^{(l)})\]

**Loss**:

\[L = \frac{1}{n}\sum_i ||a^{(L)}(\mathbf{x}_i) - y_i||^2\]

Onde L é última camada.

**Algoritmo Backpropagation**:

1. **Forward Pass**: Calcular \(a^{(l)}\) para todas camadas
2. **Backward Pass**: Calcular gradientes de trás para frente
   - \(\frac{\partial L}{\partial a^{(L)}} = 2(a^{(L)} - y)\)
   - \(\frac{\partial L}{\partial w^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot (a^{(l-1)})^T\)
   - \(\frac{\partial L}{\partial a^{(l-1)}} = (\mathbf{w}^{(l)})^T \frac{\partial L}{\partial z^{(l)}}\)

3. **Atualizar Pesos**:
   \[\mathbf{w}^{(l)} := \mathbf{w}^{(l)} - \eta \frac{\partial L}{\partial w^{(l)}}\]

**Complexidade**: O(n parâmetros) para calcular todos gradientes (eficiente!)

## 4.3 Arquiteturas de Redes Neurais

### MLP (Multi-Layer Perceptron)

**Estrutura**: Sequência de camadas densas

```
Input (d) → Hidden (h1) → Hidden (h2) → Output (k)
```

**Universalidade**: Uma rede com 1 camada oculta pode aproximar qualquer função contínua (teorema de aproximação universal).

**Prática**: Múltiplas camadas geralmente melhor (diferentes níveis de abstração).

### CNN (Convolutional Neural Network)

**Motivação**: Capturar estrutura espacial em dados (imagens)

**Operação Convolução**:

\[y[i,j] = \sum_{u,v} w[u,v] \cdot x[i+u, j+v]\]

Onde \(w\) é kernel pequeno (ex: 3×3).

**Vantagens**:
- Compartilha pesos (reduz parâmetros)
- Preserva estrutura espacial
- Detecta features hierárquicas

**Arquitetura Típica**:
```
Input → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → FC → Output
```

**Pooling** (ex: Max Pooling):

\[y = \max(x[2i:2i+2, 2j:2j+2])\]

- Reduz dimensão
- Captura feature mais saliente
- Provides translation invariance

### RNN (Recurrent Neural Network)

**Conceito**: Processar sequências com memória

**Equação Recorrente**:

\[h_t = \sigma(w^{(h)} h_{t-1} + w^{(x)} x_t + b)\]
\[y_t = w^{(o)} h_t + b^{(o)}\]

Onde \(h_t\) é estado oculto (memória).

**Problema**: Vanishing Gradient
- Gradientes decaem exponencialmente ao longo do tempo
- RNNs profundas não aprendem dependências de longo prazo

### LSTM (Long Short-Term Memory)

**Solução** ao vanishing gradient: Mecanismo de "cell state" com gates

**Componentes**:

1. **Forget Gate**: Decide o que esquecer
   \[f_t = \sigma(w^{(f)} [h_{t-1}, x_t] + b^{(f)})\]

2. **Input Gate**: Decide o que adicionar
   \[i_t = \sigma(w^{(i)} [h_{t-1}, x_t] + b^{(i)})\]
   \[\tilde{c}_t = \tanh(w^{(c)} [h_{t-1}, x_t] + b^{(c)})\]

3. **Cell State** (memória de longo prazo):
   \[c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\]

4. **Output Gate**:
   \[o_t = \sigma(w^{(o)} [h_{t-1}, x_t] + b^{(o)})\]
   \[h_t = o_t \odot \tanh(c_t)\]

Onde \(\odot\) é multiplicação element-wise.

**Vantagem**: Cell state flui linearmente (cadeia aditiva), permitindo gradientes longos

### GRU (Gated Recurrent Unit)

**Versão Simplificada** do LSTM:

\[r_t = \sigma(w^{(r)}[h_{t-1}, x_t])\]
\[\tilde{h}_t = \tanh(w^{(h)}[r_t \odot h_{t-1}, x_t])\]
\[h_t = (1-r_t) \odot h_{t-1} + r_t \odot \tilde{h}_t\]

- Menos parâmetros que LSTM
- Performance similar em muitas tarefas
- Mais fácil de treinar

### Transformers e Attention

**Inovação Principal**: Atenção (Vaswani et al., 2017)

**Self-Attention**:

\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\]

Onde:
- \(Q, K, V\): Query, Key, Value (projeções de input)
- \(d_k\): Dimensão de Key
- Cada posição atende todas outras posições em paralelo

**Multi-Head Attention**:

\[\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O\]

Onde cada head calcula self-attention independentemente.

**Transformer Block**:
1. Multi-head self-attention
2. Feed-forward (2 camadas densas com ReLU)
3. Residual connections: output = input + sublayer(input)
4. Layer normalization

**Vantagens**:
- Paralelizável (ao contrário de RNNs)
- Captura dependências de longo prazo
- Escalável para sequências muito longas

**Sucesso**: Base de GPT, BERT, LLMs modernos

## 4.4 Regularização em Deep Learning

### Dropout

**Ideia**: Desativar aleatoriamente fração de neurônios durante treinamento

**Algoritmo**:
```
Durante treinamento:
  Para cada neurônio: Com prob p, output = 0
  
Durante teste:
  Sem dropout, mas escalar outputs por (1-p)
```

**Efeito**: Força rede a aprender representações redundantes, reduzindo co-adaptação

**Dropout Rate**: Típico 0.2-0.5

### Batch Normalization

**Problema**: Mudanças em pesos causam mudanças em distribuição de ativações (internal covariate shift)

**Solução**: Normalizar ativações por batch

\[\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\]

\[y = \gamma \hat{x} + \beta\]

Onde:
- \(\mu_B, \sigma_B\): Média e variância do batch
- \(\gamma, \beta\): Parâmetros aprendíveis (scale e shift)

**Benefícios**:
- Permite learning rates maiores
- Reduz dependência de inicialização
- Efeito regularizador

### Early Stopping

**Monitorar** validation loss durante treinamento:

```
Se validation_loss não melhora por N epochs:
  Salvar modelo do melhor epoch
  Parar treinamento
```

**Previne** overfitting automaticamente

### Regularização L1 e L2

**L2** (Ridge):
\[L_{total} = L + \frac{\lambda}{2n} ||\mathbf{w}||_2^2\]

- Penaliza pesos grandes
- Favorece pesos pequenos mas não-zero

**L1** (Lasso):
\[L_{total} = L + \frac{\lambda}{n} ||\mathbf{w}||_1\]

- Pode forçar alguns pesos a zero
- Feature selection implícita

### Weight Decay em Adam e Otimizadores

```
w := w - lr * (m_hat / sqrt(v_hat + eps)) - lambda * w
```

Decaimento de peso desacoplado melhora desempenho.

## 4.5 Transfer Learning e Fine-tuning

**Conceito**: Usar modelo treinado em dataset grande, adaptar para tarefa nova

**Estratégias**:

1. **Feature Extraction**: Congelar pesos de camadas anteriores, treinar últimas camadas
   ```
   Modelo pré-treinado (frozen) → New FC Layer → Train nova camada
   ```

2. **Fine-tuning**: Descongelar todas camadas, treinar com learning rate muito pequeno
   ```
   Modelo pré-treinado → Ajustar todos pesos com lr baixa
   ```

**Quando usar**:
- Poucos dados disponíveis
- Tarefa relacionada ao modelo pré-treinado
- Recursos computacionais limitados

**Modelos Populares Pré-treinados**:
- ImageNet: ResNet, VGG, EfficientNet
- NLP: BERT, GPT, T5

---

# Módulo 5: Implementação Prática

## 5.1 Pipeline de ML Completo

### 1. Coleta e Exploração de Dados

**Processo**:
1. Coletar dados de múltiplas fontes
2. Exploração estatística descritiva
3. Visualizações (histogramas, scatter plots, correlação)
4. Identificar missings, outliers, desbalanceamento

**Exemplo Python** (vide arquivo ML-Codigo-Pronto.md):

```python
import pandas as pd
import numpy as np

# Carregar dados
df = pd.read_csv('data.csv')

# Exploração básica
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Correlação
correlation = df.corr()
```

### 2. Limpeza de Dados

**Tarefas**:
- Remover/imputar dados faltantes
- Tratar outliers (remover ou transformar)
- Corrigir inconsistências

**Imputação**:

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Estratégia simples
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# K-NN mais sofisticado
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
```

### 3. Feature Engineering

**Criação de Features**:
- Interações entre features
- Transformações polinomiais
- Extrair features de texto/data

**Seleção de Features**:
- Remover features altamente correlacionadas
- Métodos baseados em importância
- RFE (Recursive Feature Elimination)

```python
from sklearn.feature_selection import RFE, SelectKBest, f_classif

# RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

### 4. Normalização e Scaling

**Por que Normalizar**:
- Algoritmos como SVM, KNN usam distância
- Gradient descent converge mais rápido
- Regularização funciona melhor

**Técnicas**:

**StandardScaler** (Z-score):
\[\tilde{x} = \frac{x - \mu}{\sigma}\]

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Usar stats de treino!
```

**MinMaxScaler** (Normalização [0,1]):
\[\tilde{x} = \frac{x - x_{min}}{x_{max} - x_{min}}\]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
```

**RobustScaler** (Robusto a outliers):

Usa mediana e IQR (interquartile range)

### 5. Divisão Treino-Validação-Teste

```python
from sklearn.model_selection import train_test_split

# Estratégia 1: 70-15-15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Melhor: Usar StratifiedKFold para desbalanceamento
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

### 6. Treinamento de Modelo

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predições
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
```

## 5.2 Cross-Validation e Hyperparameter Tuning

### K-Fold Cross-Validation

**Princípio**: Treinar K vezes, cada vez usando diferente fold como validação

```python
from sklearn.model_selection import cross_val_score

# Avaliar modelo com 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'CV Scores: {scores}')
print(f'Mean: {scores.mean():.3f} (+/- {scores.std():.3f})')
```

**Visualização**:
```
Fold 1: Train [2,3,4,5] | Val [1]
Fold 2: Train [1,3,4,5] | Val [2]
Fold 3: Train [1,2,4,5] | Val [3]
Fold 4: Train [1,2,3,5] | Val [4]
Fold 5: Train [1,2,3,4] | Val [5]
```

### Grid Search

**Busca Exaustiva** em grid de hiperparâmetros

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1  # Use todos cores
)

grid_search.fit(X_train, y_train)
print(f'Best params: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.3f}')

# Usar melhor modelo
best_model = grid_search.best_estimator_
```

### Random Search

**Menos Computacionalmente Caro** que Grid Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=20,
    cv=5,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

### Optuna: Otimização Bayesiana

```python
import optuna

def objective(trial):
    # Sugerir hiperparâmetros
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
    # Avaliar com CV
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return score

# Otimizar
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f'Best params: {study.best_params}')
print(f'Best score: {study.best_value}')
```

## 5.3 Versionamento e Reprodutibilidade

### Seed para Reprodutibilidade

```python
import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state

def set_seeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)
```

### MLflow para Versionamento

```python
import mlflow
from mlflow import log_metric, log_params, log_model

mlflow.start_run(run_name="rf_experiment")

# Log parâmetros
mlflow.log_params({
    'n_estimators': 100,
    'max_depth': 20,
    'model_type': 'RandomForest'
})

# Log métricas
mlflow.log_metric('train_accuracy', train_acc)
mlflow.log_metric('val_accuracy', val_acc)

# Log modelo
mlflow.sklearn.log_model(model, 'model')

mlflow.end_run()
```

### DVC (Data Version Control)

```bash
# Inicializar DVC
dvc init

# Rastrear dados/modelos
dvc add data/raw/train.csv
dvc add models/model.pkl

# Reproducir pipeline
dvc repro
```

---

# Módulo 6: Avaliação e Métricas

## 6.1 Métricas de Classificação

### Matriz de Confusão

Para classificação binária:

```
                Predito
              Positivo  Negativo
Real Positivo    TP       FN
     Negativo     FP       TN
```

Onde:
- TP (True Positive): Predito positivo, é positivo
- FN (False Negative): Predito negativo, é positivo
- FP (False Positive): Predito positivo, é negativo
- TN (True Negative): Predito negativo, é negativo

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

### Accuracy (Acurácia)

\[\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}\]

- Métrica geral
- **Problema**: Desbalanceamento de classes (ex: 99% negativos)

### Precision (Precisão)

\[\text{Precision} = \frac{TP}{TP + FP}\]

- De **todos positivos preditos**, quantos são realmente positivos?
- Métrica de "Confiança nas predições positivas"
- Importante quando FP é custoso

### Recall (Sensibilidade/Cobertura)

\[\text{Recall} = \frac{TP}{TP + FN}\]

- De **todos positivos reais**, quantos foram detectados?
- Métrica de "Não deixar passar positivos"
- Importante quando FN é custoso (ex: diagnóstico)

### F1-Score

\[\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\]

- Média harmônica de Precision e Recall
- Usa quando classes desbalanceadas

### ROC-AUC

**ROC Curve** (Receiver Operating Characteristic):

- Eixo X: False Positive Rate = \(\frac{FP}{FP+TN}\)
- Eixo Y: True Positive Rate = Recall = \(\frac{TP}{TP+FN}\)

**AUC** (Area Under Curve):

- Área sob a curva ROC
- Interpretação: Probabilidade modelo classifica positivo aleatório melhor que negativo aleatório
- 1.0 = Perfeito, 0.5 = Aleatório

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
auc_score = auc(fpr, tpr)

# Plotar
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

### Macroaverage vs Microaverage

Para multiclasse:

**Macroaverage**: Calcular métrica para cada classe, depois média

\[\text{Macro-F1} = \frac{1}{K}\sum_{i=1}^{K} F1_i\]

- Trata todas classes igualmente

**Microaverage**: Calcular contribuições globais

\[\text{Micro-F1} = F1(\sum TP_i, \sum FP_i, \sum FN_i)\]

- Ponderado pelo número de amostras por classe

```python
from sklearn.metrics import f1_score

macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')
```

## 6.2 Métricas de Regressão

### Mean Squared Error (MSE)

\[\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2\]

- Penaliza erros grandes (quadrático)
- Mesmas unidades elevadas ao quadrado

### Root Mean Squared Error (RMSE)

\[\text{RMSE} = \sqrt{\text{MSE}}\]

- Mesmas unidades que target
- Interpretável como "erro médio"

### Mean Absolute Error (MAE)

\[\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|\]

- Linear (menos penaliza outliers que MSE)
- Mais robusto

### R² Score

\[R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}\]

- Proporção de variância explicada
- 1.0 = Perfeito, 0.0 = Modelo=média, <0 = Pior que média

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
```

### MAPE (Mean Absolute Percentage Error)

\[\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\]

- Erro em percentual
- Interpretável comercialmente
- Problema: Indefinido se y_i = 0

## 6.3 Métricas de Clustering

### Silhouette Score

Mede quão bem cada ponto se encaixa em seu cluster comparado a outros clusters.

\[s_i = \frac{b_i - a_i}{\max(a_i, b_i)}\]

Onde:
- \(a_i\): Distância média para outros pontos no mesmo cluster
- \(b_i\): Distância média para pontos no cluster mais próximo

\[\text{Silhouette} = \frac{1}{n}\sum_i s_i\]

- Intervalo: [-1, 1]
- 1 = Ótimo, 0 = Sobreposto, -1 = Ruim

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
```

### Davies-Bouldin Index

Razão média de dispersão de cada cluster com seu mais próximo.

\[\text{DB} = \frac{1}{K}\sum_{i=1}^{K} \max_{j \neq i} \frac{S_i + S_j}{d_{ij}}\]

- Menores valores = Melhor (0 = Ótimo)

### Calinski-Harabasz Score

Razão entre variância entre-clusters e intra-cluster.

\[\text{CH} = \frac{B/(K-1)}{W/(n-K)}\]

- Maiores valores = Melhor

```python
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

db = davies_bouldin_score(X, labels)
ch = calinski_harabasz_score(X, labels)
```

## 6.4 Validação Cruzada

### Estratificação

Para dados desbalanceados, usar StratifiedKFold:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Treinar e validar
```

Garante distribuição de classes preservada em cada fold.

### Time Series Split

Para dados temporais:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # train_idx sempre antes de val_idx (respeita ordem temporal)
```

---

# Módulo 7: Deploy e Produção

## 7.1 MLOps: CI/CD para Modelos

**MLOps** integra ML com DevOps:

1. **Data Pipeline**: Coleta, limpeza, validação
2. **Training Pipeline**: Treinamento, versionamento
3. **Model Registry**: Armazenar modelos
4. **Monitoring**: Detectar drift, performance
5. **Retraining**: Automático quando drift

### Exemplo: GitHub Actions + MLflow

```yaml
name: ML Pipeline

on: [push]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train model
        run: python train.py
      - name: Test model
        run: python test.py
      - name: Log to MLflow
        run: python log_model.py
```

## 7.2 Servindo Modelos: FastAPI

**FastAPI** para APIs REST de alta performance:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Carregar modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class InputData(BaseModel):
    features: list[float]

class Prediction(BaseModel):
    prediction: float
    probability: float

@app.post("/predict")
async def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    return Prediction(
        prediction=int(pred),
        probability=float(max(proba))
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Executar**:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Flask (Alternativa Mais Simples)

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    X = np.array(data).reshape(1, -1)
    pred = model.predict(X)[0]
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    app.run(debug=True)
```

## 7.3 Monitoramento em Produção

### Drift Detection

**Data Drift**: Distribuição de features muda

```python
from scipy.stats import ks_2samp

def detect_drift(X_train, X_new, threshold=0.05):
    p_values = []
    for i in range(X_train.shape[1]):
        stat, p_value = ks_2samp(X_train[:, i], X_new[:, i])
        p_values.append(p_value)
    
    n_drifts = sum(1 for p in p_values if p < threshold)
    return n_drifts, p_values

n_drifts, p_vals = detect_drift(X_train, X_new_batch)
if n_drifts > threshold_features:
    alert("Data drift detected! Retrain model")
```

**Concept Drift**: Relação entre X e y muda

```python
# Monitorar performance em produção
def check_concept_drift(y_true, y_pred, window_size=1000):
    if len(y_true) >= window_size:
        recent_acc = accuracy_score(
            y_true[-window_size:],
            y_pred[-window_size:]
        )
        if recent_acc < threshold:
            return True  # Concept drift detected
    return False
```

### Performance Tracking

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def log_prediction(features, prediction, confidence, timestamp=None):
    timestamp = timestamp or datetime.now()
    logger.info(f"Pred={prediction}, Conf={confidence}, Time={timestamp}")
    
    # Armazenar em DB para análise posterior
    db.insert({
        'features': features,
        'prediction': prediction,
        'confidence': confidence,
        'timestamp': timestamp
    })
```

## 7.4 Escalabilidade e Otimização

### Quantização

Reduzir precisão (float32 → float16 ou int8) para inferência mais rápida:

```python
import tensorflow as tf

# Quantização para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)
```

### Batch Inference

Processar múltiplas predições eficientemente:

```python
def batch_predict(model, data, batch_size=32):
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        preds = model.predict(batch)
        predictions.extend(preds)
    return np.array(predictions)
```

### Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def predict_cached(features_hash):
    # Evitar predizer mesmos inputs múltiplas vezes
    return model.predict(decode_features(features_hash))

def get_prediction(features):
    features_hash = hashlib.md5(str(features).encode()).hexdigest()
    return predict_cached(features_hash)
```

### Containerização (Docker)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t ml-api .

# Run
docker run -p 8000:8000 ml-api
```

---

# Extensões Avançadas

## Ensemble Methods

### Bagging (Bootstrap Aggregating)

**Princípio**: Treinar múltiplos modelos em amostras com reposição, combinar predições

**Algoritmo**:
1. Para b = 1 até B:
   - Amostrar com reposição dataset original
   - Treinar modelo M_b nessa amostra
2. Predição final: Média (regressão) ou Votação (classificação)

**Reduz Variância**: Variance reduzida por fator 1/B (aproximadamente)

### Boosting

**Princípio**: Treinar modelos sequencialmente, cada focando em exemplos que anteriores erraram

**Adaboost**:

```
1. Inicializar pesos uniformes w_i = 1/n
2. Para t = 1 até T:
   a. Treinar fraco learner h_t em dados ponderados
   b. Calcular erro: ε_t = Σ w_i 𝕀(h_t(x_i) ≠ y_i)
   c. Peso do modelo: α_t = 0.5 * ln((1-ε_t)/ε_t)
   d. Atualizar pesos: w_i := w_i * exp(-α_t * y_i * h_t(x_i))
   e. Normalizar pesos
3. Predição final: sign(Σ α_t * h_t(x))
```

**Gradient Boosting** (XGBoost, LightGBM):

```
1. f_0(x) = valor inicial (ex: média de y)
2. Para t = 1 até T:
   a. Calcular residuais: r_i = y_i - f_{t-1}(x_i)
   b. Treinar árvore h_t para predizer r_i
   c. f_t(x) = f_{t-1}(x) + ν * h_t(x)  [ν é learning rate]
3. Predição final: f_T(x)
```

**Vantagem**: Reduz Bias e Variância sequencialmente

## AutoML e Neural Architecture Search

**AutoML**: Automatizar seleção de algoritmo, features, hiperparâmetros

### Auto-sklearn

```python
from autosklearn.classification import AutoSklearnClassifier

automl = AutoSklearnClassifier(
    time_left_for_this_task=3600,  # 1 hora
    per_run_time_limit=60
)

automl.fit(X_train, y_train)
print(automl.show_models())
predictions = automl.predict(X_test)
```

### AutoKeras

```python
import autokeras as ak

# Classificação com redes neurais automáticas
clf = ak.ImageClassifier(max_trials=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### NAS (Neural Architecture Search)

Busca espaço de arquiteturas neurais automaticamente usando:
- Reinforcement Learning
- Evolutionary Algorithms
- Bayesian Optimization

## Federated Learning

**Conceito**: Treinar modelos sem centralizar dados (privacidade)

**Algoritmo FedAvg**:

```
1. Servidor inicializa pesos w_0
2. Para round t:
   a. Servidor envia w_t para K clientes
   b. Cada cliente k:
      - Baixa w_t
      - Treina em seus dados locais: w_k,t = w_t - η∇L_k(w_t)
   c. Servidor agrega: w_{t+1} = (1/K) Σ w_k,t
```

**Aplicação**: Modelos de teclado em smartphones, análise médica com privacidade

## Explicabilidade de Modelos

### SHAP (SHapley Additive exPlanations)

Baseia-se em teoria dos jogos (Shapley values)

```python
import shap

# Explicar predições de modelo
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualizar
shap.summary_plot(shap_values, X)
shap.dependence_plot("feature_name", shap_values, X)
```

**Interpretação**: Quanto cada feature contribui para mudança de predição vs baseline

### LIME (Local Interpretable Model-agnostic Explanations)

Aproximar modelo complexo localmente com modelo interpretável

```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names
)

exp = explainer.explain_instance(X_test[0], model.predict_proba)
exp.show_in_notebook()
```

### Integrated Gradients

Atribui importância calculando gradientes ao longo caminho de baseline para input

\[\text{IG}_i(x) = (x_i - x'_i) \int_0^1 \frac{\partial f(x' + \alpha(x-x'))}{\partial x_i} d\alpha\]

---

## Referências Fundamentais

### Livros Clássicos

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

### Papers Seminais

- Vaswani, A., et al. (2017). "Attention Is All You Need" - Transformers
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition" - CNNs
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory" - LSTM
- Breiman, L. (2001). "Random Forests" - Random Forest

### Datasets Populares

- **MNIST**: Dígitos escritos à mão
- **CIFAR-10/100**: Imagens 32×32
- **ImageNet**: 1.2M imagens, 1000 classes
- **IMDB**: Reviews de filmes
- **UCI ML Repository**: Datasets variados

### Bibliotecas Essenciais

- **scikit-learn**: Algoritmos clássicos
- **TensorFlow/Keras**: Deep Learning
- **PyTorch**: Deep Learning flexível
- **pandas**: Data manipulation
- **numpy**: Computação numérica
- **matplotlib/seaborn**: Visualização
- **XGBoost/LightGBM**: Gradient Boosting
- **MLflow**: Versionamento

---

**Conclusão**: Este guia fornece fundações sólidas teóricas e práticas para dominar Machine Learning. O caminho para expertise envolve estudo contínuo, experimentação com dados reais e implementação de projetos complexos.
