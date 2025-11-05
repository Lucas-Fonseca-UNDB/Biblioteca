# Guia Completo de Retrieval-Augmented Generation (RAG)
## Um Curso Estruturado sobre IA, LLMs e Sistemas de Recuperação de Informação

---

## 📋 Índice Geral

1. [Módulo 1: Fundamentos de RAG](#módulo-1-fundamentos-de-rag)
2. [Módulo 2: Arquitetura e Componentes Principais](#módulo-2-arquitetura-e-componentes-principais)
3. [Módulo 3: Embeddings e Recuperação](#módulo-3-embeddings-e-recuperação)
4. [Módulo 4: Integração com Modelos de Linguagem](#módulo-4-integração-com-modelos-de-linguagem)
5. [Módulo 5: Implementação Prática](#módulo-5-implementação-prática)
6. [Módulo 6: Avaliação e Métricas](#módulo-6-avaliação-e-métricas)
7. [Módulo 7: Casos de Uso e Melhores Práticas](#módulo-7-casos-de-uso-e-melhores-práticas)
8. [Módulo 8: Técnicas Avançadas](#módulo-8-técnicas-avançadas)

---

## **MÓDULO 1: FUNDAMENTOS DE RAG**

### Objetivos de Aprendizado
- Compreender a motivação por trás do RAG
- Diferenciar RAG de abordagens tradicionais de QA e LLMs puros
- Identificar os limites dos LLMs e como RAG os resolve
- Reconhecer benefícios e limitações do paradigma RAG

### 1.1 O que é RAG e Por Que Surgiu

**Retrieval-Augmented Generation (RAG)** é um paradigma que combina dois componentes fundamentais:

1. **Retriever**: Um mecanismo que busca informações relevantes de uma fonte externa de conhecimento
2. **Generator**: Um modelo de linguagem que produz respostas usando tanto a query do usuário quanto o contexto recuperado

A necessidade de RAG surgiu de limitações críticas dos LLMs modernos:

#### Limitações dos LLMs Puros

| Limitação | Problema | Exemplo |
|-----------|----------|---------|
| **Conhecimento estático**| Treinados apenas com dados históricos; não conhecem eventos recentes | Perguntar sobre notícias de hoje a um GPT-3 treinado até 2021 |
| **Alucinação (Hallucination)**| O modelo inventa informações quando não tem conhecimento | LLM responde com confiança um fato falso sobre uma empresa específica |
| **Falta de contexto específico**| Desconhecimento de dados proprietários da organização | Um chatbot corporativo sem acesso aos manuais internos |
| **Impossibilidade de atualização rápida**| Retreinar é custoso e lento | Incorporar novo conhecimento à medida que é publicado |
| **Problema de "distribuição de conhecimento"**| O conhecimento está espalhado nos parâmetros; difícil de rastrear fontes | "De onde você tirou isso?" → Impossível citar a origem |

**RAG resolve esses problemas transformando o LLM em um "sistema de leitura ativa"**: em vez de depender apenas do conhecimento memorizado, o modelo pode *buscar* informação relevante em tempo real e então *gerar* respostas baseadas nela.

### 1.2 Diferença Entre RAG e Abordagens Tradicionais

#### Geração Pura vs. Geração com Recuperação

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Tradicional (Puro)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query → [LLM com conhecimento parametrizado] → Resposta   │
│                     ↓                                           │
│            (Risco alto de alucinação)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│              Retrieval-Augmented Generation (RAG)                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Query → [Retriever] → Contexto Relevante                       │
│       ↓                              ↓                               │
│       └──────────→ [LLM] ← Contexto Recuperado → Resposta Grounded   │
│                     ↓                                                │
│            (Resposta factualmente precisa com citações)              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

#### Comparação com Métodos Clássicos de QA

|      Aspecto        | QA Clássico (Extractive)  |      LLM Puro        |              RAG             |
|---------------------|---------------------------|----------------------|------------------------------|
| **Entrada**         | Query + Documentos        | Query apenas         | Query + Base de Conhecimento |
| **Saída**           | Span de texto (extrativo) | Texto livre gerado   | Texto livre com grounding    |
| **Fonte de Verdade**| Documentos fornecidos     | Parâmetros do modelo | Documentos + Parâmetros      |
| **Flexibilidade**   | Baixa (limitado a spans)  | Alta (geração livre) | Alta + Factual               |
| **Factualidade**    | Alta (se doc existe)      | Baixa (alucinação)   | Alta (documento rastreável)  |
| **Rastreabilidade** | Sim (span do doc)         | Não                  | Sim (documentos citáveis)    |

### 1.3 Benefícios e Limitações do RAG

#### ✅ Benefícios

1. **Redução de Alucinações**: Responses grounded em dados reais
2. **Conhecimento Atualizado**: Integração com dados em tempo real
3. **Rastreabilidade**: Citations mostram a origem da informação
4. **Custo-Efetivo**: Não requer fine-tuning completo; usa modelos pré-treinados
5. **Flexibilidade**: Fácil atualizar a base de conhecimento sem retreinar
6. **Privacidade**: Dados sensíveis podem ficar em repos locais (sem enviá-los ao LLM)
7. **Autoridade de Domínio**: Incorpora conhecimento específico do domínio

#### ❌ Limitações

1. **Latência**: Requer operação de busca (trade-off velocidade vs. qualidade)
2. **Qualidade de Recuperação**: Se o retriever falha, o gerador não consegue compensar ("garbage in, garbage out")
3. **Contexto Limitado**: LLMs têm janela de contexto finita; nem sempre conseguem usar todo o contexto recuperado
4. **Ranking de Relevância**: Documentos irrelevantes no top-K podem confundir o LLM
5. **Custo Computacional**: Manutenção de bases vetoriais e pipelines de busca
6. **Complexidade de Avaliação**: Difícil distinguir se o erro é do retriever ou do gerador

### 1.4 Casos de Uso Motivadores

|         Caso de Uso          |               Contexto                 |                 Benefício do RAG                       |
|------------------------------|----------------------------------------|--------------------------------------------------------|
| **Q&A Aberto (Open-Domain)** | Perguntas sobre fatos gerais           | Acesso a Wikipedia/web em tempo real                   |
| **Busca Corporativa**        | Funcionários buscam políticas internas | Respostas precisas sobre documentos proprietários      |
| **Suporte Técnico**          | Chatbots respondendo tickets           | Referência a manuais e FAQs; reduz erros               |
| **Análise de Documentos**    | Revisar contatos legais, pesquisas     | Extração de informação contextualizada                 |
| **E-commerce**               | Recomendações de produtos              | Busca semântica + geração de descrições personalizadas |
| **Healthcare**               | Assistência diagnóstica                | Busca de literatura médica + raciocínio do LLM         |

---

## **MÓDULO 2: ARQUITETURA E COMPONENTES PRINCIPAIS**

### Objetivos de Aprendizado
- Entender a estrutura end-to-end de um sistema RAG
- Identificar cada componente e sua função
- Reconhecer variações arquiteturais (RAG-Sequence vs. RAG-Token)
- Compreender o fluxo de dados passo a passo

### 2.1 Estrutura Geral: Componentes Principais

Um sistema RAG clássico possui 4 componentes interdependentes:

```
┌──────────────────────────────────────────────────────────────────┐
│                      ARQUITETURA RAG                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                                │
│  │  Documentos  │                                                │
│  │  (PDFs, web, │                                                │
│  │   DB, APIs)  │                                                │
│  └──────┬───────┘                                                │
│         │ ┌─────────────────────────────────┐                    │
│         ├─→ 1. INDEXADOR (Preprocessamento) │                    │
│         │ └─────┬───────────────────────────┘                    │
│         │       │                                                │
│  ┌──────▼───────▼─────────────────┐                              │
│  │  2. BANCO VETORIAL             │                              │
│  │  (Vector Store: FAISS, Chroma, │                              │
│  │   Weaviate, Pinecone, etc.)    │                              │
│  └──────┬─────────────────────────┘                              │
│         │                                                        │
│         │ ◄─────────────────────────────┐                        │
│         │                               │                        │
│  ┌──────▼─────────────────────────┐  ┌──┴─────────────────┐      │
│  │  3. RETRIEVER                  │  │ User Query (Q)     │      │
│  │  - Conversão em embedding      │  │                    │      │
│  │  - Busca de similaridade       │  └──────┬─────────────┘      │
│  │  - Top-K retrieval             │         │                    │
│  │                                │         │                    │
│  │  Tipos:                        │         │                    │
│  │  • Dense (Embeddings)          │         │                    │
│  │  • Sparse (BM25, TF-IDF)       │         │                    │
│  │  • Hybrid (Ambos)              │         │                    │
│  └──────┬─────────────────────────┘         │                    │
│         │                                   │                    │
│         │ ◄───────────────────────────────┬─┘                    │
│         │                                 │                      │
│  ┌──────▼──────────────────────────────┐  │                      │
│  │  4. GERADOR (LLM)                   │  │                      │
│  │  - Prompt engineering               │  │                      │
│  │  - Context + Query concatenation    │  │                      │
│  │  - Geração de resposta              │  │                      │
│  │                                     │  │                      │
│  │  Input: [Context] + [Query]         │  │                      │
│  │  Output: Resposta grounded          │  │                      │
│  └──────┬──────────────────────────────┘  │                      │
│         │                                 │                      │
│         │ ◄────────────────────────────┬──┘                      │
│         │                              │                         │
│         ▼                              │                         │
│  [Resposta Final + Citations]          │                         │
│                                        │                         │
└────────────────────────────────────────┼─────────────────────────┘
                                         │
                                    User Interface
```

#### 1. **Indexador (Preprocessing & Ingestion)**

Responsável por transformar documentos brutos em forma indexável:

**Passos:**
1. **Carregamento**: Ler documentos (PDFs, TXTs, JSONs, etc.)
2. **Parsing**: Extrair conteúdo estruturado
3. **Limpeza**: Remover ruído, normalizar texto
4. **Chunking**: Dividir em pedaços menores (256-1024 tokens típico)
5. **Metadata Extração**: Tags, datas, autores, etc.
6. **Embedding**: Converter cada chunk em vetor semântico

#### 2. **Banco Vetorial (Vector Store)**

Armazena embeddings e permite busca por similaridade eficiente.

**Funções:**
- Armazenar milhões/bilhões de embeddings
- Buscas rápidas de vizinhos mais próximos (HNSW, IVF)
- Retornar metadados associados
- Suportar filtros por atributos

**Exemplos:** FAISS, Weaviate, Chroma, Milvus, Pinecone

#### 3. **Retriever (Mecanismo de Busca)**

Encontra os documentos mais relevantes para uma query.

**Tipos:**
- **Dense Retriever**: Usa embeddings semânticos (BERT-based, Contriever, ColBERT)
- **Sparse Retriever**: Usa keywords (BM25, TF-IDF)
- **Hybrid Retriever**: Combina ambos (Melhor performance geral)

#### 4. **Gerador (LLM)**

Lê o contexto recuperado e gera uma resposta.

**Entrada:** `[System Prompt] + [Context] + [User Query]`
**Saída:** Resposta contextualizada

### 2.2 Pipeline de Consulta: Passo a Passo

```
EXEMPLO: User pergunta "Qual é a política de férias da empresa?"

┌─ PASSO 1: Query Ingestion ───────────────────────┐
│  Input: "Qual é a política de férias da empresa?"│
│  Ação: Validação e limpeza de texto              │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌─ PASSO 2: Query Embedding ───────────────────────┐
│  Ação: Converter query em vetor (768-dim típico) │
│  Modelo: Mesmo embedding model usado na index    │
│  Resultado: Query Vector Q                       │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌─ PASSO 3: Busca de Similaridade ─────────────────┐
│  Ação: top-k busca (típico k=5)                  │
│  Método: Cosine similarity ou outro              │
│  Resultado: 5 chunks mais similares + scores     │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌─ PASSO 4: Ranking (Opcional) ────────────────────┐
│  Ação: Re-rank usando cross-encoder              │
│  Objetivo: Melhorar ordem de relevância          │
│  Resultado: Chunks reordenados                   │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌─ PASSO 5: Context Assembly ──────────────────────┐
│  Ação: Concatenar chunks em window               │
│  Formato: [Chunk 1]\n[Chunk 2]\n...              │
│  Resultado: Context string                       │
└──────────────────────────────────────────────────┘
                        │
                        ▼
┌─ PASSO 6: Prompt Construction ──────────────────┐
│  Template:                                      │
│  "Use o seguinte contexto para responder:"      │
│  [CONTEXT]                                      │
│  Pergunta: [QUERY]                              │
│  Responda brevemente.                           │
│                                                 │
│  Resultado: Full Prompt para LLM                │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─ PASSO 7: LLM Inference ────────────────────────┐
│  LLM lê prompt completo                         │
│  Gera resposta token-by-token                   │
│  Resposta: "A política de férias é 20 dias...   │
│             (conforme documento X)"             │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─ PASSO 8: Post-Processing ──────────────────────┐
│  Ações:                                         │
│  • Extrair citations (quais chunks foram usados)│
│  • Validar factualidade (opcional)              │
│  • Formatar para user                           │
└─────────────────────────────────────────────────┘
                        │
                        ▼
               [Final Response + Citations]
```

### 2.3 Variações Arquiteturais

#### RAG-Sequence vs. RAG-Token

Existem duas formulações principais de RAG, conforme proposto por Lewis et al. (2020):

**RAG-Sequence (mais comum)**
```
┌─────────────────────────────────────────────┐
│ Retriever retorna documentos UMA VEZ        │
│ Todos os tokens de resposta usam o mesmo    │
│ contexto recuperado                         │
├─────────────────────────────────────────────┤
│ p(y|x) = Σ_z p(z|x) * p(y|z,x)              │
│          (mesmos z para todo y)             │
└─────────────────────────────────────────────┘

EXEMPLO:
Query: "Como fazer bolo de chocolate?"
Retriever busca: [Receita de bolo, técnicas culinária]
Generator usa AMBOS para gerar TODA resposta
```

**RAG-Token (mais flexível mas custoso)**
```
┌────────────────────────────────────────────────┐
│ Pode recuperar DIFERENTES documentos           │
│ para cada token da resposta                    │
├────────────────────────────────────────────────┤
│ p(y|x) = Π_t Σ_z p(z|x,y_<t) * p(y_t|z,x,y_<t) │
│          (z diferente para cada y_t)           │
└────────────────────────────────────────────────┘

EXEMPLO:
Resposta token-by-token com busca dinâmica:
Token 1 ("Misture") → busca "técnicas de mistura"
Token 2 ("ingredientes") → busca "ingredientes bolo"
...
```

**Comparação:**

|       Aspecto     |      RAG-Sequence     |             RAG-Token           |
|-------------------|-----------------------|---------------------------------|
| **Complexidade**  | Simples               | Complexa                        |
| **Custo**         | Uma busca/query       | Busca por token (~50-100+)      |
| **Flexibilidade** | Menos (contexto fixo) | Mais (contexto dinâmico)        |
| **Performance**   | Boa em geral          | Melhor em multi-hop mas custoso |
| **Uso Prático**   | Predominante          | Pesquisa/casos específicos      |

### 2.4 Fluxo de Dados Detalhado

```
┌──────────────────────────────────────────────────────────────┐
│              FLUXO DE DADOS EM RAG                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ FASE 1: OFFLINE (Pré-processamento)                          │
│ ────────────────────────────────────                         │
│ [Documentos Brutos]                                          │
│     ↓ (chunk_size=512, overlap=50)                           │
│ [Chunks: D1, D2, ..., DN]                                    │
│     ↓ (embedding_model="sentence-transformers/...")          │
│ [Embeddings: E1, E2, ..., EN] ∈ ℝ^768                        │
│     ↓ (indexing="hnsw")                                      │
│ [Vector Store com índice]                                    │
│                                                              │
│ FASE 2: ONLINE (Tempo de Inferência)                         │
│ ────────────────────────────────────                         │
│ [Query do usuário]                                           │
│     ↓ (same embedding model)                                 │
│ [Query Embedding] Q ∈ ℝ^768                                  │
│     ↓ (similarity search, k=5)                               │
│ [Top-5 chunks + cosine similarity scores]                    │
│ Exemplo: D_i: score=0.92, D_j: score=0.85, ...               │
│     ↓ (concatenate + truncate to max_context)                │
│ [Context String C]                                           │
│     ↓ (prompt template)                                      │
│ [Prompt P = System + Context + Query]                        │
│     ↓ (LLM forward pass)                                     │
│ [Token generation with attention over P]                     │
│     ↓ (extraction de citations)                              │
│ [Response R + Source References]                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## **MÓDULO 3: EMBEDDINGS E RECUPERAÇÃO**

### Objetivos de Aprendizado
- Entender o conceito e cálculo de embeddings semânticos
- Dominar métricas de similaridade vetorial
- Conhecer bancos vetoriais populares e suas trade-offs
- Aprender otimizações para preparar bases de conhecimento

### 3.1 Embeddings Semânticos: Conceito Fundamental

**O que é um embedding?**

Um embedding é uma **representação numérica densa** de texto que captura significado semântico em espaço vetorial de alta dimensão.

#### Intuição Matemática

```
┌────────────────────────────────────────────────────────┐
│ Transformação: Texto → Vetor                           │
├────────────────────────────────────────────────────────┤
│                                                        │
│ Entrada: "A inteligência artificial é transformadora"  │
│                                                        │
│ ┌─ Tokenização: ["A", "inteligência", "artificial",    │
│ │                "é", "transformadora"]                │
│ │                                                      │
│ ├─ Embedding de cada token:                            │
│ │  [0.2, -0.5, 0.8, ...] (e.g., 768-dim)               │
│ │                                                      │
│ ├─ Composição (mean pooling + attention):              │
│ │  Combinar embeddings de tokens                       │
│ │                                                      │
│ └─ Saída: Vetor final [0.15, -0.3, 0.5, ...] ∈ ℝ^768   │
│                                                        │
│ Propriedade Chave:                                     │
│ Textos semanticamente similares → vetores próximos     │
│ no espaço vetorial                                     │
│                                                        │
└────────────────────────────────────────────────────────┘
```

#### Como São Criados?

Embeddings são aprendidos através de **treinamento de modelos de linguagem**:

1. **Modelos de Base**: BERT, RoBERTa, etc.
   - Treinados com objetivo de predição mascarada
   - Aprendem relações semânticas em linguagem natural

2. **Modelos Especializados em Similaridade**: Sentence-BERT, Contriever, ColBERT
   - Treinados com **contrastive learning**
   - Exemplos: Pares (query, doc_relevante, doc_irrelevante)
   - Objetivo: maximizar similiaridade entre pares relevantes

3. **Fine-tuning em Domínios**
   - Treinar embeddings em dados específicos do domínio
   - Exemplo: embeddings legais, médicos, etc.

**Equação de Contrastive Learning (Simplified):**

```
Loss = -log[ exp(sim(q, d+) / τ) / 
            (exp(sim(q, d+) / τ) + Σ exp(sim(q, d-) / τ)) ]

Onde:
- q: embedding da query
- d+: embedding do documento relevante
- d-: embedding de documentos irrelevantes
- τ: temperatura
- sim(): função de similaridade (coseno, dot-product)
```

### 3.2 Métricas de Similaridade Vetorial

#### Cosine Similarity (Principal)

**Definição matemática:**

```
cos(A, B) = (A · B) / (||A|| × ||B||)
          = Σ(A_i * B_i) / (√(Σ A_i²) × √(Σ B_i²))
```

**Propriedades:**
- Retorna valores em [-1, 1]
- **1**: vetores idênticos (mesma direção)
- **0**: ortogonais (sem correlação)
- **-1**: opostos
- **Invariante a magnitude**: Dois vetores com mesma direção mas magnitudes diferentes têm similaridade 1

**Por que Cosine para embeddings?**

Em espaços de alta dimensão (768-1536 dims típico para LLM embeddings):
- Distâncias Euclidianas tendem a convergir (todos pontos "longe" um do outro)
- Cosine similarity mede **ângulo** entre vetores, não magnitude
- Mais interpretável e estável

**Exemplo Prático:**

```python
# Dois embeddings 3D (simplificado)
A = [1, 0, 0]  # Representa conceito "gato"
B = [0.9, 0.2, 0.05]  # Representa conceito "felino" (similar)
C = [0, 1, 0]  # Representa conceito "máquina" (diferente)

cos(A, B) = (1×0.9 + 0×0.2 + 0×0.05) / (1 × √(0.81+0.04+0.0025))
          = 0.9 / 0.929 ≈ 0.968 (Muito similar!)

cos(A, C) = (1×0 + 0×1 + 0×0) / (1 × 1)
          = 0 (Ortogonais - conceitos independentes)
```

#### Outras Métricas de Similaridade

|         Métrica        |              Fórmula            |               Quando Usar           |                      Pros/Cons               |
|------------------------|---------------------------------|-------------------------------------|----------------------------------------------|
| **Euclidean Distance** | √(Σ(A_i - B_i)²)                | Distâncias absolutas                | Intuição geométrica; ineficiente em alta dim |
| **Manhattan Distance** | Σ\|A_i - B_i\|                  | Espaços estruturados                | Computacionalmente eficiente; menos preciso  |
| **Dot Product**        | A · B                           | Embeddings normalizados             | Rápido; precisa normalização prévia          |
| **Hamming Distance**   | Contagem de diferenças (bits)   | Vetores binários/hash               | Muito rápido; perda de informação            |

#### Escolha Prática

Para RAG com embeddings de LLMs:
✅ **Recomendado: Cosine Similarity**
- Embedding models modernos já estão normalizados (cos é equivalente ao dot-product normalizado)
- Eficiente computacionalmente (FAISS, Pinecone otimizados para cosine)
- Interpretação intuitiva

### 3.3 Bancos Vetoriais Populares

#### FAISS (Facebook AI Similarity Search)

```
┌─────────────────────────────────────────┐
│ Características:                        │
├─────────────────────────────────────────┤
│ • Open-source, desenvolvido pelo Meta   │
│ • Muito rápido (GPU acceleration)       │
│ • Suporta índices: HNSW, IVF, LSH       │
│ • Memory-efficient com quantização      │
│ • Sem suporte nativo a metadados        │
│                                         │
│ Ideal Para: Busca em larga escala,      │
│ prototipagem local, pesquisa            │
└─────────────────────────────────────────┘

# Exemplo de uso:
import faiss
import numpy as np

# Criar índice (1M embeddings de 768-dim)
embeddings = np.random.rand(1000000, 768).astype('float32')
index = faiss.IndexFlatL2(768)  # Euclidean
index.add(embeddings)

# Busca
query = np.random.rand(1, 768).astype('float32')
distances, indices = index.search(query, k=5)

# Para performance, usar índices estruturados:
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(768), 768, 100)
index.train(embeddings)
index.add(embeddings)
```

#### Chroma

```
┌─────────────────────────────────────────┐
│ Características:                        │
├─────────────────────────────────────────┤
│ • Projeto recente, Python-native        │
│ • Foco em developer experience          │
│ • Armazena embeddings + metadados       │
│ • Integração natural com LangChain      │
│ • Suporta persistência local (SQLite)   │
│ • Escalabilidade: até milhões de docs   │
│                                         │
│ Ideal Para: Prototipagem rápida,        │
│ RAG para PDFs/docs, pequeno-médio scale │
└─────────────────────────────────────────┘

# Exemplo de uso:
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="/path/to/data"
))

# Criar collection
collection = client.create_collection(name="documents")

# Adicionar documentos com embeddings
collection.add(
    ids=["doc1", "doc2"],
    embeddings=[[...], [...]],  # auto-generated se embedding_function providida
    metadatas=[{"source": "pdf1"}, {"source": "pdf2"}],
    documents=["Conteúdo do doc1", "Conteúdo do doc2"]
)

# Buscar
results = collection.query(
    query_embeddings=[[...]],
    n_results=5,
    where={"source": {"$eq": "pdf1"}}  # Filtros!
)
```

#### Weaviate

```
┌─────────────────────────────────────────┐
│ Características:                        │
├─────────────────────────────────────────┤
│ • Cloud-native, arquitetura distribuída │
│ • Knowledge Graph + Vector Search       │
│ • GraphQL API poderosa                  │
│ • Suporta metadados e relacionamentos   │
│ • Enterprise-ready com RBAC             │
│ • Multi-modal (texto, imagem)           │
│                                         │
│ Ideal Para: Sistemas corporativos,      │
│ dados complexos, escalas Enterprise     │
└─────────────────────────────────────────┘

# Exemplo de uso (Python):
import weaviate
from weaviate.connect import ConnectionParams

client = weaviate.connect_to_local(port=6379)

# Define schema
client.collections.create(
    name="Document",
    vectorizer_config=weaviate.config.Configure.Vectorizer.text2vec_huggingface(),
    properties=[
        weaviate.config.Property(
            name="title", data_type=weaviate.config.DataType.TEXT
        ),
        weaviate.config.Property(
            name="content", data_type=weaviate.config.DataType.TEXT
        ),
        weaviate.config.Property(
            name="source", data_type=weaviate.config.DataType.TEXT
        ),
    ]
)

# Adicionar dados
collection = client.collections.get("Document")
collection.data.insert({
    "title": "AI 101",
    "content": "Artificial Intelligence fundamentals...",
    "source": "Wikipedia"
})

# Buscar
results = collection.query.hybrid(
    query="O que é IA?",
    limit=5,
    where=weaviate.query.Filter.by_property("source").equal("Wikipedia")
)
```

#### Milvus

```
┌──────────────────────────────────────────┐
│ Características:                         │
├──────────────────────────────────────────┤
│ • Open-source, cloud-native (CNCF)       │
│ • Escalável para bilhões de vetores      │
│ • Kubernetes-ready                       │
│ • Clustering e replicação automática     │
│ • Suporta múltiplos tipos de índices     │
│ • Benchmarks de alta performance         │
│                                          │
│ Ideal Para: Escala web-scale,            │
│ ambientes containerizados, performance   │
│ crítica                                  │
└──────────────────────────────────────────┘

# Exemplo (simplificado):
from pymilvus import connections, Collection, FieldSchema, \
    CollectionSchema, DataType, create_index

connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
]
schema = CollectionSchema(fields)

# Create collection
collection = Collection("documents", schema)

# Index para performance
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 8, "efConstruction": 200}
}
collection.create_index("embedding", index_params)

# Buscar
collection.load()  # Carregar em memória
search_params = {"metric_type": "L2", "params": {"ef": 200}}
results = collection.search(query_vectors, "embedding", search_params, limit=5)
```

#### Pinecone

```
┌──────────────────────────────────────────┐
│ Características:                         │
├──────────────────────────────────────────┤
│ • Fully Managed (SaaS)                   │
│ • Sem overhead operacional               │
│ • Escalabilidade automática              │
│ • Pod-based pricing (previsível)         │
│ • Metadata filtering, sparse-dense hybrid│
│ • Enterprise SLA                         │
│ • Downside: Vendor lock-in, custo        │
│                                          │
│ Ideal Para: Empresas que querem          │
│ zero ops, scale automática, SLA garantida│
└──────────────────────────────────────────┘

# Exemplo de uso:
import pinecone

pinecone.init(api_key="xxx", environment="us-west4-gcp")

# Create index
pinecone.create_index(
    name="documents",
    dimension=768,
    metric="cosine",
    pod_type="p1"
)

index = pinecone.Index("documents")

# Upsert vectors
index.upsert(vectors=[
    ("doc-1", [0.1, 0.2, ..., 0.8], {"source": "pdf1", "page": 1}),
    ("doc-2", [0.2, 0.3, ..., 0.7], {"source": "pdf2", "page": 2}),
])

# Query
results = index.query(
    vector=[0.15, 0.25, ..., 0.75],
    top_k=5,
    filter={"source": {"$eq": "pdf1"}}  # Metadata filter
)
```

#### Comparação Resumida

|       Critério     |       FAISS     |    Chroma     |  Weaviate  |   Milvus   |    Pinecone    |
|--------------------|-----------------|---------------|------------|------------|----------------|
| **Setup**          | Simples         | Muito Simples | Complexo   | Moderado   | Trivial (SaaS) |
| **Escalabilidade** | Até 1B (single) | Até 10M+      | 100B+      | 1B+        | Ilimitada      |
| **Metadados**      | Não nativo      | Sim           | Sim, Graph | Sim        | Sim            |
| **Custo**          | Grátis          | Grátis        | Grátis     | Grátis     | Pago           |
| **Ops**            | Manual          | Mínima        | Kubernetes | Kubernetes | Zero           |
| **Melhor Para**    | Pesquisa, local | Prototipo RAG | Enterprise | Escala     | Simplici       |

### 3.4 Preparação e Otimização da Base de Conhecimento

#### Estratégias de Chunking

A qualidade da recuperação depende muito de como os documentos são divididos.

**Problema Fundamental:**
```
Chunks PEQUENOS demais:
├─ ✓ Fácil encontrar exatamente o relevante
├─ ✗ Perdem contexto
└─ ✗ Fragmentação semântica

Chunks GRANDES demais:
├─ ✓ Mantêm contexto rico
├─ ✗ Dificuldade em recuperação precisa
├─ ✗ Excede janela de contexto do LLM
└─ ✗ Ruído (muito texto irrelevante junto)

Optimal é BALANCEADO (350-512 tokens típico)
```

**Estratégias Principais:**

1. **Fixed-Size Chunking** (Simples)
```python
def chunk_text(text, chunk_size=512, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks
```
Pros: Simples, rápido
Cons: Pode quebrar no meio de sentenças/conceitos

2. **Sentence-Based Chunking** (Melhor)
```python
import nltk
from nltk.tokenize import sent_tokenize

def chunk_by_sentences(text, target_chunk_size=512):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        if len(current_chunk) + len(sent) < target_chunk_size:
            current_chunk += " " + sent
        else:
            chunks.append(current_chunk)
            current_chunk = sent
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
```
Pros: Respeita limites semânticos
Cons: Chunks podem variar em tamanho

3. **Recursive Splitting** (Recomendado)
```python
# Tenta manter estrutura hierárquica
# Split por: "\n\n" (parágrafos) 
#  → "\n" (linhas)
#  → "." (sentenças)
#  → " " (palavras)

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_text(text)
```
Pros: Mantém estrutura; lida com vários formatos
Cons: Mais lento

4. **Hierarchical Chunking** (Avançado)
```
Documento
 ├─ Capítulo 1 (chunk grande = section)
 │  ├─ Seção 1.1 (chunk médio)
 │  │  ├─ Parágrafo 1.1.a (chunk pequeno)
 │  │  └─ Parágrafo 1.1.b
 │  └─ Seção 1.2
 └─ Capítulo 2
```
Permite multi-hop retrieval (buscar no nivel apropriado)

#### Enriquecimento de Metadados

```python
# Cada chunk deve ter:
chunk_with_metadata = {
    "content": "texto do chunk...",
    "source": "documento.pdf",
    "page": 3,
    "section": "Capítulo 2: Introdução",
    "chunk_id": "chunk_001",
    "timestamp": "2024-01-15",
    "tags": ["IA", "RAG", "NLP"],
    "summary": "Resumo em 1-2 sentenças do chunk"  # Útil para retriever
}
```

Benefícios:
- Filtros mais precisos (`where source=="contrato.pdf"`)
- Rastreabilidade de citations
- Ajuste de ranking por metadados

#### Otimizações para Retriever

1. **Query Expansion**
```
Query original: "Como fazer login?"

Expandido:
├─ "authentication process"
├─ "user sign in"
├─ "account access"
└─ "password reset"

Buscar com todas as variações → mais recalls
```

2. **Contextual Window para Chunks**
```
Problema: Um chunk sozinho pode ser ambíguo
Solução: Recuperar chunk + contexto antes/depois

# Padrão "Parent Document":
- Index com chunks pequenos (256 tokens)
- Mas retornar chunks MAIORES que os indexados
```

3. **Embedding Caching**
```python
# Calcular embedding uma vez, reusar múltiplas vezes
embeddings_cache = {}

def get_embedding(text, model):
    if text in embeddings_cache:
        return embeddings_cache[text]
    embedding = model.encode(text)
    embeddings_cache[text] = embedding
    return embedding
```

---

## **MÓDULO 4: INTEGRAÇÃO COM MODELOS DE LINGUAGEM**

### Objetivos de Aprendizado
- Entender como LLMs consomem contexto recuperado
- Dominar estratégias de prompt engineering para RAG
- Aprender técnicas de controle de contexto (chunking, windowing)
- Implementar estratégias anti-alucinação

### 4.1 Como LLMs Consomem Contexto Recuperado

#### Arquitetura Interna de Atenção

Um LLM baseia-se em **Transformer architecture** com mecanismo de atenção:

```
┌────────────────────────────────────────────────────────┐
│         Fluxo de Processamento no Transformer          │
├────────────────────────────────────────────────────────┤
│                                                        │
│ INPUT: [CONTEXT_TOKEN_1, ..., CONTEXT_TOKEN_m,         │
│         QUERY_TOKEN_1, ..., QUERY_TOKEN_n]             │
│                                                        │
│ ↓ Embedding Layer                                      │
│                                                        │
│ [e1, e2, ..., em, q1, q2, ..., qn] ∈ ℝ^d               │
│                                                        │
│ ↓ Positional Encoding (mantém ordem dos tokens)        │
│                                                        │
│ ↓ Multi-Head Self-Attention                            │
│   ┌─────────────────────────────────────┐              │
│   │ Atención = softmax((Q·K^T)/√d) · V  │              │
│   │                                     │              │
│   │ Cada token "attende" (pesa)         │              │
│   │ para todos outros tokens            │              │
│   └─────────────────────────────────────┘              │
│   → Resultado: Cada token sabe qual contexto           │
│     é relevante para ele                               │
│                                                        │
│ ↓ Feed-Forward Networks                                │
│                                                        │
│ ↓ Layer Normalization & Residual Connections           │
│                                                        │
│ [repeat para 12-96 layers dependendo modelo]           │
│                                                        │
│ ↓ Output Layer (LM Head)                               │
│                                                        │
│ [p(token_next | todos_tokens_anteriores)]              │
│ ∈ [0,1]^vocab_size                                     │
│                                                        │
│ OUTPUT: Próximo token (via argmax ou sampling)         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

#### Como Contexto Influencia Geração

**Empiricamente observado (papers):**

1. **Primeira parte do contexto tem mais influência** (primacy bias)
   - Tokens iniciais recebem mais atenção
   - ⚠️ Implicação: Colocar informação crítica no início do contexto

2. **Tokens duplicados em contexto amplificam sua influência**
   - Se informação aparece múltiplas vezes → mais peso
   - ✓ Usar quando quer enforce certa resposta

3. **Contexto muito longo diminui efetividade** (lost in the middle)
   - LLMs tendem a ignorar informação no meio de contextos longos
   - Ótimo em ~500-1000 tokens; decresce depois

4. **Formato e estrutura do contexto importam**
   - Markdown estruturado: melhor
   - XML tags: bom para separar seções
   - Plain text: funciona mas menos eficiente

### 4.2 Estratégias de Prompt Engineering para RAG

#### Template Base Efetivo

```
┌─────────────────────────────────────────────────────────┐
│              PROMPT TEMPLATE RECOMENDADO                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ SYSTEM:                                                 │
│ "Você é um assistente útil e preciso. Responda          │
│  perguntas baseado EXCLUSIVAMENTE no contexto           │
│  fornecido. Se a resposta não estiver no contexto,      │
│  diga 'Não encontrei informação relevante'."            │
│                                                         │
│ USER:                                                   │
│ "Contexto:                                              │
│ ────────────────────────                                │
│ {CONTEXT_HERE}                                          │
│ ────────────────────────                                │
│                                                         │
│ Pergunta: {QUERY}                                       │
│                                                         │
│ Responda em 2-3 sentenças. Cite suas fontes."           │
│                                                         │
│ ASSISTANT:                                              │
│ [LLM gera resposta]                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Tecnicas Avançadas de Prompt Engineering

**1. Chain-of-Thought (CoT) para RAG**
```
SYSTEM: "Você é um assistente analítico que raciocina 
passo a passo."

USER: "{CONTEXT}

Pergunta: {QUERY}

Pense passo a passo:
1. Quais informações no contexto são relevantes?
2. Que conexões posso fazer?
3. Qual é minha resposta final?"

Efeito: LLM gera explicitamente seu raciocínio
       → Mais accurate, hallucina menos
       → Pode validar raciocínio antes de responder
```

**2. Few-Shot Prompting**
```
SYSTEM: "Responda perguntas baseado no contexto.
Aqui estão exemplos:"

EXEMPLO 1:
Contexto: "Python é uma linguagem de programação..."
Pergunta: "O que é Python?"
Resposta: "Python é uma linguagem de programação 
conforme afirmado no documento."

EXEMPLO 2:
[... mais exemplos ...]

USER: "{ATUAL_CONTEXTO}
Pergunta: {ATUAL_QUERY}"

Efeito: LLM tem "padrão" para seguir
       → Mais consistent, evita desvios
```

**3. Structured Output**
```
USER: "{CONTEXT}

Pergunta: {QUERY}

Responda em JSON:
{
  'answer': 'sua resposta aqui',
  'confidence': 0-100,
  'sources': ['chunk1', 'chunk2'],
  'reasoning': 'por que chegou nesta resposta'
}"

Efeito: Output estruturado
       → Fácil de parse automaticamente
       → Includes confidence/reasoning
```

### 4.3 Chunking e Windowing Avançado

#### Problema de Context Window

```
LLM Context Window: 4096 tokens (típico)

Distribuição:
├─ System Prompt: 100 tokens
├─ Contexto Recuperado: 3000 tokens
├─ Query: 50 tokens
└─ Geração (espaço disponível): 946 tokens

PROBLEMA: Resposta pode ser truncada!
```

#### Estratégia 1: Sliding Window
```python
def apply_sliding_window(
    context_chunks: List[str],
    max_context_tokens: int = 2000,
    overlap_ratio: float = 0.1
):
    """
    Em vez de usar todos chunks, usa 'janela' que
    se move sobre o contexto.
    """
    window_tokens = 0
    window = []
    
    for chunk in context_chunks:
        tokens = len(chunk.split())
        
        if window_tokens + tokens > max_context_tokens:
            # Estourou limite, para de adicionar
            break
        
        window.append(chunk)
        window_tokens += tokens
    
    return " ".join(window)

# Uso:
limited_context = apply_sliding_window(
    top_k_chunks,
    max_context_tokens=2000
)
```

#### Estratégia 2: Hierarchical Summarization
```python
def summarize_context(
    chunks: List[str],
    llm,
    target_tokens: int = 1500
):
    """
    Se contexto é muito grande, resumir chunks
    antes de passar para generator.
    """
    summaries = []
    
    for chunk in chunks:
        summary_prompt = f"""
        Resuma o seguinte em 2-3 sentenças:
        
        {chunk}
        """
        summary = llm.generate(summary_prompt)
        summaries.append(summary)
    
    full_context = "\n".join(summaries)
    
    if len(full_context.split()) > target_tokens:
        # Ainda muito grande, sumarizar novamente
        return summarize_context(
            [full_context],
            llm,
            target_tokens
        )
    
    return full_context
```

#### Estratégia 3: Reranking Inteligente
```python
from sentence_transformers import CrossEncoder

def rerank_by_relevance(
    query: str,
    chunks: List[str],
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    top_k: int = 5,
    max_context_tokens: int = 2000
):
    """
    Use cross-encoder para re-ranquear chunks
    baseado em relevância PAIRWISE.
    
    Cross-encoder é mais preciso que dense retrieval
    mas mais custoso (O(n) similarity scores).
    """
    reranker = CrossEncoder(reranker_model)
    
    # Score cada chunk contra query
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    # Sort por score (descendente)
    ranked_chunks = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Selecionar top-K até limite de tokens
    selected = []
    token_count = 0
    
    for chunk, score in ranked_chunks:
        chunk_tokens = len(chunk.split())
        if token_count + chunk_tokens > max_context_tokens:
            break
        selected.append(chunk)
        token_count += chunk_tokens
    
    return selected
```

### 4.4 Controle de Alucinação

#### Tipos de Alucinação em RAG

```
1. ALUCINAÇÃO INTRÍN SECA (não relacionada ao contexto)
   Query: "Quantos satélites tem Marte?"
   Contexto: (sobre Vênus)
   Saída: "Marte tem 12 satélites" (inventado)
   
2. ALUCINAÇÃO CONTEXTUAL (distorção do contexto)
   Contexto: "A população cresceu 10% no ano X"
   Saída: "A população dobrou" (exagerado)
   
3. ALUCINAÇÃO DE CITAÇÃO (fake attribution)
   Saída: "Conforme documento Z... (nunca mencionado)"
   
4. ALUCINAÇÃO COMPOSITIVA (combinação errada)
   Contexto: "A é B" e "C é D"
   Saída: "A é D" (falsa combinação)
```

#### Técnicas de Mitigação

**1. Confidence Scoring**
```python
def score_hallucination_risk(
    generated_text: str,
    context: str,
    llm
) -> float:
    """
    Score de 0-1 indicando risco de alucinação.
    """
    verify_prompt = f"""
    Dado o contexto abaixo:
    
    CONTEXTO:
    {context}
    
    A seguinte declaração é suportada pelo contexto?
    
    DECLARAÇÃO:
    {generated_text}
    
    Responda: SIM, NÃO, ou PARCIAL
    
    Se PARCIAL, explique qual parte é suportada.
    """
    
    verification = llm.generate(verify_prompt)
    
    if "SIM" in verification.upper():
        return 0.0  # Sem risco
    elif "PARCIAL" in verification.upper():
        return 0.5  # Risco moderado
    else:
        return 1.0  # Alto risco
```

**2. Grounding Enforcement via System Prompt**
```python
system_prompt = """
REGRAS RIGOROSAS:
1. Responda EXCLUSIVAMENTE baseado no contexto.
2. Se informação não estiver no contexto, 
   diga: "Não encontrei informação no contexto."
3. Nunca invente ou suponha fatos.
4. Cite a seção específica de cada afirmação.
5. Se tiver dúvida, peça confirmação.

Formato de resposta:
- Declaração: [statement]
- Fonte: [Seção X do documento Y]
- Confiança: [ALTA/MÉDIA/BAIXA]
"""
```

**3. Retrieval-Verification Loop**
```python
def rag_with_verification(
    query: str,
    retriever,
    generator_llm,
    verifier_llm,
    max_iterations: int = 3
) -> str:
    """
    Loop iterativo: Retrieve → Generate → Verify → 
    Se falha verificação, retrieve mais contexto
    """
    
    for iteration in range(max_iterations):
        # Retrieve
        context = retriever.retrieve(query)
        
        # Generate
        response = generator_llm.generate(
            context=context,
            query=query
        )
        
        # Verify
        verify_prompt = f"""
        O seguinte contexto suporta esta resposta?
        
        CONTEXTO: {context}
        RESPOSTA: {response}
        
        Responda: VÁLIDO ou INVÁLIDO
        Se INVÁLIDO, explique por quê.
        """
        
        verification = verifier_llm.generate(verify_prompt)
        
        if "VÁLIDO" in verification:
            return response  # ✓ Response is grounded
        
        # Se inválido, tentar com mais contexto
        print(f"Iteração {iteration+1}: Verificação falhou. "
              f"Razão: {verification}")
        
        # Aumentar k para próxima retrieval
        retriever.k += 2
    
    return response  # Retornar mesmo que não verificado
```

---

## **MÓDULO 5: IMPLEMENTAÇÃO PRÁTICA**

### Objetivos de Aprendizado
- Implementar pipeline RAG funcional de ponta a ponta
- Integrar com LangChain e LlamaIndex
- Trabalhar com APIs de LLM (OpenAI, Anthropic, Mistral)
- Construir sistema completo com persistência

### 5.1 RAG Completo com LangChain

```python
# ═══════════════════════════════════════════════════════════
# IMPLEMENTAÇÃO COMPLETA DE RAG COM LANGCHAIN
# ═══════════════════════════════════════════════════════════

# 1. INSTALAÇÕES
"""
pip install langchain langchain-community
pip install langchain-openai  # ou outro provider
pip install chroma  # Vector store
pip install pypdf  # Para ler PDFs
pip install python-dotenv  # Para gerenciar env vars
"""

# 2. IMPORTS
import os
from dotenv import load_dotenv
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# ───────────────────────────────────────────────────────────
# PASSO 1: CARREGAR DOCUMENTOS
# ───────────────────────────────────────────────────────────

def load_documents(pdf_paths: List[str]) -> List:
    """
    Carrega PDFs e extrai conteúdo.
    """
    documents = []
    
    for pdf_path in pdf_paths:
        print(f"Carregando: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
    
    print(f"Total de documentos carregados: {len(documents)}")
    return documents

# Uso:
pdf_files = ["documento1.pdf", "documento2.pdf"]
raw_documents = load_documents(pdf_files)

# ───────────────────────────────────────────────────────────
# PASSO 2: CHUNKING (DIVISÃO EM PEQUENOS PEDAÇOS)
# ───────────────────────────────────────────────────────────

def split_documents(documents, chunk_size: int = 512, 
                   chunk_overlap: int = 50):
    """
    Divide documentos em chunks menores mantendo contexto.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks criados: {len(chunks)}")
    
    return chunks

chunks = split_documents(raw_documents)

# ───────────────────────────────────────────────────────────
# PASSO 3: CRIAR EMBEDDINGS E STORE VETORIAL
# ───────────────────────────────────────────────────────────

def create_vector_store(
    chunks,
    embedding_model: str = "text-embedding-3-small",
    persist_dir: str = "./chroma_db"
):
    """
    Cria embeddings e armazena em Chroma.
    """
    
    # Usar embeddings OpenAI (ou outro modelo)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    
    # Criar vector store com persistência
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"Vector store criado em: {persist_dir}")
    return vector_store

vector_store = create_vector_store(chunks)

# ───────────────────────────────────────────────────────────
# PASSO 4: CONFIGURAR RETRIEVER
# ───────────────────────────────────────────────────────────

def setup_retriever(vector_store, search_type: str = "similarity", 
                   k: int = 5):
    """
    Cria retriever a partir do vector store.
    """
    
    retriever = vector_store.as_retriever(
        search_type=search_type,  # "similarity" ou "similarity_score_threshold"
        search_kwargs={
            "k": k,  # Top-k resultados
            # "score_threshold": 0.5  # Opcional: min score
        }
    )
    
    return retriever

retriever = setup_retriever(vector_store, k=5)

# ───────────────────────────────────────────────────────────
# PASSO 5: CONFIGURAR LLM GENERATOR
# ───────────────────────────────────────────────────────────

def setup_llm(model: str = "gpt-4", temperature: float = 0):
    """
    Cria instância do LLM para geração.
    """
    
    llm = ChatOpenAI(
        model_name=model,
        temperature=temperature,  # 0 = determinístico
        max_tokens=1024
    )
    
    return llm

llm = setup_llm()

# ───────────────────────────────────────────────────────────
# PASSO 6: CRIAR PROMPT CUSTOMIZADO
# ───────────────────────────────────────────────────────────

def create_rag_prompt():
    """
    Cria template de prompt para RAG.
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Você é um assistente útil especializado 
em responder perguntas baseado em documentos fornecidos.

REGRAS:
1. Responda EXCLUSIVAMENTE baseado no contexto fornecido
2. Se a informação não estiver no contexto, 
   diga: "Não encontrei informação relevante nos documentos"
3. Cite as fontes de suas respostas
4. Seja preciso e conciso"""),
        
        ("human", """Contexto dos documentos:
────────────────────────────────
{context}
────────────────────────────────

Pergunta do usuário: {question}

Responda em 3-4 sentença, citando as fontes.""")
    ])
    
    return prompt_template

prompt = create_rag_prompt()

# ───────────────────────────────────────────────────────────
# PASSO 7: CRIAR CHAIN RAG
# ───────────────────────────────────────────────────────────

from langchain.chains import RetrievalQA

def create_rag_chain(llm, retriever, prompt):
    """
    Combina retriever + LLM em um chain.
    """
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Concatena todos docs
        retriever=retriever,
        return_source_documents=True,  # Retorna chunks usados
        chain_type_kwargs={
            "prompt": prompt
        }
    )
    
    return rag_chain

rag_chain = create_rag_chain(llm, retriever, prompt)

# ───────────────────────────────────────────────────────────
# PASSO 8: USAR O SISTEMA RAG
# ───────────────────────────────────────────────────────────

def query_rag(rag_chain, query: str) -> dict:
    """
    Execute uma query no sistema RAG.
    """
    
    print(f"\n🔍 Pergunta: {query}\n")
    
    result = rag_chain({"query": query})
    
    print(f"📝 Resposta:\n{result['result']}\n")
    
    print("📚 Documentos utilizados:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"  {i}. Página {doc.metadata.get('page', 'N/A')} - "
              f"{doc.metadata.get('source', 'Desconhecida')}")
    
    return result

# Exemplos de uso
queries = [
    "Qual é a política de férias da empresa?",
    "Quais são os benefícios oferecidos?",
    "Como solicitar um dia de folga?"
]

for query in queries:
    query_rag(rag_chain, query)

# ───────────────────────────────────────────────────────────
# PASSO 9: MELHORAMENTOS OPCIONAIS
# ───────────────────────────────────────────────────────────

# Recarregar vector store persistente (próxima execução)
vector_store_loaded = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

# Adicionar documentos novos
new_docs = load_documents(["novo_documento.pdf"])
new_chunks = split_documents(new_docs)
vector_store_loaded.add_documents(new_chunks)

# Usar Reranker para melhorar resultados
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

base_retriever = vector_store_loaded.as_retriever(search_kwargs={"k": 10})

compressor = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
    )
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Usar compression_retriever em vez de base_retriever
```

### 5.2 RAG com LlamaIndex (Alternativa Modular)

```python
# ═══════════════════════════════════════════════════════════
# IMPLEMENTAÇÃO COM LLAMAINDEX (MAIS ESPECIALIZAADO PARA RAG)
# ═══════════════════════════════════════════════════════════

"""
pip install llama-index llama-index-llms-openai
pip install llama-index-embeddings-openai
pip install llama-index-readers-file
pip install pypdf
"""

import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engines import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser

# ───────────────────────────────────────────────────────────
# PASSO 1: CONFIGURAÇÃO GLOBAL
# ───────────────────────────────────────────────────────────

Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ───────────────────────────────────────────────────────────
# PASSO 2: CARREGAR DOCUMENTOS
# ───────────────────────────────────────────────────────────

documents = SimpleDirectoryReader("./documents").load_data()
print(f"Documentos carregados: {len(documents)}")

# ───────────────────────────────────────────────────────────
# PASSO 3: CRIAR ÍNDICE VETORIAL
# ───────────────────────────────────────────────────────────

# Com persistência
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_data")
chroma_collection = chroma_client.get_or_create_collection("documents")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Criar índice
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

print("Índice criado com sucesso!")

# ───────────────────────────────────────────────────────────
# PASSO 4: CONFIGURAR RETRIEVER COM OPÇÕES AVANÇADAS
# ───────────────────────────────────────────────────────────

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,  # Top-5 resultados
)

# Adicionar post-processador para re-ranking
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)

# ───────────────────────────────────────────────────────────
# PASSO 5: CRIAR QUERY ENGINE COM PROMPT CUSTOMIZADO
# ───────────────────────────────────────────────────────────

from llama_index.core.prompts import PromptTemplate

qa_prompt_str = """Contexto das informações:
────────────────────────────────────────
{context_str}
────────────────────────────────────────

Pergunta: {query_str}

Instruções:
1. Responda EXCLUSIVAMENTE baseado no contexto
2. Se não souber, diga "Não encontrei informação"
3. Cite as fontes
4. Responda em 2-3 sentenças"""

qa_prompt = PromptTemplate(qa_prompt_str)

# Criar query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[postprocessor],
    text_qa_template=qa_prompt
)

# ───────────────────────────────────────────────────────────
# PASSO 6: EXECUTAR QUERIES
# ───────────────────────────────────────────────────────────

response = query_engine.query("Qual é a política de férias?")

print(f"Resposta: {response}")
print(f"\nFontes utilizadas:")
for node in response.source_nodes:
    print(f"  - {node.metadata.get('file_name', 'Unknown')}")

# ───────────────────────────────────────────────────────────
# PASSO 7: INTEGRAÇÃO COM LANGCHAIN (OPTIONAL)
# ───────────────────────────────────────────────────────────

from langchain.retrievers import LlamaIndexRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Converter retriever LlamaIndex para LangChain
langchain_retriever = LlamaIndexRetriever(retriever)

# Usar em LangChain chain
langchain_llm = ChatOpenAI(model="gpt-4")
langchain_chain = RetrievalQA.from_chain_type(
    llm=langchain_llm,
    chain_type="stuff",
    retriever=langchain_retriever
)

# Usar como antes
result = langchain_chain({"query": "Qual é a política de férias?"})
print(result['result'])
```

### 5.3 Exemplo Prático: Chatbot sobre PDFs

```python
# ═══════════════════════════════════════════════════════════
# CHATBOT RAG INTERATIVO SOBRE MÚLTIPLOS PDFS
# ═══════════════════════════════════════════════════════════

import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# ───────────────────────────────────────────────────────────
# UI COM STREAMLIT
# ───────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 Assistente de Documentos com RAG")

# Sidebar para upload
with st.sidebar:
    st.header("Upload de Documentos")
    uploaded_files = st.file_uploader(
        "Selecione PDFs",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    process_button = st.button("Processar Documentos")
    
    # Parâmetros
    chunk_size = st.slider("Tamanho do Chunk", 256, 1024, 512)
    k_results = st.slider("Top-K resultados", 1, 10, 5)

# ───────────────────────────────────────────────────────────
# PROCESSAMENTO DE ARQUIVOS
# ───────────────────────────────────────────────────────────

@st.cache_resource
def process_documents(uploaded_files_list, chunk_sz):
    """Process uploads e cria vector store (cached)"""
    
    documents = []
    
    for uploaded_file in uploaded_files_list:
        # Salvar temporário
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Carregar
        loader = PyPDFLoader(f"temp_{uploaded_file.name}")
        docs = loader.load()
        
        # Adicionar metadata
        for doc in docs:
            doc.metadata['source'] = uploaded_file.name
        
        documents.extend(docs)
    
    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_sz,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    # Vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        chunks,
        embeddings
    )
    
    # Limpeza
    for uploaded_file in uploaded_files_list:
        os.remove(f"temp_{uploaded_file.name}")
    
    return vector_store

# Processar ao clicar
if process_button and uploaded_files:
    with st.spinner("Processando documentos..."):
        vector_store = process_documents(uploaded_files, chunk_size)
    st.success("✅ Documentos processados!")
    st.session_state.vector_store = vector_store

# ───────────────────────────────────────────────────────────
# CHAT INTERFACE
# ───────────────────────────────────────────────────────────

if 'vector_store' in st.session_state:
    # Criar chain
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": k_results}
    )
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente útil. "
                   "Responda baseado no contexto fornecido."),
        ("human", "{context}\n\nPergunta: {question}")
    ])
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    if query := st.chat_input("Faça uma pergunta sobre os documentos..."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.spinner("Procurando informação..."):
            result = rag_chain({"query": query})
        
        response_text = result['result']
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text
        })
        
        with st.chat_message("assistant"):
            st.write(response_text)
            
            # Mostrar fontes
            with st.expander("📄 Ver fontes"):
                for doc in result['source_documents']:
                    st.write(f"**{doc.metadata.get('source', 'Unknown')}**")
                    st.write(doc.page_content[:500] + "...")

else:
    st.info("👈 Faça upload de PDFs no painel lateral para começar!")
```

---

## **MÓDULO 6: AVALIAÇÃO E MÉTRICAS**

### Objetivos de Aprendizado
- Compreender métricas de recuperação (Recall, Precision, MRR)
- Aprender métricas de geração (Factualidade, Relevância)
- Implementar frameworks de avaliação automatizados
- Construir pipelines de testes para RAG

### 6.1 Métricas de Recuperação

#### Recall@K (Relembrança)

**Definição:**
```
Recall@K = (Número de documentos relevantes no top-K) 
           / (Total de documentos relevantes no corpus)
```

Responde: "De TODOS os documentos relevantes que existem, 
quantos foram encontrados?"

**Exemplo:**
```
Corpus: 1000 documentos
Documentos relevantes TOTAIS: 50

Query retorna top-5 com 3 relevantes
Recall@5 = 3 / 50 = 0.06 (6%)

Mesma query, top-100 com 30 relevantes
Recall@100 = 30 / 50 = 0.60 (60%)

→ Alto recall@100 mas baixo recall@5
→ Retriever encontra coisas, mas precisa buscar muito
```

**Quando Usar:**
- ✅ Importa encontrar TUDO (busca legal, análise completa)
- ❌ Não importa quando tempo é crítico (latência)

#### Precision@K (Precisão)

**Definição:**
```
Precision@K = (Número de documentos relevantes no top-K)
              / (Total de documentos retornados no top-K)
```

Responda: "Dos documentos que retornamos, quantos são realmente relevantes?"

**Exemplo:**
```
Retorno top-5 documentos
Dos 5, 3 são relevantes
Precision@5 = 3 / 5 = 0.60 (60%)

→ 60% dos meus retornos são bons
→ Menos falsos positivos
```

**Quando Usar:**
- ✅ Importa evitar ruído (buscas corporativas, suporte)
- ❌ Não importa omitir alguns resultados

#### Mean Reciprocal Rank (MRR)

**Definição:**
```
MRR = (1/N) * Σ (1 / rank_i)

Onde rank_i é a posição do PRIMEIRO documento relevante
para query i
```

Responde: "Em média, em que posição o primeiro resultado 
relevante aparece?"

**Exemplo:**
```
Query 1: Primeiro relevante na posição 2 → 1/2 = 0.5
Query 2: Primeiro relevante na posição 1 → 1/1 = 1.0
Query 3: Nenhum relevante (rank=∞) → 1/∞ ≈ 0

MRR = (0.5 + 1.0 + 0) / 3 = 0.5

→ Em média, primeira coisa relevante é achada na posição 2
→ Boa para cenários onde SÓ A PRIMEIRA resposta importa
```

**Quando Usar:**
- ✅ Informação-seeking (usuário quer resposta rápida)
- ✅ QA systems

#### NDCG (Normalized Discounted Cumulative Gain)

**Conceito:**
```
NDCG mede qualidade do RANKING, não só presença/ausência
Documents podem ser "relevantes" em graus (0-5 stars)

Fórmula:
DCG@K = Σ (rel_i / log2(i+1))

Onde rel_i é relevância do documento na posição i

NDCG@K = DCG@K / IDCG@K
(Normalizado pelo melhor ranking possível)
```

**Exemplo:**
```
Ideal ranking: [5, 5, 4, 3, 2] (melhor possível)
IDCG = 5/log(2) + 5/log(3) + 4/log(4) + 3/log(5) + 2/log(6)
     ≈ 5 + 3.15 + 2 + 1.29 + 0.73 = 12.17

Meu retriever retorna: [5, 3, 4, 5, 1]
DCG = 5/log(2) + 3/log(3) + 4/log(4) + 5/log(5) + 1/log(6)
    ≈ 5 + 1.89 + 2 + 2.15 + 0.15 = 11.19

NDCG@5 = 11.19 / 12.17 = 0.92 (92%)
```

#### Tabela Comparativa

|      Métrica    |          O que mede      |    Valor Ideal  |        Cenário        |
|-----------------|--------------------------|-----------------|-----------------------|
| **Precision@5** | Pureza dos top-5         | 1.0 (100%)      | Buscas corporativas   |
| **Recall@10**   | Cobertura dos relevantes | 1.0 (100%)      | Análise legal         |
| **MRR**         | Rank do primeiro bom     | 1.0 (posição 1) | Google-like search    |
| **NDCG@10**     | Qualidade do ranking     | 1.0 (perfeito)  | Mecanismos de ranking |

### 6.2 Métricas de Geração

#### Faithfulness (Factualidade)

**Definição:**
Mede se a resposta gerada é suportada pelo contexto recuperado.

**Implementação com LLM-based Metric:**

```python
from ragas.metrics import Faithfulness
from datasets import Dataset

# Dataset estruturado
eval_dataset = Dataset.from_dict({
    "question": ["O que é IA?", "Quando foi fundada?"],
    "contexts": [
        [["IA é um campo da computação..."]],
        [["A empresa foi fundada em 2020..."]]
    ],
    "answer": [
        "IA (Inteligência Artificial) é um ramo da computação.",
        "A empresa foi fundada em 2020."
    ]
})

# Métrica
faithfulness = Faithfulness()

# Avaliar
scores = faithfulness.score(eval_dataset)
print(f"Faithfulness: {scores['faithfulness']}")  # 0-1
```

**Método Manual:**
```python
def evaluate_faithfulness(
    answer: str,
    context: str,
    llm
) -> float:
    """
    Score 0-1 de quão factual a resposta é.
    """
    
    prompt = f"""
    Dado o CONTEXTO abaixo:
    
    CONTEXTO:
    {context}
    
    A seguinte RESPOSTA é suportada pelos fatos no contexto?
    
    RESPOSTA:
    {answer}
    
    Responda como JSON:
    {{"suportada": true/false, "score": 0-1, "razao": "..."}}
    """
    
    # LLM avalia
    eval_response = llm.generate(prompt)
    
    import json
    result = json.loads(eval_response)
    
    return result["score"]
```

#### Answer Relevance

**Definição:**
Mede se a resposta responde à pergunta original.

```python
from ragas.metrics import AnswerRelevancy

# Usar RAGAS framework (recomendado)
relevancy = AnswerRelevancy()

# Se a pergunta era "Qual é a capital?"
# E resposta é "A capital é Paris."
# Answer Relevance seria high (1.0)

# Se resposta fosse "Paris fica na Europa"
# Answer Relevance seria lower (pode ser 0.7)

scores = relevancy.score(eval_dataset)
```

#### Answer Correctness (Acurácia)

**Definição:**
Compara resposta gerada com resposta esperada (ground truth).

```python
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def evaluate_answer_correctness(
    generated_answer: str,
    ground_truth_answer: str,
    metric: str = "rouge"
) -> float:
    """
    Compara resposta gerada com esperada.
    """
    
    if metric == "rouge":
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth_answer, generated_answer)
        return scores['rougeL'].fmeasure
    
    elif metric == "bert":
        # BERTScore usa similaridade semântica
        precision, recall, f1 = bert_score(
            [generated_answer],
            [ground_truth_answer],
            lang="en"
        )
        return f1[0].item()  # F1 score
    
    elif metric == "exact_match":
        return 1.0 if generated_answer.lower() == ground_truth_answer.lower() else 0.0

# Exemplo:
accuracy = evaluate_answer_correctness(
    "A capital da França é Paris",
    "Paris é a capital da França"
)
print(f"Accuracy: {accuracy}")  # ~ 0.95
```

### 6.3 Framework de Avaliação Completo

```python
# ═══════════════════════════════════════════════════════════
# PIPELINE DE AVALIAÇÃO AUTOMATIZADO PARA RAG
# ═══════════════════════════════════════════════════════════

from typing import List, Dict
import json
from datetime import datetime
from tqdm import tqdm

# ───────────────────────────────────────────────────────────
# 1. DEFINIR DATASET DE AVALIAÇÃO
# ───────────────────────────────────────────────────────────

class EvalDataset:
    """Dataset estruturado para avaliação."""
    
    def __init__(self):
        self.samples = []
    
    def add_sample(self, question: str, expected_answer: str, 
                   relevant_docs: List[str], tags: List[str] = None):
        """Adicionar sample de teste."""
        self.samples.append({
            "question": question,
            "expected_answer": expected_answer,
            "relevant_docs": relevant_docs,
            "tags": tags or []
        })
    
    def load_from_json(self, filepath: str):
        """Carregar dataset de arquivo JSON."""
        with open(filepath) as f:
            self.samples = json.load(f)
    
    def save_to_json(self, filepath: str):
        """Salvar dataset."""
        with open(filepath, 'w') as f:
            json.dump(self.samples, f, indent=2)

# Exemplo de dataset
eval_dataset = EvalDataset()
eval_dataset.add_sample(
    question="Qual é a política de férias?",
    expected_answer="20 dias úteis por ano",
    relevant_docs=["Capítulo 3 - Benefícios", "Seção 3.2 - Férias"],
    tags=["benefits", "hr_policy"]
)

# ───────────────────────────────────────────────────────────
# 2. IMPLEMENTAR MÉTRICAS
# ───────────────────────────────────────────────────────────

class RAGEvaluator:
    """Classe para avaliar sistema RAG."""
    
    def __init__(self, rag_chain, eval_llm):
        self.rag_chain = rag_chain
        self.eval_llm = eval_llm
    
    def eval_retrieval_metrics(self, 
                              retrieved_docs: List[str],
                              relevant_docs: List[str]) -> Dict:
        """Calcular métricas de retrieval."""
        
        # Precision@K
        k = len(retrieved_docs)
        relevant_retrieved = sum(1 for doc in retrieved_docs 
                                if doc in relevant_docs)
        precision = relevant_retrieved / k if k > 0 else 0
        
        # Recall@K
        recall = relevant_retrieved / len(relevant_docs) \
            if len(relevant_docs) > 0 else 0
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved_count": k,
            "relevant_count": len(relevant_docs)
        }
    
    def eval_generation_metrics(self,
                               generated_answer: str,
                               expected_answer: str,
                               context: str) -> Dict:
        """Calcular métricas de generation."""
        
        # Faithfulness (LLM-based)
        faithful_prompt = f"""
        Contexto: {context}
        
        Resposta: {generated_answer}
        
        A resposta é suportada pelo contexto? (0-1)
        Responda apenas com um número.
        """
        
        faithfulness_response = self.eval_llm.generate(faithful_prompt)
        faithfulness = float(faithfulness_response) / 100 \
            if "%" in faithfulness_response else float(faithfulness_response)
        
        # Answer Relevance (LLM-based)
        relevance_prompt = f"""
        Pergunta: {expected_answer}
        
        Resposta: {generated_answer}
        
        A resposta aborda a pergunta? (0-1)
        """
        
        relevance_response = self.eval_llm.generate(relevance_prompt)
        relevance = float(relevance_response) / 100 \
            if "%" in relevance_response else float(relevance_response)
        
        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "avg_generation_score": (faithfulness + relevance) / 2
        }
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """Avaliar um sample completo."""
        
        question = sample["question"]
        expected = sample["expected_answer"]
        relevant_docs_info = sample["relevant_docs"]
        
        # Executar RAG
        result = self.rag_chain({"query": question})
        generated_answer = result["result"]
        retrieved_docs = [doc.page_content for doc 
                         in result.get("source_documents", [])]
        context = "\n".join(retrieved_docs)
        
        # Avaliar retrieval
        retrieval_metrics = self.eval_retrieval_metrics(
            retrieved_docs,
            relevant_docs_info
        )
        
        # Avaliar generation
        generation_metrics = self.eval_generation_metrics(
            generated_answer,
            expected,
            context
        )
        
        return {
            "question": question,
            "generated_answer": generated_answer,
            "expected_answer": expected,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "combined_score": (
                retrieval_metrics["f1"] * 0.4 +
                generation_metrics["avg_generation_score"] * 0.6
            )
        }
    
    def evaluate_dataset(self, eval_dataset: EvalDataset) -> Dict:
        """Avaliar dataset inteiro."""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(eval_dataset.samples),
            "samples": [],
            "aggregated_metrics": {}
        }
        
        # Avaliar cada sample
        for sample in tqdm(eval_dataset.samples, desc="Avaliando"):
            result = self.evaluate_sample(sample)
            results["samples"].append(result)
        
        # Agregar métricas
        results["aggregated_metrics"] = self._aggregate_results(results["samples"])
        
        return results
    
    def _aggregate_results(self, samples: List[Dict]) -> Dict:
        """Agregrar resultados de múltiplos samples."""
        
        avg_precision = sum(s["retrieval_metrics"]["precision"] 
                           for s in samples) / len(samples)
        avg_recall = sum(s["retrieval_metrics"]["recall"] 
                        for s in samples) / len(samples)
        avg_f1 = sum(s["retrieval_metrics"]["f1"] 
                    for s in samples) / len(samples)
        
        avg_faithfulness = sum(s["generation_metrics"]["faithfulness"] 
                              for s in samples) / len(samples)
        avg_relevance = sum(s["generation_metrics"]["relevance"] 
                           for s in samples) / len(samples)
        
        avg_combined = sum(s["combined_score"] 
                          for s in samples) / len(samples)
        
        return {
            "retrieval": {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1
            },
            "generation": {
                "avg_faithfulness": avg_faithfulness,
                "avg_relevance": avg_relevance
            },
            "overall_score": avg_combined
        }

# ───────────────────────────────────────────────────────────
# 3. USAR O EVALUADOR
# ───────────────────────────────────────────────────────────

from langchain_openai import ChatOpenAI

evaluator = RAGEvaluator(rag_chain=rag_chain, 
                         eval_llm=ChatOpenAI(model="gpt-4"))

# Avaliar
results = evaluator.evaluate_dataset(eval_dataset)

# Salvar resultados
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Imprimir resumo
print(f"\n{'='*50}")
print(f"EVALUATION RESULTS")
print(f"{'='*50}")
print(f"\nSamples avaliados: {results['total_samples']}")
print(f"\nRetrievalMetrics:")
for metric, value in results["aggregated_metrics"]["retrieval"].items():
    print(f"  {metric}: {value:.3f}")

print(f"\nGeneration Metrics:")
for metric, value in results["aggregated_metrics"]["generation"].items():
    print(f"  {metric}: {value:.3f}")

print(f"\nOverall Score: {results['aggregated_metrics']['overall_score']:.3f}")
```

---

## **MÓDULO 7: CASOS DE USO E MELHORES PRÁTICAS**

### Objetivos de Aprendizado
- Explorar aplicações reais de RAG em diversos setores
- Aprender padrões de deployment em produção
- Implementar manutenção e atualização de índices
- Comparar RAG com fine-tuning e LoRA

### 7.1 Aplicações Reais em Empresas

#### Case Study 1: Busca Corporativa (Enterprise Search)

```
┌───────────────────────────────────────┐
│   PROBLEMA EMPRESARIAL                │
├───────────────────────────────────────┤
│ • 10.000+ documentos (PDFs, Wikis)    │
│ • Funcionários perdem tempo buscando  │
│ • Informação desatualizada            │
│ • Difícil encontrar contexto          │
└───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│   SOLUÇÃO RAG                         │
├───────────────────────────────────────┤
│ 1. Indexar TODOS os docs corporativos │
│    (intranets, políticas, manuais)    │
│                                       │
│ 2. Retriever: Busca semântica         │
│    → "Como solicitar férias?"         │
│    ← Retorna seções relevantes        │
│                                       │
│ 3. Generator: LLM gera resposta       │
│    → "Você pode solicitar por..."     │
│    ← Com citation exata               │
│                                       │
│ 4. Update automático: Novos docs      │
│    → Sync com Google Drive/Sharepoint │
│                                       │
└───────────────────────────────────────┘

BENEFÍCIOS MENSURADOS:
• Redução de 40% em tempo de busca
• 30% menos suporte tickets
• 95% satisfação do usuário
```

#### Case Study 2: Suporte Técnico Automatizado

```
PIPELINE:
Ticket do cliente 
    ↓
Query: "Produto não liga"
    ↓
RAG Retrieves:
  • FAQ #123: "Verificar bateria"
  • Troubleshooting Guide
  • Common Issues Database
    ↓
LLM Gera resposta:
  "Obrigado por entrar em contato.
   Por favor, verifique:
   1. Bateria carregada (FAQ #123)
   2. Botão liga/desliga (TG-45)
   ..."
    ↓
Resposta enviada em <30s (vs. 2h manual)

ROI: 60% redução de custos operacionais
```

#### Case Study 3: Análise de Contratos Legais

```
CENÁRIO:
Departamento legal tem 1000+ contratos
Precisa encontrar cláusulas específicas rapidamente

IMPLEMENTAÇÃO RAG:
├─ Chunking semântico (por cláusula)
├─ Metadados: cliente, data, tipo contrato
├─ Retriever: Dense + Sparse (Hybrid)
├─ LLM: Especializado em análise legal
│   (Fine-tuned ou prompt engenheirado)
└─ Output: Extração estruturada (JSON)

EXEMPLO QUERY:
"Encontre todas as cláusulas de indenização 
com limite > $1M desde 2022"

RAG RESPOSTA:
{
  "matching_clauses": [
    {
      "contract": "ACC-2023-0451.pdf",
      "clause": "Section 4.2 - Indemnification",
      "excerpt": "...",
      "limit": "$2.5M",
      "effective_date": "2023-01-15"
    },
    ...
  ],
  "total_matches": 47
}

IMPACTO: Aceleração 10x em análise contratual
```

#### Case Study 4: Healthcare - Assistência Diagnóstica

```
SISTEMA: Assistente de Diagnóstico com RAG

KNOWLEDGE BASE:
├─ Literatura médica (PubMed papers)
├─ Protocolos clínicos
├─ Histórico de pacientes (anonymized)
├─ Guidelines de tratamento
└─ Estudos de caso

WORKFLOW:
1. Médico: "Paciente com febre 39°C + tosse"
   
2. RAG Retrieves:
   • Papers sobre infecções respiratórias
   • Protocolos de avaliação
   • Históricos similares
   
3. LLM Gera análise:
   "Baseado na literatura:
    - Considerar pneumonia viral/bacteriana
    - Recomenda-se teste COVID-19 (Guideline X)
    - Se prescrever antibiótico, considerar..."
    
4. Citations permitem verificação
   médica

COMPLIANCE:
✓ HIPAA compliance (dados locais)
✓ Rastreabilidade (citations)
✓ Sem diagnosis automática (auxilia MD)
```

### 7.2 Estratégias de Deployment em Produção

```python
# ═══════════════════════════════════════════════════════════
# DEPLOYMENT PRODUCTION-READY DE RAG
# ═══════════════════════════════════════════════════════════

import logging
from typing import Optional
import asyncio
from datetime import datetime
import redis

# ───────────────────────────────────────────────────────────
# 1. LOGGING E MONITORING
# ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGMetricsCollector:
    """Coleta métricas de performance em produção."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def log_query(self, query_id: str, query_text: str, 
                 response_time: float, num_docs_retrieved: int,
                 model_used: str):
        """Registrar query e métricas associadas."""
        
        metrics = {
            "query_id": query_id,
            "query_text": query_text,
            "response_time_ms": response_time,
            "docs_retrieved": num_docs_retrieved,
            "model": model_used,
            "timestamp": datetime.now().isoformat()
        }
        
        # Salvar em Redis para análise
        self.redis.rpush("rag_metrics", str(metrics))
        logger.info(f"Query logged: {query_id}, Time: {response_time}ms")

# ───────────────────────────────────────────────────────────
# 2. CACHING PARA PERFORMANCE
# ───────────────────────────────────────────────────────────

class CachedRAG:
    """RAG com caching de queries comuns."""
    
    def __init__(self, rag_chain, cache_ttl_seconds: int = 3600):
        self.rag_chain = rag_chain
        self.cache_ttl = cache_ttl_seconds
        self.cache = redis.Redis(host='localhost', port=6379)
    
    def query(self, question: str) -> str:
        """Query com cache."""
        
        # Normalizar para cache
        cache_key = f"rag:{question.lower().strip()}"
        
        # Checar cache
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit for: {question}")
            return cached.decode()
        
        # Cache miss - execute RAG
        logger.info(f"Cache miss for: {question}")
        result = self.rag_chain({"query": question})
        answer = result["result"]
        
        # Salvar no cache
        self.cache.setex(cache_key, self.cache_ttl, answer)
        
        return answer

# ───────────────────────────────────────────────────────────
# 3. RETRY LOGIC E ERROR HANDLING
# ───────────────────────────────────────────────────────────

import time
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustRAG:
    """RAG com retry automático e tratamento de erro."""
    
    def __init__(self, rag_chain, max_retries: int = 3):
        self.rag_chain = rag_chain
        self.max_retries = max_retries
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _execute_rag(self, question: str):
        """Execute com retry automático."""
        return self.rag_chain({"query": question})
    
    def query(self, question: str) -> Optional[dict]:
        """Query robusta."""
        try:
            result = self._execute_rag(question)
            logger.info(f"Query successful: {question[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Query failed after {self.max_retries} retries: {e}")
            return {
                "result": "Desculpe, não foi possível processar sua pergunta. "
                         "Por favor, tente novamente mais tarde.",
                "error": str(e)
            }

# ───────────────────────────────────────────────────────────
# 4. AUTO-REFRESH DE ÍNDICE
# ───────────────────────────────────────────────────────────

import schedule
from pathlib import Path

class AutoRefreshingRAG:
    """RAG com índice que se atualiza automaticamente."""
    
    def __init__(self, rag_chain, documents_directory: str,
                 refresh_interval_hours: int = 24):
        self.rag_chain = rag_chain
        self.docs_dir = Path(documents_directory)
        self.refresh_interval = refresh_interval_hours
        self.last_refresh = None
    
    def check_for_new_documents(self) -> list:
        """Verificar novos documentos."""
        
        new_docs = []
        current_time = datetime.now()
        
        for doc_file in self.docs_dir.glob("*.pdf"):
            mod_time = datetime.fromtimestamp(doc_file.stat().st_mtime)
            
            if self.last_refresh is None or mod_time > self.last_refresh:
                new_docs.append(doc_file)
        
        return new_docs
    
    async def auto_refresh(self):
        """Refresh periódico do índice."""
        
        while True:
            await asyncio.sleep(self.refresh_interval * 3600)
            
            new_docs = self.check_for_new_documents()
            
            if new_docs:
                logger.info(f"Refreshing index with {len(new_docs)} new docs")
                self._refresh_index(new_docs)
                self.last_refresh = datetime.now()
    
    def _refresh_index(self, new_docs: list):
        """Atualizar índice com novos documentos."""
        # Implementar lógica de refresh específica
        pass

# ───────────────────────────────────────────────────────────
# 5. RATE LIMITING E QUOTAS
# ───────────────────────────────────────────────────────────

from functools import wraps

class RateLimitedRAG:
    """RAG com rate limiting por user."""
    
    def __init__(self, rag_chain, 
                 queries_per_minute: int = 30):
        self.rag_chain = rag_chain
        self.qpm = queries_per_minute
        self.query_history = {}  # user_id -> [timestamps]
    
    def is_allowed(self, user_id: str) -> bool:
        """Verificar se user atingiu limite."""
        
        now = time.time()
        one_minute_ago = now - 60
        
        # Limpar queries antigas
        if user_id not in self.query_history:
            self.query_history[user_id] = []
        
        recent_queries = [ts for ts in self.query_history[user_id]
                         if ts > one_minute_ago]
        self.query_history[user_id] = recent_queries
        
        # Checar limite
        return len(recent_queries) < self.qpm
    
    def query(self, user_id: str, question: str) -> dict:
        """Query com rate limit."""
        
        if not self.is_allowed(user_id):
            logger.warning(f"Rate limit exceeded for user: {user_id}")
            return {
                "error": "Rate limit exceeded. "
                        f"Máximo {self.qpm} queries por minuto.",
                "retry_after_seconds": 60
            }
        
        # Log do timestamp
        self.query_history[user_id].append(time.time())
        
        # Execute
        return self.rag_chain({"query": question})
```

### 7.3 Manutenção do Índice

#### Estratégia de Versionamento

```python
class VersionedVectorStore:
    """Vector store com versionamento."""
    
    def __init__(self, base_path: str = "./vector_stores"):
        self.base_path = Path(base_path)
        self.current_version = self._get_latest_version()
    
    def _get_latest_version(self) -> int:
        """Encontrar última versão."""
        versions = [
            int(d.name.split('_')[1])
            for d in self.base_path.glob("store_*")
        ]
        return max(versions) if versions else 0
    
    def create_new_version(self, documents: list) -> str:
        """Criar nova versão do store."""
        
        new_version = self.current_version + 1
        version_path = self.base_path / f"store_v{new_version}"
        version_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating new vector store version: {new_version}")
        
        # Criar store
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(version_path)
        )
        
        # Registrar versão
        metadata = {
            "version": new_version,
            "created_at": datetime.now().isoformat(),
            "num_docs": len(documents),
            "status": "active"
        }
        
        with open(version_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        self.current_version = new_version
        return str(version_path)
    
    def rollback_to_version(self, version: int):
        """Voltar a versão anterior."""
        
        version_path = self.base_path / f"store_v{version}"
        
        if not version_path.exists():
            raise ValueError(f"Version {version} not found")
        
        logger.info(f"Rolling back to version: {version}")
        self.current_version = version
```

### 7.4 RAG vs Fine-tuning vs LoRA

#### Comparação Detalhada

```
┌──────────────────────────────────────────────────────────────┐
│              RAG vs Fine-Tuning vs LoRA                      │
├─────────────────┬──────────────────┬─────────────────────────┤
│ Aspecto         │ RAG              │ Fine-Tuning   │ LoRA    │
├─────────────────┼──────────────────┼─────────────────────────┤
│ CONCEITO        │ Retrieve + Gen   │ Train params  │ Train   │
│                 │                  │ completos     │ subset  │
├─────────────────┼──────────────────┼─────────────────────────┤
│ Custo Treino    │ $0 (indexação)   │ $$$ alto      │ $ baixo │
│                 │                  │               │         │
│ Tempo Treino    │ Horas (index)    │ Dias/Semanas  │ Horas   │
│                 │                  │               │         │
│ Memória Treino  │ GPU 24GB         │ GPU 40-80GB   │ GPU 16GB│
│                 │                  │               │         │
│ Latência        │ ~500ms (busca)   │ <100ms        │ <100ms  │
│                 │                  │               │         │
│ Atualização     │ Trivial (add doc)│ Retreinar     │ Retrein │
│                 │                  │               │ (rápido)│
│                 │                  │               │         │
│ Conhecimento    │ Dinâmico (sempre │ Estático      │ Estátic │
│                 │ atualizado)      │ (fixed)       │ o       │
│                 │                  │               │         │
│ Interpretab.    │ Alto (cita doc)  │ Baixa (black  │ Baixa   │
│                 │                  │ box)          │         │
│                 │                  │               │         │
│ Rastreab.       │ Sim (fonte clara)│ Não           │ Não     │
│                 │                  │               │         │
│ Privacidade     │ Sim (dados local)│ Depende       │ Depende │
│                 │                  │               │         │
│ Casos de Uso    │ Docs, QA,        │ Tarefas       │ Adapt.  │
│                 │ lookup, search   │ específicas   │ rápida  │
└─────────────────┴──────────────────┴─────────────────────────┘
```

#### Guia de Decisão: Qual Usar?

```
┌─ Dados atualizados frequentemente?
│  ├─ SIM → RAG (fácil update)
│  └─ NÃO → Considerar Fine-Tuning
│
├─ Precisa citar fontes?
│  ├─ SIM → RAG (transparência)
│  └─ NÃO → Fine-Tuning/LoRA
│
├─ Dados sensíveis/propriet ários?
│  ├─ SIM → RAG (dados locais)
│  └─ NÃO → Qualquer um
│
├─ Budget limitado?
│  ├─ SIM → LoRA (<$1000)
│  └─ NÃO → Fine-Tuning completo
│
├─ Latência crítica (<100ms)?
│  ├─ SIM → Fine-Tuning/LoRA (sem retrieval)
│  └─ NÃO → RAG OK (~500ms)
│
└─ Tarefas variadas/multi-domínio?
   ├─ SIM → RAG (adaptável)
   └─ NÃO → Fine-Tuning especializado
```

#### Implementação Prática: Combinando Abordagens

```python
# ═══════════════════════════════════════════════════════════
# HYBRID: RAG + Fine-Tuned Generator
# ═══════════════════════════════════════════════════════════

class HybridRAG:
    """
    Combina RAG com generator fine-tuned.
    
    Usa RAG para retrieval + modelo specializado para generation.
    Melhor de dois mundos: conhecimento dinâmico + precisão.
    """
    
    def __init__(self, 
                 vector_store,
                 finetuned_llm,  # LLM specializado
                 domain: str = "legal"):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        self.llm = finetuned_llm
        self.domain = domain
    
    def query(self, question: str) -> dict:
        """Query hybrid."""
        
        # 1. Retrieve com RAG
        context_docs = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # 2. Generate com modelo fine-tuned
        # O LLM já foi treinado para entender o domínio
        prompt = self._build_domain_prompt(question, context)
        
        answer = self.llm.generate(prompt)
        
        return {
            "answer": answer,
            "sources": [doc.metadata for doc in context_docs],
            "model_type": "hybrid_rag"
        }
    
    def _build_domain_prompt(self, question: str, context: str) -> str:
        """Prompt específico do domínio."""
        
        if self.domain == "legal":
            return f"""ANÁLISE LEGAL
Contexto de contratos:
{context}

Pergunta: {question}

Analise conforme direito contratual. Cite cláusulas relevantes."""
        
        elif self.domain == "medical":
            return f"""CONSULT A MÉDICA
Contexto de literatura:
{context}

Paciente: {question}

Baseado em evidências e guidelines."""
        
        else:
            return f"Contexto:\n{context}\n\nPergunta: {question}"
```

---

## **MÓDULO 8: TÉCNICAS AVANÇADAS**

### Objetivos de Aprendizado
- Implementar RAG multimodal (texto + imagem)
- Entender RAG híbrido (multiple retrieval backends)
- Explorar retriever baseados em modelos avançados (ColBERT, Contriever)
- Aplicar compressão de contexto e técnicas de seleção

### 8.1 RAG Multimodal

```python
# ═══════════════════════════════════════════════════════════
# MULTIMODAL RAG: Texto + Imagens
# ═══════════════════════════════════════════════════════════

from langchain.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_community.document_loaders.pdf import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PIL import Image
import base64
from io import BytesIO

class MultimodalDocumentProcessor:
    """Processa documentos com texto e imagens."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
    
    def extract_text_and_images_from_pdf(self, pdf_path: str) -> list:
        """Extrair texto e imagens de PDF."""
        
        documents = []
        images = []
        
        # Usar PyPDFium2 que mantém imagens
        loader = PyPDFium2Loader(pdf_path)
        text_docs = loader.load()
        
        # Extrair imagens de PDFs
        import fitz  # PyMuPDF
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Texto
            text_content = page.get_text()
            if text_content.strip():
                documents.append(Document(
                    page_content=text_content,
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "type": "text"
                    }
                ))
            
            # Imagens
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                pix = fitz.Pixmap(pdf_document, xref)
                
                # Converter para base64
                img_data = pix.tobytes("png")
                img_b64 = base64.b64encode(img_data).decode()
                
                documents.append(Document(
                    page_content=f"[IMAGE: {img_index}]",
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "type": "image",
                        "image_base64": img_b64
                    }
                ))
        
        return documents

class MultimodalRetriever:
    """Retriever que processa texto e imagem."""
    
    def __init__(self, vector_store, vision_llm):
        self.vector_store = vector_store
        self.vision_llm = vision_llm  # Claude-3 ou GPT-4V
    
    def retrieve_and_describe_images(self, query: str, k: int = 5):
        """Retrieve documentos e descrever imagens relevantes."""
        
        # Retrieve padrão (texto)
        text_docs = self.vector_store.similarity_search(query, k=k)
        
        # Se houver imagens, descrever com vision model
        results = []
        
        for doc in text_docs:
            if doc.metadata.get("type") == "image":
                # Descrever imagem
                image_b64 = doc.metadata.get("image_base64")
                
                description = self.vision_llm.generate(
                    f"Descreva esta imagem em relação a: {query}",
                    image=image_b64
                )
                
                results.append({
                    "content": f"[Imagem] {description}",
                    "source": doc.metadata["source"],
                    "type": "image_with_description"
                })
            else:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata["source"],
                    "type": "text"
                })
        
        return results

class MultimodalRAGChain:
    """Chain RAG multimodal completo."""
    
    def __init__(self, vector_store, text_llm, vision_llm):
        self.retriever = MultimodalRetriever(vector_store, vision_llm)
        self.text_llm = text_llm
    
    def query(self, question: str) -> str:
        """Query multimodal."""
        
        # Retrieve
        retrieved = self.retriever.retrieve_and_describe_images(question)
        
        # Construir contexto
        context_parts = []
        for item in retrieved:
            if item["type"] == "image_with_description":
                context_parts.append(f"[IMAGEM RELEVANTE]\n{item['content']}")
            else:
                context_parts.append(f"[TEXTO]\n{item['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate
        prompt = f"""
        Pergunta: {question}
        
        Contexto (texto e imagens):
        {context}
        
        Responda usando texto e imagens quando relevante.
        """
        
        answer = self.text_llm.generate(prompt)
        
        return answer
```

### 8.2 RAG Híbrido (Multiple Backends)

```python
# ═══════════════════════════════════════════════════════════
# HYBRID RAG: Dense + Sparse + Knowledge Graph
# ═══════════════════════════════════════════════════════════

from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple

class HybridRetriever:
    """Combina Dense (embeddings) + Sparse (BM25) + Graph retrieval."""
    
    def __init__(self, 
                 vector_store,
                 documents: List[str],
                 knowledge_graph = None,
                 weights: dict = None):
        
        self.vector_store = vector_store
        self.documents = documents
        self.kg = knowledge_graph
        
        # Pesos: quanto cada backend contribui
        self.weights = weights or {
            "dense": 0.5,
            "sparse": 0.3,
            "graph": 0.2
        }
        
        # Inicializar BM25
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def retrieve_dense(self, query: str, k: int = 5) -> List[Tuple]:
        """Dense retrieval (embeddings)."""
        
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Normalizar scores [0, 1]
        scores = np.array([score for _, score in docs])
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return [(doc.page_content, float(score)) for doc, score in zip(docs, normalized_scores)]
    
    def retrieve_sparse(self, query: str, k: int = 5) -> List[Tuple]:
        """Sparse retrieval (BM25 - keyword)."""
        
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Top-K
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # Normalizar scores
        normalized_scores = scores[top_indices] / (scores[top_indices].max() + 1e-8)
        
        return [
            (self.documents[i], float(normalized_scores[j]))
            for j, i in enumerate(top_indices)
        ]
    
    def retrieve_from_graph(self, query: str, k: int = 5) -> List[Tuple]:
        """Knowledge Graph retrieval (se disponível)."""
        
        if not self.kg:
            return []
        
        # Buscar entidades mencionadas na query
        entities = self.kg.extract_entities(query)
        
        # Buscar documentos relacionados a essas entidades
        related_docs = []
        for entity in entities:
            related = self.kg.find_documents_by_entity(entity, k=k)
            related_docs.extend(related)
        
        # Remover duplicatas, ranking por freq
        from collections import Counter
        doc_freq = Counter([doc for doc, _ in related_docs])
        
        # Normalizar
        max_freq = max(doc_freq.values()) if doc_freq else 1
        normalized = [
            (doc, float(count / max_freq))
            for doc, count in doc_freq.most_common(k)
        ]
        
        return normalized
    
    def retrieve_hybrid(self, query: str, k: int = 5) -> List[Tuple]:
        """Combinar todos os retrievers."""
        
        # Retrieve com cada backend
        dense_results = self.retrieve_dense(query, k=k)
        sparse_results = self.retrieve_sparse(query, k=k)
        graph_results = self.retrieve_from_graph(query, k=k)
        
        # Agregar scores ponderados
        combined_scores = {}
        
        for doc, score in dense_results:
            combined_scores[doc] = combined_scores.get(doc, 0) + \
                self.weights["dense"] * score
        
        for doc, score in sparse_results:
            combined_scores[doc] = combined_scores.get(doc, 0) + \
                self.weights["sparse"] * score
        
        for doc, score in graph_results:
            combined_scores[doc] = combined_scores.get(doc, 0) + \
                self.weights["graph"] * score
        
        # Sort e retornar top-K
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return sorted_results
```

### 8.3 Retrievers Avançados: ColBERT e Contriever

```python
# ═══════════════════════════════════════════════════════════
# ADVANCED RETRIEVERS: ColBERT, Contriever
# ═══════════════════════════════════════════════════════════

class ColBERTRetriever:
    """
    ColBERT: Late Interaction Retrieval
    
    Vantagem: Compara embeddings TOKEN-LEVEL entre query e doc
    Resultado: Mais preciso que dense retrieval padrão
    Desvantagem: Mais custoso computacionalmente
    """
    
    def __init__(self, checkpoint: str = "colbert-ir/colbertv2.0"):
        from colbert.infra import ColBERTConfig
        from colbert.modeling.colbert import ColBERT as ColBERTModel
        
        config = ColBERTConfig(
            do_answer_search=False,
            root="/tmp/colbert"
        )
        
        self.model = ColBERTModel(checkpoint=checkpoint, config=config)
    
    def encode_documents(self, documents: List[str]):
        """Pré-computar embeddings de documentos."""
        
        self.doc_embeddings = []
        for doc in documents:
            # ColBERT retorna matriz [num_tokens, hidden_dim]
            embeddings = self.model.encode(doc)
            self.doc_embeddings.append(embeddings)
    
    def retrieve(self, query: str, k: int = 5):
        """Retrieve usando ColBERT."""
        
        # Encode query (também matriz de tokens)
        query_embeddings = self.model.encode(query)  # [Q, hidden_dim]
        
        # Compute similarity scores (late interaction)
        scores = []
        
        for doc_emb in self.doc_embeddings:
            # Query: [Q_tokens, hidden]
            # Doc: [D_tokens, hidden]
            # Score = max over doc tokens of (max similarity to query tokens)
            
            similarity_matrix = query_embeddings @ doc_emb.T  # [Q, D]
            max_sim_per_query_token = similarity_matrix.max(dim=1)[0]  # [Q]
            score = max_sim_per_query_token.mean()  # Média
            
            scores.append(score)
        
        # Top-K
        top_indices = np.argsort(scores)[-k:][::-1]
        
        return [(self.documents[i], scores[i]) for i in top_indices]


class ContrieverRetriever:
    """
    Contriever: Contrastive Learning Dense Retriever
    
    Treinado com contrastive learning em dados não-supervisionados
    Ótimo para zero-shot retrieval sem fine-tuning
    """
    
    def __init__(self, model_name: str = "facebook/contriever"):
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name)
    
    def retrieve(self, query: str, documents: List[str], k: int = 5):
        """Retrieve com Contriever."""
        
        # Encode query e docs
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        doc_embeddings = self.model.encode(documents, convert_to_tensor=True)
        
        # Similaridade coseno
        from torch.nn.functional import cosine_similarity
        scores = cosine_similarity(query_embedding.unsqueeze(0), 
                                   doc_embeddings)[0]
        
        # Top-K
        top_indices = scores.argsort(descending=True)[:k]
        
        return [
            (documents[i], float(scores[i]))
            for i in top_indices
        ]
```

### 8.4 Compressão de Contexto

```python
class ContextCompressor:
    """Técnicas para comprimir contexto sem perder informação."""
    
    def __init__(self, compression_llm):
        self.llm = compression_llm
    
    def compress_by_summarization(self, context: str, 
                                  compression_ratio: float = 0.5) -> str:
        """Resumir contexto para ocupar menos tokens."""
        
        target_length = int(len(context.split()) * compression_ratio)
        
        prompt = f"""
        Resuma o seguinte contexto em ~{target_length} palavras,
        mantendo informações críticas:
        
        {context}
        """
        
        summary = self.llm.generate(prompt)
        return summary
    
    def compress_by_extraction(self, context: str, query: str) -> str:
        """Extrair apenas as partes relevantes para a query."""
        
        prompt = f"""
        Contexto:
        {context}
        
        Query: {query}
        
        Extraia APENAS as sentenças do contexto relevantes para a query.
        Mantenha ordem original. Retorne apenas as sentenças, nenhum texto extra.
        """
        
        extracted = self.llm.generate(prompt)
        return extracted
    
    def compress_by_quantization(self, context: str) -> str:
        """Quantizar informação (simplificar linguagem)."""
        
        prompt = f"""
        Simplifique o seguinte texto, usando termos mais simples
        e estrutura mais concisa:
        
        {context}
        """
        
        simplified = self.llm.generate(prompt)
        return simplified
```

---

## **RESUMO EXECUTIVO**

### Conceitos Chave

**RAG (Retrieval-Augmented Generation)** é um paradigma que combina:
1. **Retriever**: Busca semântica em base de conhecimento
2. **Generator**: LLM que gera respostas usando contexto recuperado

### Benefícios Principais
- ✅ Reduz alucinações (respostas grounded em dados reais)
- ✅ Conhecimento sempre atualizado (sem retreinar)
- ✅ Rastreabilidade (citations dos documentos)
- ✅ Custo-efetivo (usa modelos pré-treinados)

### Limitações
- ⚠️ Latência de retrieval (~500ms)
- ⚠️ Qualidade depende do retriever ("garbage in, garbage out")
- ⚠️ Contexto limitado pela janela do LLM

### Stack Tecnológico Recomendado

```
┌─────────────────────────────────────────┐
│ APLICAÇÃO (Streamlit, FastAPI, etc.)    │
├─────────────────────────────────────────┤
│ LangChain / LlamaIndex (Orquestração)   │
├─────────────────────────────────────────┤
│ OpenAI GPT-4 / Claude 3 (LLM)           │
├─────────────────────────────────────────┤
│ Chroma / FAISS (Vector Store)           │
├─────────────────────────────────────────┤
│ Sentence Transformers (Embeddings)      │
├─────────────────────────────────────────┤
│ Documentos (PDFs, Wikis, APIs)          │
└─────────────────────────────────────────┘
```

### Próximos Passos

1. **Implementar MVP**: Comece simples (1 retriever, 1 LLM)
2. **Avaliar performance**: Use RAGAS para métricas
3. **Otimizar**: Ajuste chunking, embedding model, prompts
4. **Escalar**: Adicione monitoramento, caching, versionamento
5. **Experimentar**: Teste RAG híbrido, multimodal, avançado

---

## **RECURSOS ADICIONAIS**

### Papers Fundamentais

1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)
   - Paper seminal que introduz RAG
   - https://arxiv.org/abs/2005.11401

2. **A Comprehensive Survey of Retrieval-Augmented Generation** (2024)
   - Overview estado-da-arte
   - https://arxiv.org/abs/2410.12837

3. **RAGAS: Automated Evaluation of Retrieval Augmented Generation** (2023)
   - Framework de avaliação
   - https://arxiv.org/abs/2309.15217

4. **Dense Passage Retrieval for Open-Domain Question Answering** (Karpukhin et al., 2020)
   - DPR: Dense retrieval foundational
   - https://arxiv.org/abs/2004.04906

### Bibliotecas Python

- **LangChain**: Orquestração, chains, memoria
- **LlamaIndex**: Indexação, retrieval especializado
- **Chroma**: Vector store simples e rápido
- **FAISS**: Vector search em escala
- **Weaviate**: Vector DB enterprise
- **Sentence-Transformers**: Embeddings
- **RAGAS**: Avaliação automática

### Tutoriais Online

- LangChain Documentation: https://python.langchain.com
- LlamaIndex Docs: https://docs.llamaindex.ai
- DeepLearning.AI Short Courses (RAG)
- YouTube: Josh Maker, Matt Shumer, etc.

### Ferramentas de Prototipagem

- **Hugging Face**: Modelos pré-treinados
- **LiteLLM**: Integração com múltiplos LLMs
- **Streamlit**: UI rápida para demos
- **Gradio**: Interface para modelos

---

## **CONCLUSÃO**

RAG é um paradigma poderoso e prático que resolve limitações fundamentais dos LLMs modernos. Ao combinar retrieval dinâmico com generação em tempo de inferência, RAG permite sistemas mais precisos, atualizáveis e interpretáveis.

O futuro de RAG está em:
- **Híbrido**: Combinando múltiplos retrievers (dense + sparse + graph)
- **Multimodal**: Integrando texto, imagem, código, tabelas
- **Agentic**: Usando RAG em agentes que raciocinam e planejam
- **Especializado**: Fine-tuning de componentes para domínios específicos

Comece simples, experimente, avalie, e escale gradualmente conforme ganhar experiência!

---

**Última atualização**: Novembro 2025  
**Nível**: Avançado (pré-requisitos: Python, NLP básico, famili aridade com LLMs)  
**Tempo estimado de leitura/aprendizado**: 20-30 horas
