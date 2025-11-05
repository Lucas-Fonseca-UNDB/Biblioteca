# RAG: Resumo Executivo & Quick Reference Guide
## Referência Rápida para Implementação Prática

---

## 🎯 Resumo em 5 Minutos

### O que é RAG?
**Retrieval-Augmented Generation** = Buscar informação relevante + Gerar resposta com contexto

```
Pergunta do usuário
    ↓
Buscar docs relevantes (Retriever)
    ↓
Passar contexto + pergunta para LLM (Generator)
    ↓
Resposta precisa com citations
```

### Por que usar?
- ✅ Reduz alucinações do LLM
- ✅ Conhecimento sempre atualizado (sem retreinar)
- ✅ Rastreabilidade (cita fontes)
- ✅ Funciona com dados proprietários
- ✅ Custo-efetivo (sem fine-tuning completo)

### Quando usar?
- ✅ Documentos corporativos
- ✅ FAQ / Suporte técnico
- ✅ Análise de contratos
- ✅ Healthcare / Jurídico
- ✅ Qualquer base de conhecimento dinâmica

### Quando NÃO usar?
- ❌ Tarefas de raciocínio lógico puro
- ❌ Geração criativa (poesia, ficção)
- ❌ Tasks que requerem <100ms latência

---

## 📚 Stack Técnico Recomendado (Mínimo)

```
┌─────────────────────────────────────────────┐
│ Aplicação (Streamlit/FastAPI)               │
├─────────────────────────────────────────────┤
│ LangChain (orquestração de pipeline)        │
├─────────────────────────────────────────────┤
│ OpenAI GPT-4o (LLM generator)               │
├─────────────────────────────────────────────┤
│ Chroma (vector store - local)               │
├─────────────────────────────────────────────┤
│ sentence-transformers (embeddings)          │
├─────────────────────────────────────────────┤
│ PDFs + Documentos (sua base de conhecimento)│
└─────────────────────────────────────────────┘
```

**Custo estimado**: $0-$50/mês (com modelo open-source, ~$100-300/mês com GPT-4)

---

## 💻 Setup Mínimo em 10 Minutos

### 1. Instalações
```bash
pip install langchain langchain-community langchain-openai chroma-db pypdf python-dotenv
```

### 2. Código Minimal
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Carregar PDF
loader = PyPDFLoader("documento.pdf")
documents = loader.load()

# Dividir em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Criar embeddings
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

# Setup LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Criar chain RAG
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Use
result = rag({"query": "Qual é a política de férias?"})
print(result["result"])
```

**Pronto!** Sistema funcional em ~30 linhas.

---

## 🔍 Decisões Chave (Trade-offs)

### Vector Store: Qual Escolher?

|   Critério  |      FAISS     |    Chroma   |  Weaviate  | Pinecone |
|-------------|----------------|-------------|------------|----------|
| Setup       | Fácil          | Muito fácil | Complexo   | Trivial  |
| Melhor para | Pesquisa local | Prototipo   | Enterprise | Sem-ops  |
| Custo       | $0             | $0          | $0         | $$       |
| **Recomendação para começar** | ✗ | ✅ | ✗ | ✗ |

**👉 Use Chroma para começar**

### Embedding Model: Qual Escolher?

|                  Modelo                 | Dimensão |  Velocidade  | Qualidade | Custo |
|-----------------------------------------|----------|--------------|-----------|-------|
| text-embedding-3-small                  | 1536     | Rápido       | Excelente | $$    |
| sentence-transformers/all-MiniLM-L6-v2  | 384      | Muito rápido | Boa       | $0    |
| sentence-transformers/all-mpnet-base-v2 | 768      | Moderado     | Melhor    | $0    |

**👉 Para começar**: `all-MiniLM-L6-v2` (grátis, rápido, suficiente)  
**👉 Produção**: `text-embedding-3-small` (melhor qualidade)

### LLM Generator: Qual Escolher?

|    Modelo   | Custo | Qualidade | Latência |     Ideal para     |
|-------------|-------|-----------|----------|--------------------|
| GPT-4o      | $$$   | Excelente | 1-2s     | Production         |
| Claude 3.5  | $$$   | Excelente | 1-2s     | Análise complexa   |
| Llama 2-70B | Free* | Boa       | 2-5s     | Local, open-source |
| Mistral 7B  | Free* | Boa       | 1-2s     | Rápido, local      |

**👉 Para prototipo**: GPT-4o (melhor custo-benefício)  
**👉 Privacidade total**: Llama local

### Chunking: Qual Estratégia?

```
┌─ Tamanho do chunk?
│  ├─ Pequeno (256): Mais preciso, menos contexto
│  ├─ Médio (512): ✅ RECOMENDADO
│  └─ Grande (1024): Mais contexto, menos precisão
│
├─ Tipo de chunking?
│  ├─ Fixed-size: Simples, rápido
│  ├─ Sentence-based: Melhor, mantém semântica
│  └─ Recursive: ✅ RECOMENDADO
│
└─ Overlap?
   ├─ Nenhum: Rápido, pode perder info
   ├─ 50 tokens: ✅ RECOMENDADO
   └─ 100+ tokens: Muito overlap, ineficiente
```

---

## 📊 Checklist de Implementação

### Fase 1: Prototipagem (1-2 dias)
- [ ] Ambiente setup (Python, dependências)
- [ ] Dados coletados (PDFs, documentos)
- [ ] Vector store local criado
- [ ] LLM conectado (OpenAI API key)
- [ ] Query de teste funcionando
- [ ] Interface Streamlit básica

### Fase 2: Otimização (3-5 dias)
- [ ] Métricas de avaliação definidas
- [ ] Chunking strategy otimizado
- [ ] Embedding model escolhido
- [ ] Prompt engineering refinado
- [ ] Reranker implementado (opcional)
- [ ] Caching ativado

### Fase 3: Deployment (1-2 dias)
- [ ] Logging e monitoring
- [ ] Error handling robusto
- [ ] Rate limiting
- [ ] Auto-refresh de índice
- [ ] CI/CD pipeline
- [ ] Documentação

### Fase 4: Manutenção (Ongoing)
- [ ] Monitoramento de performance
- [ ] Atualização de documentos
- [ ] Análise de queries falhadas
- [ ] A/B testing de prompts

---

## 🚀 Métricas para Monitorar

### Em Desenvolvimento
```python
# Evaluate retrieval quality
- Precision@5: % de docs relevantes no top-5
- Recall@10: % de todos os docs relevantes encontrados
- MRR: Posição do primeiro doc correto

# Evaluate generation quality
- Faithfulness: % de resposta suportada pelo contexto
- Relevance: % de resposta que aborda a pergunta
```

### Em Produção
```python
# Performance
- Latência P50, P95, P99
- Throughput (queries/segundo)
- Taxa de erro

# Satisfação
- User feedback (thumbs up/down)
- Fallback rate (reqs que falharam)
- Query diversity (distribuição de tópicos)
```

---

## 🐛 Problemas Comuns & Soluções

### Problema 1: "Resposta não relacionada à pergunta"

**Diagnóstico:**
```python
# Verificar retrieval
retrieved_docs = retriever.get_relevant_documents(query)
print(retrieved_docs)  # Estão relevantes?
```

**Soluções:**
1. Aumentar k (top-5 → top-10)
2. Mudar embedding model
3. Refinar chunking (tamanho/overlap)
4. Adicionar reranker

### Problema 2: "Alucinação ainda acontece"

**Diagnóstico:**
```python
# Verificar se informação está nos docs
context = "\n".join([doc.page_content for doc in retrieved_docs])
# Informação da resposta está em `context`?
```

**Soluções:**
1. Reforçar no system prompt: "Responda EXCLUSIVAMENTE baseado no contexto"
2. Usar model com menos tendência a alucinação (Claude vs. GPT)
3. Implementar verification loop

### Problema 3: "Latência muito alta (>2s)"

**Diagnóstico:**
```python
import time
start = time.time()
result = rag.run(query)
print(f"Latência total: {time.time() - start:.2f}s")
# Onde está o tempo?
```

**Soluções:**
1. Adicionar cache (Redis)
2. Reduzir k (top-5 em vez de top-20)
3. Usar embedding model mais rápido
4. Usar modelo LLM menor/mais rápido
5. Paralelizar retrieval + reranking

### Problema 4: "Vector store cresceu muito (GBs)"

**Soluções:**
1. Usar quantização (FAISS IndexIVFFlat)
2. Usar binary vectors (menos storage)
3. Mover para FAISS GPU
4. Usar Pinecone (cloud-based)
5. Remover documentos antigos

---

## 📈 Roadmap de Evolução

```
FASE 1 (v1.0 - Básico)
└─ Dense retrieval simples
   └─ LLM generator padrão
   └─ Sem cache

FASE 2 (v1.5 - Otimizado)
├─ Reranker
├─ Caching
├─ Melhor prompt engineering
└─ Logging básico

FASE 3 (v2.0 - Robusto)
├─ Hybrid retrieval (dense + sparse)
├─ Multi-LLM fallback
├─ Monitoring completo
├─ Auto-refresh de índice
└─ A/B testing

FASE 4 (v2.5 - Avançado)
├─ RAG Multimodal
├─ Agentic RAG (iterativo)
├─ Fine-tuned retriever
├─ Knowledge graph integration
└─ Cost optimization

FASE 5 (v3.0 - Escalado)
├─ Distributed indexing
├─ Real-time updates
├─ ML-based ranking
├─ Personalization por user
└─ Advanced analytics
```

---

## 💰 Estimativas de Custo

### Scenario 1: Startup (Volume Baixo)
```
Docs: <100K
Queries/mês: <10K
Custo mensal: $20-50

├─ OpenAI API: $10-30 (GPT-4o)
├─ Chroma (local): $0
├─ VPS (optional): $10-20
└─ Dev time: Grátis com open-source
```

### Scenario 2: Médio (Volume Moderado)
```
Docs: 100K-1M
Queries/mês: 100K
Custo mensal: $200-500

├─ OpenAI API: $100-300
├─ Pinecone (storage): $50-150
├─ VPS: $30-50
└─ Dev/ops: Part-time
```

### Scenario 3: Enterprise (Volume Alto)
```
Docs: 1M+
Queries/mês: 1M+
Custo mensal: $2000-5000

├─ LLM API: $1000-3000
├─ Weaviate/Elasticsearch: $500-1000
├─ Infrastructure: $300-1000
├─ Infra ops: Full-time
└─ Security/compliance: $200-1000
```

**ROI típico**: 3-6 meses payback (redução de custos operacionais)

---

## 📚 Projetos de Prática Recomendados

### Projeto 1: Chatbot sobre PDFs (1 semana)
**Dificuldade**: ⭐⭐ Fácil

```
Objetivo: Criar chatbot que responde perguntas sobre seus PDFs
Tecnologias: Streamlit + Chroma + GPT-4
Tempo: 3-5 horas

Deliverables:
1. Web UI para upload de PDFs
2. Chatbot interativo
3. Exibição de sources
4. Feedback do usuário
```

### Projeto 2: Sistema de Suporte Técnico (2 semanas)
**Dificuldade**: ⭐⭐⭐ Médio

```
Objetivo: Automação de tickets de suporte
Tecnologias: LangChain + Chroma + Hybrid Retrieval + FastAPI
Tempo: 10-15 horas

Features:
1. Automático categorização de tickets
2. Resposta automática (com human review)
3. Escalation para especialistas
4. Feedback loop
5. Conhecimento base auto-update
```

### Projeto 3: RAG para Análise de Contratos (1 mês)
**Dificuldade**: ⭐⭐⭐⭐ Avançado

```
Objetivo: Análise inteligente de contratos legais
Tecnologias: Weaviate + Claude 3 + Fine-tuned Retriever
Tempo: 40-60 horas

Features:
1. Extração de cláusulas
2. Comparação entre contratos
3. Alertas de risco
4. Recomendações legais
5. Dashboard de analytics
```

---

## 🔗 Links Úteis

### Documentação Oficial
- LangChain: https://python.langchain.com/docs
- LlamaIndex: https://docs.llamaindex.ai
- Chroma: https://docs.trychroma.com
- FAISS: https://github.com/facebookresearch/faiss

### Papers Importantes
- RAG Original: https://arxiv.org/abs/2005.11401 (Lewis et al., 2020)
- RAG Survey: https://arxiv.org/abs/2410.12837 (2024)
- Evaluation: https://arxiv.org/abs/2309.15217 (RAGAS)

### Tutoriais
- DeepLearning.AI: "Building RAG Applications" (grátis)
- YouTube: "LangChain RAG Tutorial" (Matt Shumer)
- Blog: "RAG Best Practices" (Anthropic Engineering)

### Comunidades
- GitHub Discussions (LangChain, LlamaIndex)
- Discord: LLaMA Community, LangChain Official
- Twitter: #RAG #LLM #AI

---

## ✅ Quick Decision Tree

```
Tenho dados dinâmicos (atualizados freq)?
├─ SIM → RAG (melhor solução)
└─ NÃO → Considerar fine-tuning

Preciso manter privacidade dos dados?
├─ SIM → RAG local + Open-source LLM
└─ NÃO → Cloud RAG OK

Budget é limitado?
├─ SIM → Open-source stack (Chroma + Mistral)
└─ NÃO → Managed services (Pinecone + GPT-4)

Latência crítica (<100ms)?
├─ SIM → Fine-tuning (sem retrieval overhead)
└─ NÃO → RAG OK (~500ms)

Preciso de explainability?
├─ SIM → RAG (com citations) + interpretabilidade
└─ NÃO → Fine-tuning OK

→ Se respondeu SIM a 3+ critérios RAG → Use RAG!
```

---

## 🎓 Próximos Passos

1. **Dia 1**: Ler Módulos 1-2 (Conceitos)
2. **Dia 2**: Ler Módulo 3 (Embeddings)
3. **Dia 3**: Implementar Módulo 5 (Código)
4. **Dia 4**: Avaliar com Módulo 6 (Métricas)
5. **Dia 5**: Explorar Módulo 8 (Avançado)

**Tempo total**: ~5-10 horas hands-on

---

## 📧 Suporte & Comunidade

Estiver preso em um problema:

1. **Stack Overflow**: Tag `langchain` ou `llamaindex`
2. **GitHub Issues**: Abra issue no repo oficial
3. **Discord Communities**: LangChain, LLaMA, etc.
4. **Blog Posts**: Medium, Dev.to com tag `RAG`

---

**Boa sorte com seu projeto RAG! 🚀**

Para dúvidas específicas do seu use case, refira-se ao Guia Completo (RAG-Guia-Completo.md).
