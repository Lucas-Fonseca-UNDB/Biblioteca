# RAG: Exemplos de Código Prontos
## Snippets Prontos para Copy-Paste e Adaptação

---

## 1️⃣ RAG Simples com Chroma + OpenAI

### Arquivo: `rag_simple.py`

```python
"""
RAG mais simples possível.
Requisitos: OpenAI API key em .env
"""

import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# 1. Carregar documentos
print("📄 Carregando documentos...")
loader = PyPDFLoader("documento.pdf")  # Substitua por seu PDF
documents = loader.load()
print(f"✓ {len(documents)} páginas carregadas")

# 2. Dividir em chunks
print("✂️  Dividindo em chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✓ {len(chunks)} chunks criados")

# 3. Criar embeddings e vector store
print("🔢 Criando embeddings (pode levar 1-2 min)...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(chunks, embeddings)
print(f"✓ Vector store criado com {len(chunks)} embeddings")

# 4. Criar retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 5. Configurar LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 6. Criar prompt customizado
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil. Responda baseado no contexto fornecido. "
               "Se não souber, diga 'Não encontrei informação'."),
    ("human", "Contexto:\n{context}\n\nPergunta: {question}")
])

# 7. Criar RAG chain
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 8. Executar queries
queries = [
    "Qual é a política de férias?",
    "Quais são os benefícios oferecidos?",
    "Como solicitar férias?"
]

for query in queries:
    print(f"\n🔍 Query: {query}")
    result = rag({"query": query})
    print(f"📝 Resposta: {result['result']}")
    print(f"📚 Fontes: {len(result['source_documents'])} documentos")
```

**Para rodar:**
```bash
export OPENAI_API_KEY="sk-..."
python rag_simple.py
```

---

## 2️⃣ RAG com Streamlit (Web UI)

### Arquivo: `rag_app.py`

```python
"""
Interface web para RAG usando Streamlit.
Rodas com: streamlit run rag_app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# Config da página
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📚 RAG Assistant - Sua Base de Conhecimento")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuração")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    process_btn = st.button("🔄 Processar Documentos")
    
    chunk_size = st.slider("Tamanho do chunk", 256, 1024, 512)
    k_results = st.slider("Top-K resultados", 1, 10, 5)

# Cache para processamento
@st.cache_resource
def load_rag_system(files, chunk_sz):
    documents = []
    for file in files:
        with open(f"temp_{file.name}", "wb") as f:
            f.write(file.getbuffer())
        
        loader = PyPDFLoader(f"temp_{file.name}")
        docs = loader.load()
        for doc in docs:
            doc.metadata['file'] = file.name
        documents.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_sz,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    for file in files:
        os.remove(f"temp_{file.name}")
    
    return vector_store

# Processar arquivos
if process_btn and uploaded_files:
    with st.spinner("Processando... pode levar 1-2 minutos"):
        vector_store = load_rag_system(uploaded_files, chunk_size)
    st.success("✅ Documentos processados!")
    st.session_state.vector_store = vector_store

# Se vector store carregado
if 'vector_store' in st.session_state:
    # Setup RAG
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": k_results}
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Input
    if query := st.chat_input("Faça uma pergunta sobre os documentos..."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)
        
        # Process
        with st.spinner("Buscando informação..."):
            result = rag({"query": query})
        
        response = result["result"]
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.write(response)
            
            # Show sources
            with st.expander("📄 Ver documentos utilizados"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.write(f"**{i}. {doc.metadata.get('file', 'Unknown')}**")
                    st.text(doc.page_content[:300] + "...")

else:
    st.info("👈 Faça upload de PDFs no painel lateral para começar!")
```

**Para rodar:**
```bash
streamlit run rag_app.py
```

---

## 3️⃣ RAG com Reranking (Melhor Precisão)

### Arquivo: `rag_advanced.py`

```python
"""
RAG com reranking para melhorar relevância.
Requer: pip install sentence-transformers
"""

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# 1. Load & chunk
documents = PyPDFLoader("documento.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 2. Create vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(chunks, embeddings)

# 3. Base retriever (retorna mais docs para reranking)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 4. Reranker (reordena os top-10 para top-5 melhor)
compressor = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
)

# 5. Compression retriever (combina base + reranker)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 6. RAG com retriever melhorado
llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,  # ← Use compression_retriever
    return_source_documents=True
)

# Test
result = rag({"query": "Qual é a política de férias?"})
print(f"Resposta: {result['result']}\n")
print("Fontes mais relevantes:")
for doc in result['source_documents']:
    print(f"  - {doc.page_content[:100]}...")
```

---

## 4️⃣ RAG com Caching (Mais Rápido)

### Arquivo: `rag_cached.py`

```python
"""
RAG com Redis cache para queries frequentes.
Requer: pip install redis
         redis-server rodando (brew install redis)
"""

import redis
import hashlib
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)
CACHE_TTL = 3600  # 1 hora

class CachedRAG:
    def __init__(self, rag_chain):
        self.rag = rag_chain
    
    def _get_cache_key(self, query: str) -> str:
        """Gerar chave de cache determinística."""
        return "rag:" + hashlib.md5(query.encode()).hexdigest()
    
    def query(self, question: str, use_cache: bool = True) -> dict:
        """Execute query com cache."""
        
        cache_key = self._get_cache_key(question)
        
        # Tentar recuperar do cache
        if use_cache:
            cached = redis_client.get(cache_key)
            if cached:
                print(f"✅ Cache hit para: {question[:50]}...")
                return json.loads(cached)
            else:
                print(f"❌ Cache miss para: {question[:50]}...")
        
        # Executar RAG
        result = self.rag({"query": question})
        
        # Salvar no cache (sem source_documents pois são objetos)
        cache_data = {
            "result": result["result"],
            "num_sources": len(result.get("source_documents", []))
        }
        
        redis_client.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(cache_data)
        )
        
        return result

# Use
documents = PyPDFLoader("documento.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(chunks, embeddings)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Wrapping com cache
cached_rag = CachedRAG(rag)

# Primeira call: slow (~2s)
result1 = cached_rag.query("Qual é a política de férias?")
print(f"Resposta: {result1['result']}\n")

# Segunda call: fast (~10ms) - do cache
result2 = cached_rag.query("Qual é a política de férias?")
print(f"Resposta (cached): {result2['result']}")
```

---

## 5️⃣ RAG com Hybrid Retrieval (Melhor Recall)

### Arquivo: `rag_hybrid.py`

```python
"""
Hybrid RAG: Dense (embeddings) + Sparse (BM25).
Requer: pip install rank-bm25
"""

import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Combina dense retrieval + BM25."""
    
    def __init__(self, vector_store, documents: list, k: int = 5):
        self.vector_store = vector_store
        self.documents_text = documents
        self.k = k
        
        # Setup BM25
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def get_relevant_documents(self, query: str):
        """Retrieve com hybrid approach."""
        
        # Dense retrieval
        dense_docs = self.vector_store.similarity_search_with_score(query, k=self.k*2)
        
        # BM25 retrieval
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[-self.k*2:][::-1]
        
        # Combine e deduplicate
        combined = {}
        
        # Add dense results
        for doc, score in dense_docs:
            key = doc.page_content[:50]  # Use content as unique key
            combined[key] = (doc, score * 0.6)  # Weight dense
        
        # Add BM25 results
        for idx in top_bm25_indices:
            doc_text = self.documents_text[idx]
            key = doc_text[:50]
            if key in combined:
                combined[key] = (combined[key][0], combined[key][1] + bm25_scores[idx] * 0.4)
            else:
                # Need to create a Document object
                from langchain.schema import Document
                doc = Document(page_content=doc_text)
                combined[key] = (doc, bm25_scores[idx] * 0.4)
        
        # Sort by combined score e retornar top-k
        sorted_results = sorted(combined.values(), key=lambda x: x[1], reverse=True)[:self.k]
        return [doc for doc, _ in sorted_results]

# Use
documents = PyPDFLoader("documento.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(chunks, embeddings)

# Extract text from chunks
chunk_texts = [chunk.page_content for chunk in chunks]

# Create hybrid retriever
hybrid_retriever = HybridRetriever(vector_store, chunk_texts, k=5)

# Wrap em um retriever LangChain
from langchain.schema.retriever import BaseRetriever
from typing import List

class HybridRetrieverWrapper(BaseRetriever):
    def __init__(self, hybrid_retriever):
        self.hybrid = hybrid_retriever
    
    def _get_relevant_documents(self, query: str) -> List:
        return self.hybrid.get_relevant_documents(query)

wrapped_retriever = HybridRetrieverWrapper(hybrid_retriever)

# Use com RAG
llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=wrapped_retriever,
    return_source_documents=True
)

result = rag({"query": "Política de férias"})
print(result["result"])
```

---

## 6️⃣ RAG com Verificação de Alucinação

### Arquivo: `rag_verification.py`

```python
"""
RAG com verificação automática de alucinação.
"""

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

class VerifiedRAG:
    """RAG com verificação de hallucination."""
    
    def __init__(self, rag_chain, llm_verifier):
        self.rag = rag_chain
        self.verifier = llm_verifier
    
    def verify_response(self, response: str, context: str) -> bool:
        """Verificar se resposta é suportada pelo contexto."""
        
        verify_prompt = f"""
        Contexto dos documentos:
        {context}
        
        Resposta do assistente:
        {response}
        
        A resposta é totalmente suportada pelo contexto?
        Responda com SIM ou NÃO.
        """
        
        verification = self.verifier.generate(verify_prompt)
        
        return "SIM" in verification.upper()
    
    def query(self, question: str, max_retries: int = 2) -> dict:
        """Query com verificação."""
        
        for attempt in range(max_retries):
            print(f"Tentativa {attempt + 1}/{max_retries}...")
            
            # Execute RAG
            result = self.rag({"query": question})
            response = result["result"]
            context = "\n".join([
                doc.page_content for doc in result.get("source_documents", [])
            ])
            
            # Verificar
            if self.verify_response(response, context):
                print("✅ Resposta verificada como correta!")
                return result
            else:
                print("❌ Resposta contém informação não suportada. Tentando novamente...")
                
                # Se falhou, tentar com mais contexto
                if attempt < max_retries - 1:
                    # Aumentar k para próxima tentativa
                    # (implementation específica do seu retriever)
                    pass
        
        # Retornar mesmo se não verificado
        print("⚠️  Máximo de tentativas atingido. Retornando resposta.")
        return result

# Use
documents = PyPDFLoader("documento.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(chunks, embeddings)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
verifier_llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

verified_rag = VerifiedRAG(rag, verifier_llm)

result = verified_rag.query("Qual é a política de férias?")
print(f"\nResposta Final: {result['result']}")
```

---

## 7️⃣ RAG com Logging e Monitoring

### Arquivo: `rag_monitored.py`

```python
"""
RAG com logging e monitoring para produção.
"""

import logging
import time
import json
from datetime import datetime
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MonitoredRAG:
    """RAG com logging e métricas."""
    
    def __init__(self, rag_chain):
        self.rag = rag_chain
        self.metrics = {
            "total_queries": 0,
            "total_latency": 0,
            "failed_queries": 0
        }
    
    def query(self, question: str) -> dict:
        """Query com logging."""
        
        query_id = f"{datetime.now().isoformat()}_{hash(question) % 10000}"
        start_time = time.time()
        
        logger.info(f"[{query_id}] Query iniciada: {question[:50]}...")
        
        try:
            # Execute
            result = self.rag({"query": question})
            
            # Metrics
            latency_ms = (time.time() - start_time) * 1000
            num_docs = len(result.get("source_documents", []))
            
            self.metrics["total_queries"] += 1
            self.metrics["total_latency"] += latency_ms
            
            # Log
            log_entry = {
                "query_id": query_id,
                "query": question,
                "response": result["result"][:100] + "...",
                "latency_ms": latency_ms,
                "num_docs_retrieved": num_docs,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[{query_id}] Query concluída em {latency_ms:.0f}ms")
            logger.debug(json.dumps(log_entry))
            
            return result
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics["failed_queries"] += 1
            
            logger.error(f"[{query_id}] Erro na query: {str(e)}")
            logger.error(f"[{query_id}] Latência antes de erro: {latency_ms:.0f}ms")
            
            raise
    
    def get_metrics(self) -> dict:
        """Retornar métricas."""
        
        avg_latency = self.metrics["total_latency"] / max(self.metrics["total_queries"], 1)
        
        return {
            **self.metrics,
            "avg_latency_ms": avg_latency,
            "error_rate": self.metrics["failed_queries"] / max(self.metrics["total_queries"], 1)
        }

# Use
documents = PyPDFLoader("documento.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(chunks, embeddings)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

monitored_rag = MonitoredRAG(rag)

# Execute
for query in ["Política de férias", "Benefícios", "Como solicitar?"]:
    result = monitored_rag.query(query)
    print(f"✓ {query}\n")

# Show metrics
print("\n📊 Métricas:")
print(json.dumps(monitored_rag.get_metrics(), indent=2))
```

---

## 8️⃣ RAG Evaluation com RAGAS

### Arquivo: `rag_evaluation.py`

```python
"""
Avaliar qualidade do RAG com framework RAGAS.
Requer: pip install ragas datasets
"""

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from ragas import evaluate

# Setup RAG
documents = PyPDFLoader("documento.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(chunks, embeddings)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Criar dataset de avaliação
eval_queries = [
    "Qual é a política de férias?",
    "Quais são os benefícios?",
    "Como solicitar um dia de folga?"
]

# Expected answers (ground truth)
expected_answers = [
    "20 dias úteis por ano",
    "Seguro saúde, 13º, vale refeição",
    "Solicitar via sistema RH"
]

# Executar RAG
eval_results = []

for query, expected in zip(eval_queries, expected_answers):
    result = rag({"query": query})
    
    eval_results.append({
        "question": query,
        "answer": result["result"],
        "contexts": [doc.page_content for doc in result["source_documents"]],
        "ground_truth": expected
    })

# Converter para Dataset RAGAS
eval_dataset = Dataset.from_list(eval_results)

# Avaliar
scores = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ]
)

print("📊 RAGAS Evaluation Scores:")
print(scores)

# Análise detalhada
print("\n📈 Análise por Query:")
for i, score in enumerate(scores):
    print(f"\nQuery {i+1}: {eval_queries[i]}")
    print(f"  Faithfulness: {score['faithfulness']:.3f}")
    print(f"  Answer Relevancy: {score['answer_relevancy']:.3f}")
    print(f"  Context Recall: {score['context_recall']:.3f}")
    print(f"  Context Precision: {score['context_precision']:.3f}")
```

---

## 📋 Checklist de Deploy

```markdown
### Pre-Deploy Checklist

- [ ] Código testado localmente
- [ ] Environment variables (.env) configuradas
- [ ] API keys validadas
- [ ] Logging implementado
- [ ] Error handling completo
- [ ] Rate limiting ativado
- [ ] Cache funcionando
- [ ] Métricas de performance definidas
- [ ] Documentação atualizada
- [ ] Tests automatizados passar

### Deployment Checklist

- [ ] Vector store sincronizado
- [ ] LLM API testada em produção
- [ ] Monitoring/alerting configurado
- [ ] Backup strategy implementado
- [ ] Plano de rollback
- [ ] Logs centralizados
- [ ] Health checks
- [ ] Load tests

### Post-Deploy Checklist

- [ ] Monitorar latência P50/P95/P99
- [ ] Verificar taxa de erro
- [ ] Coletar feedback de usuários
- [ ] Analisar queries mais comuns
- [ ] Otimizar índice se necessário
```

---

**Todos esses exemplos estão prontos para usar. Adapte conforme sua situação!** 🚀
