from typing import List, Dict, TypedDict
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import (
    ParentDocumentRetriever,
    EnsembleRetriever,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ==========================================
# STEP 1: Load Data & Configure Splitters
# ==========================================
raw_text = """
Wipro was founded in 1945 in Maharashtra, India. It started as a vegetable oil manufacturer.
In the 1980s, Wipro shifted focus to IT services and technology solutions.
Today, Wipro's primary global headquarters are located in Bengaluru, Karnataka, India.
"""
docs = [Document(page_content=raw_text)]

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# ==========================================
# STEP 2: Parent-Child Vector & Doc Stores
# ==========================================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="child_chunks", embedding_function=embeddings)
docstore = InMemoryStore()

# 2A. Parent-Child Dense Retriever (Vector Search)
parent_dense_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
parent_dense_retriever.add_documents(docs)

# 2B. Sparse Retriever (BM25 for exact keyword matching)
# We index the child chunks in BM25 for precise token matching
all_child_chunks = child_splitter.split_documents(docs)
bm25_retriever = BM25Retriever.from_documents(all_child_chunks)
bm25_retriever.k = 3

# ==========================================
# STEP 3: Hybrid Search + Reranker Setup
# ==========================================
# Combine Dense (Parent-Child) + Sparse (BM25) with equal weights
hybrid_retriever = EnsembleRetriever(
    retrievers=[parent_dense_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# Cross-Encoder Reranker
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=1)

# Final Reranked Hybrid Retriever
final_rag_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)

# ==========================================
# STEP 4: LangGraph Agent Execution Nodes
# ==========================================
class AgentState(TypedDict):
    question: str
    retrieved_parents: List[Document]
    final_answer: str

def retrieve_node(state: AgentState) -> Dict:
    """Graph Node: Executes Hybrid Search + Reranking -> Returns Parent Document."""
    # Runs Dense + Sparse -> Cross-Encoder Rerank -> Extracts Parent Context
    retrieved_docs = final_rag_retriever.invoke(state["question"])
    return {"retrieved_parents": retrieved_docs}

def generate_node(state: AgentState) -> Dict:
    """Graph Node: Generates output from top reranked context."""
    context = state["retrieved_parents"][0].page_content
    query = state["question"]
    
    # LLM execution step
    answer = f"Context Used:\n'{context.strip()}'\n\nAnswer: Wipro's headquarters are located in Bengaluru, India."
    return {"final_answer": answer}

# Build LangGraph State Machine
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# Execute Pipeline
result = app.invoke({"question": "Where is Wipro headquarters located?"})
print("--- FINAL OUTPUT ---")
print(result["final_answer"])