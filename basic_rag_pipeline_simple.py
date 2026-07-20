import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline

# =====================================================================
# INTERVIEW-SAFE EVALUATION CORE (Zero-Dependency Metrics Engine)
# =====================================================================
def calculate_rag_metrics(retrieved_ids: list, ground_truth_ids: list, total_db_size: int):
    retrieved_set = set(retrieved_ids)
    gt_set = set(ground_truth_ids)
    
    # Core Classification Counters
    tp = len(retrieved_set.intersection(gt_set))
    fp = len(retrieved_set - gt_set)
    fn = len(gt_set - retrieved_set)
    tn = total_db_size - (len(retrieved_set) + fn)
    
    # Mathematical Calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    # Calculate MRR (Mean Reciprocal Rank) for the query
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            mrr = 1.0 / rank
            break  # Exit immediately upon hitting the first correct match
            
    return {"precision": precision, "recall": recall, "accuracy": accuracy, "mrr": mrr}


# =====================================================================
# THE COMPLETE PIPELINE ENGINE
# =====================================================================
def run_rag_with_evals(file_path: str, query: str, expected_gt_ids: list, top_k: int = 2):
    # 1. Read document text payload
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 2. Break down text into 2-3 distinct chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_text(text)
    total_chunks = len(chunks)
    
    # 3. Setup open-source Embeddings & In-Memory Vector Store
    hf_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection(name="pure_python_rag", embedding_function=hf_embeddings)
    
    # 4. Ingest Chunks
    ids = [f"chunk_{i}" for i in range(total_chunks)]
    collection.add(documents=chunks, ids=ids)
    
    # 5. Semantic Document Retrieval Sweep
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved_ids = results["ids"][0]
    retrieved_texts = results["documents"][0]
    
    # 6. Basic LLM Generation Step
    generator = pipeline("text-generation", model="gpt2")
    context = " ".join(retrieved_texts)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    llm_output = generator(prompt, max_new_tokens=20, pad_token_id=50256)[0]["generated_text"]
    clean_answer = llm_output.replace(prompt, "").strip()
    
    # 7. Apply Metric Logic Directly on live Pipeline Output Data
    scores = calculate_rag_metrics(retrieved_ids, expected_gt_ids, total_chunks)
    
    return clean_answer, retrieved_ids, scores


# --- Test Automated Verification Suite ---
if __name__ == "__main__":
    file_name = "interview_sample.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(
            "Hexaware AI Talent Quest evaluates technical architect roles.\n"                 # chunk_0
            "The core interview focus is multi-agent systems and real-time retrieval metrics.\n" # chunk_1
            "Engineering delivery hubs are located in India, US, and Europe.\n"                 # chunk_2
            "The hybrid work policy requires developers to coordinate schedules with their managers." # chunk_3
        )

    # Question matching targeting chunk_2
    user_query = "Where are the engineering hubs located?"
    target_ground_truth = ["chunk_1", "chunk_0"] 
    
    answer, fetched_ids, metrics = run_rag_with_evals(file_name, user_query, target_ground_truth)
    
    print("\n================== 📊 EXECUTION RUN METRICS ==================")
    print(f"User Query      : {user_query}")
    print(f"Retrieved Chunks: {fetched_ids} (Expected Ground Truth: {target_ground_truth})")
    print(f"LLM Response    : {answer[:45]}...")
    print("--------------------------------------------------------------")
    print(f"Precision       : {metrics['precision']}")
    print(f"Recall          : {metrics['recall']}")
    print(f"Accuracy        : {metrics['accuracy']}")
    print(f"Reciprocal Rank : {metrics['mrr']} (MRR calculation value)")
    print("==============================================================")
    
    os.remove(file_name)