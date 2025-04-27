import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
import matplotlib.pyplot as plt

# Load the fine-tuned retriever
model_path = "fine_tuned_retriever"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the FAISS index and metadata
index = faiss.read_index("aml_finetuned_index.faiss")
with open('aml_finetuned_meta.json', 'r') as f:
    metadata = json.load(f)

# Load the test data (hold-out set from train_qa_pairs.json)
with open('test_qa_pairs.json', 'r') as f:
    test_data = json.load(f)

# Evaluation metrics
precision_at_k = []
recall_at_k = []
ndcg_scores = []
mrr_scores = []

# Process test questions
k_values = [1, 3, 5, 10]
for sample in test_data:
    question = sample['question']
    
    # Get ground truth document IDs
    relevant_doc_ids = [doc['id'] for doc in sample['docs']]
    
    # Encode the question
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        query_emb = model(**inputs).pooler_output.cpu().numpy()
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_emb)
    
    # Search for top-k documents
    max_k = max(k_values)
    scores, doc_indices = index.search(query_emb, max_k)
    
    # Convert document indices to document IDs
    retrieved_doc_ids = [metadata["doc_ids"][idx] for idx in doc_indices[0]]
    
    # 在评估脚本中添加调试信息
    print("Sample retrieved doc_ids:", retrieved_doc_ids[:3])
    print("Sample relevant doc_ids:", relevant_doc_ids[:3])
    
    # Calculate precision and recall at different k values
    for k in k_values:
        top_k_doc_ids = retrieved_doc_ids[:k]
        
        # Count relevant documents in top-k
        relevant_retrieved = [doc_id for doc_id in top_k_doc_ids if doc_id in relevant_doc_ids]
        
        # Calculate precision@k and recall@k
        precision = len(relevant_retrieved) / k if k > 0 else 0
        recall = len(relevant_retrieved) / len(relevant_doc_ids) if len(relevant_doc_ids) > 0 else 0
        
        precision_at_k.append((k, precision))
        recall_at_k.append((k, recall))
    
    # Calculate NDCG@10
    y_true = np.zeros(max_k)
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            y_true[i] = 1
    
    y_score = np.array(scores[0])
    if np.sum(y_true) > 0:  # Only calculate NDCG if there are relevant documents
        ndcg = ndcg_score(np.array([y_true]), np.array([y_score]))
        ndcg_scores.append(ndcg)
    
    # Calculate MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / (i + 1)
            break
    mrr_scores.append(mrr)

# Calculate average metrics
avg_precision = {}
avg_recall = {}
for k in k_values:
    k_precision = [p for k_val, p in precision_at_k if k_val == k]
    k_recall = [r for k_val, r in recall_at_k if k_val == k]
    avg_precision[k] = sum(k_precision) / len(k_precision) if k_precision else 0
    avg_recall[k] = sum(k_recall) / len(k_recall) if k_recall else 0

avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0

# Print evaluation results
print("Retriever Evaluation Results")
print("===========================")
for k in k_values:
    print(f"Precision@{k}: {avg_precision[k]:.4f}")
    print(f"Recall@{k}: {avg_recall[k]:.4f}")
print(f"NDCG@10: {avg_ndcg:.4f}")
print(f"MRR: {avg_mrr:.4f}")

# Plot precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(list(avg_recall.values()), list(avg_precision.values()), 'o-', markersize=8)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Retriever')
plt.grid(True)
plt.savefig('precision_recall_curve.png')
plt.close()

# Compare with baseline (optional)
print("\nComparison with Baseline Model")
print("=============================")
# Here you would load a baseline model (e.g., BM25 or original Contriever)
# and compare the metrics