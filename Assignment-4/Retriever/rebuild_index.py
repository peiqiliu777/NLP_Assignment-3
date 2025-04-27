import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load the fine-tuned retriever
model_path = "fine_tuned_retriever"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the clean documents
with open('aml_clean.json', 'r') as f:
    docs = json.load(f)

# Generate document embeddings
doc_ids = []
doc_texts = []
doc_urls = []
embeddings = []

batch_size = 8
for i in range(0, len(docs), batch_size):
    batch_docs = docs[i:i+batch_size]
    batch_texts = [doc["text"] for doc in batch_docs]
    
    # Tokenize batch
    inputs = tokenizer(batch_texts, 
                      padding=True, 
                      truncation=True, 
                      return_tensors="pt").to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeddings = outputs.pooler_output.cpu().numpy()
    
    # Store results
    embeddings.append(batch_embeddings)
    doc_ids.extend([doc["doc_id"] for doc in batch_docs])
    doc_texts.extend(batch_texts)
    doc_urls.extend([doc["url"] for doc in batch_docs])
    
    if (i + batch_size) % 100 == 0:
        print(f"Processed {i + batch_size}/{len(docs)} documents")

# Combine all embeddings
embeddings = np.vstack(embeddings)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
index.add(embeddings)

# Save the index and metadata
faiss.write_index(index, "aml_finetuned_index.faiss")
with open('aml_finetuned_meta.json', 'w') as f:
    json.dump({
        "doc_ids": doc_ids,
        "doc_texts": doc_texts,
        "urls": doc_urls
    }, f, ensure_ascii=False, indent=2)

print("Retrieval index built and saved!")