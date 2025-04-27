import json
import faiss
import numpy as np
import torch
import sys
sys.path.append("/gpfsnyu/scratch/yx2432/Research/Zhuzi/NLP_Project3/self-rag-main/retrieval_lm")

from src.contriever import load_retriever

# load cleaned data
with open('aml_clean.json') as f:
    docs = json.load(f)

# initialize Contriever model
retriever, tokenizer, _ = load_retriever("/gpfsnyu/scratch/yx2432/models/models--facebook--contriever-msmarco/snapshots/abe8c1493371369031bcb1e02acb754cf4e162fa")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retriever = retriever.to(device)

# generate file vector
doc_vectors = []
for doc in docs:
    inputs = tokenizer(doc["text"], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        emb = retriever(**inputs).cpu().numpy()
    doc_vectors.append(emb)

# build FAISS retireval index
doc_vectors = np.vstack(doc_vectors)
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_vectors)

# save index and metadata
faiss.write_index(index, "aml_index.faiss")
with open('aml_meta.json', 'w') as f:
    json.dump({
        "doc_ids": [d["doc_id"] for d in docs],
        "doc_texts": [d["text"] for d in docs],
        "urls": [d["url"] for d in docs]
    }, f)