import sys
sys.path.append("/gpfsnyu/scratch/yx2432/Research/Zhuzi/NLP_Project3/self-rag-main/retrieval_lm")

from passage_retrieval import Retriever

retriever = Retriever({})
retriever.setup_retriever_demo(
    model_name_or_path="/gpfsnyu/scratch/yx2432/models/models--facebook--contriever-msmarco/snapshots/abe8c1493371369031bcb1e02acb754cf4e162fa",
    passages="/gpfsnyu/scratch/yx2432/Research/Zhuzi/Project4/pre_prossessing/corpus.jsonl",
    passages_embeddings="/gpfsnyu/scratch/yx2432/Research/Zhuzi/Project4/pre_prossessing/embedded_data/*",
    n_docs=5,
    save_or_load_index=True
)

# Test the retriever
query = "What is the risk-based approach to money laundering?"
results = retriever.search_document_demo(query, n_docs=3)
for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Title: {doc['title']}")
    print(f"Text: {doc['text'][:100]}...")
    print()