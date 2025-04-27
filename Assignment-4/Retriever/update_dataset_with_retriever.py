import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import argparse
import numpy as np
from tqdm import tqdm
import copy

# helper function to preserve complete sentences in the summary
def create_meaningful_summary(text, max_length=200):
    # return text if it's already short enough
    if len(text) <= max_length:
        return text
    
    # find the last complete sentence within the max_length
    sentences = text.split('. ')
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) + 2 <= max_length:
            if summary:
                summary += ". " + sentence
            else:
                summary = sentence
        else:
            break
    
    # ensure the summary ends with a period
    if summary and not summary.endswith('.'):
        summary += '.'
        
    return summary

class FinetuneRetriever:
    def __init__(self, model_path, index_path, meta_path):
        # load the fine-tuned retrieval model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # load FAISS index
        self.index = faiss.read_index(index_path)
        
        # load metadata
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
    
    def retrieve(self, query, top_k=5):
        # tokenize the query
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            query_emb = self.model(**inputs).pooler_output.cpu().numpy()
        
        # normalize the query embedding
        faiss.normalize_L2(query_emb)
        
        # search top-k documents
        scores, doc_indices = self.index.search(query_emb, top_k)
        
        # return retrieved documents and their scores
        results = []
        for i, idx in enumerate(doc_indices[0]):
            if idx < len(self.metadata["doc_ids"]):  # 安全检查
                text = self.metadata["doc_texts"][idx]
                results.append({
                    "id": self.metadata["doc_ids"][idx],
                    "title": f"AML Topic {i+1}",
                    "text": text,
                    "score": float(scores[0][i]),
                    "summary": create_meaningful_summary(text, 200), 
                    "extraction": text[:400] if len(text) > 400 else text 
                })
        
        return results

def convert_and_update_dataset(input_files, output_file, retriever, ndocs=5):
    """
    1. load train_qa_pairs.json and test_qa_pairs.json
    2. use fine-tuned retriever and update retrieved documents
    3. convert the dataset to a new format for later inference use
    """
    all_samples = []
    
    # process each input file
    for input_file in input_files:
        print(f"process input file: {input_file}")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # create a new list to store updated samples
        for sample in tqdm(data, desc=f"Update retrieval results in {input_file} "):
            new_sample = copy.deepcopy(sample)  
            
            # get the question
            question = sample.get("question", "")
            
            # ensure a question exist
            if not question:
                continue
                
            # use fine-tuned retriever to get new retrieved documents
            retrieved_docs = retriever.retrieve(question, top_k=ndocs)
            
            # update doc
            new_sample["docs"] = retrieved_docs
            
            all_samples.append(new_sample)
    
    # convert format
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset updated and saved to {output_file}")
    return len(all_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use finr-tuned retriever to update dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Fine-tuned model path")
    parser.add_argument("--index_path", type=str, required=True, help="FAISS index path") 
    parser.add_argument("--meta_path", type=str, required=True, help="index metadate path")
    parser.add_argument("--input_files", nargs='+', required=True, help="input path")
    parser.add_argument("--output_file", type=str, required=True, help="output path")
    parser.add_argument("--ndocs", type=int, default=5, help="number of docs for each question")
    
    args = parser.parse_args()
    
    # initialize retriever
    retriever = FinetuneRetriever(
        model_path=args.model_path,
        index_path=args.index_path,
        meta_path=args.meta_path
    )
    
    # conversion and update
    sample_count = convert_and_update_dataset(
        args.input_files,
        args.output_file,
        retriever,
        args.ndocs
    )
    
    print(f"Dataset updated, processed {sample_count} items")

    
"""
python update_dataset_with_retriever.py \
  --model_path fine_tuned_retriever \
  --index_path aml_finetuned_index.faiss \
  --meta_path aml_finetuned_meta.json \
  --input_files train_qa_pairs.json test_qa_pairs.json \
  --output_file aml_self_rag_dataset.json \
  --ndocs 5
  """