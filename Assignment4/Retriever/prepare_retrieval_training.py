import json
import random

# Load the existing training data
with open('train_qa_pairs.json', 'r') as f:
    raw_data = json.load(f)

# Load the processed document chunks for negative sampling
with open('aml_clean.json', 'r') as f:
    all_chunks = json.load(f)

retriever_training_data = []

for sample in raw_data:
    question = sample['question']
    
    # Collect positive documents (relevant to the question)
    pos_docs = []
    for doc in sample['docs']:
        pos_docs.append(doc['text'])
    
    # For each positive document, create a training example
    for pos_doc in pos_docs:
        # Randomly sample negative documents (irrelevant to the question)
        # Exclude documents from the current sample
        current_doc_ids = [d['id'] for d in sample['docs']]
        neg_candidates = [chunk for chunk in all_chunks 
                         if chunk['doc_id'] not in current_doc_ids]
        
        # Sample 3-5 negative examples for each positive example
        neg_count = min(5, len(neg_candidates))
        if neg_count > 0:
            neg_docs = random.sample(neg_candidates, neg_count)
            for neg_doc in neg_docs:
                retriever_training_data.append({
                    "query": question,
                    "pos_doc": pos_doc,
                    "neg_doc": neg_doc['text']
                })

# Save the formatted retriever training data
with open('retriever_training_data.json', 'w') as f:
    json.dump(retriever_training_data, f, ensure_ascii=False, indent=2)

print(f"Generated {len(retriever_training_data)} training samples for retriever fine-tuning")