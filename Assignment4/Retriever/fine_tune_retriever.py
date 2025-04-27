import json
import torch
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

# Load the formatted training data
with open('retriever_training_data.json', 'r') as f:
    training_data = json.load(f)

# Define a dataset class for contrastive learning
class RetrieverDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = self.tokenizer(item['query'], 
                              truncation=True, 
                              max_length=self.max_length, 
                              padding='max_length', 
                              return_tensors="pt")
        
        pos_doc = self.tokenizer(item['pos_doc'], 
                                truncation=True, 
                                max_length=self.max_length, 
                                padding='max_length', 
                                return_tensors="pt")
        
        neg_doc = self.tokenizer(item['neg_doc'], 
                                truncation=True, 
                                max_length=self.max_length, 
                                padding='max_length', 
                                return_tensors="pt")
        
        return {
            'query_input_ids': query['input_ids'].squeeze(),
            'query_attention_mask': query['attention_mask'].squeeze(),
            'pos_doc_input_ids': pos_doc['input_ids'].squeeze(),
            'pos_doc_attention_mask': pos_doc['attention_mask'].squeeze(),
            'neg_doc_input_ids': neg_doc['input_ids'].squeeze(),
            'neg_doc_attention_mask': neg_doc['attention_mask'].squeeze()
        }

# Load pre-trained retriever model
model_name = "/gpfsnyu/scratch/yx2432/models/models--facebook--contriever-msmarco/snapshots/abe8c1493371369031bcb1e02acb754cf4e162fa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Add domain-specific tokens (optional)
domain_tokens = ["AML", "KYC", "SAR", "FDIC", "BSA"]
tokenizer.add_tokens(domain_tokens)
model.resize_token_embeddings(len(tokenizer))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare dataset and dataloader
dataset = RetrieverDataset(training_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training parameters
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Define the contrastive loss function
def contrastive_loss(query_emb, pos_emb, neg_emb, temperature=0.1):
    # Normalize embeddings
    query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
    neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)
    
    # Calculate similarity scores
    pos_score = torch.sum(query_emb * pos_emb, dim=1) / temperature
    neg_score = torch.sum(query_emb * neg_emb, dim=1) / temperature
    
    # Calculate loss
    scores = torch.cat([pos_score.unsqueeze(1), neg_score.unsqueeze(1)], dim=1)
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    loss = torch.nn.functional.cross_entropy(scores, labels)
    
    return loss

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        # Move batch to device
        query_input_ids = batch['query_input_ids'].to(device)
        query_attention_mask = batch['query_attention_mask'].to(device)
        pos_doc_input_ids = batch['pos_doc_input_ids'].to(device)
        pos_doc_attention_mask = batch['pos_doc_attention_mask'].to(device)
        neg_doc_input_ids = batch['neg_doc_input_ids'].to(device)
        neg_doc_attention_mask = batch['neg_doc_attention_mask'].to(device)
        
        # Forward pass
        query_emb = model(query_input_ids, attention_mask=query_attention_mask).pooler_output
        pos_emb = model(pos_doc_input_ids, attention_mask=pos_doc_attention_mask).pooler_output
        neg_emb = model(neg_doc_input_ids, attention_mask=neg_doc_attention_mask).pooler_output
        
        # Calculate loss
        loss = contrastive_loss(query_emb, pos_emb, neg_emb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_retriever")
tokenizer.save_pretrained("fine_tuned_retriever")
print("Fine-tuning completed and model saved!")