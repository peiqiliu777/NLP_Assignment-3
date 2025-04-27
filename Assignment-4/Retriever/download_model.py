from transformers import AutoTokenizer, Contriever

# save to local path
save_path = "/gpfsnyu/scratch/yx2432/models/mcontriever-msmarco"

# load tokenizer and model
print("Loading tokenizer and model from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained("facebook/mcontriever-msmarco")
model = Contriever.from_pretrained("facebook/mcontriever-msmarco")

# save
print(f"Saving model and tokenizer to: {save_path}")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Done! Model saved successfully.")
