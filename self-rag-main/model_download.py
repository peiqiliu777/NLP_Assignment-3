from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Define model name and save path
model_name = "selfrag/selfrag_llama2_7b"
save_directory = "/gpfsnyu/scratch/yx2432/models/selfrag_llama2_7b"

# Ensure the save directory exists
os.makedirs(save_directory, exist_ok=True)

# Download and save the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save model and tokenizer
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model downloaded and saved to {save_directory}")
