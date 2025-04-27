# from transformers import AutoTokenizer, Contriever

# custom_path = "/gpfsnyu/scratch/yx2432/models"

# # Download and save tokenizer and model to the specified path
# tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco", cache_dir=custom_path)
# model = Contriever.from_pretrained("facebook/contriever-msmarco", cache_dir=custom_path)


from transformers import AutoTokenizer, AutoModel

model_name = "facebook/contriever-msmarco"
custom_path = "/gpfsnyu/scratch/yx2432/models"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_path)
model = AutoModel.from_pretrained(model_name, cache_dir=custom_path)
