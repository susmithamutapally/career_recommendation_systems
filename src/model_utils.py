from transformers import AutoModel, AutoTokenizer
import torch

# Load model and tokenizer once
model_name = "ElenaSenger/career-path-representation-mpnet-karrierewege"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_career_path_embedding(text):
    """Returns the embedding for a given career path string."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output