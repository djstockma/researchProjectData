import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Model
model_name = "zai-org/GLM-4.7-Flash"

# Load tokenizer and model (trust_remote_code=True required for GLM)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)

# Sample prompt
prompt = "Write a Python function that returns the nth Fibonacci number."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))