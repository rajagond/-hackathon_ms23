from huggingface_hub import HfFolder, login 
  
# Replace 'your_token' with your actual Hugging Face API token  
_token = "<Access_token>"  
  
# Save the token  
HfFolder.save_token(_token)  
login(_token)

import torch

print("Number of GPUs available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print(torch.cuda.is_available())

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/phi-1_5"
base_model_path="./huggingface/microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16)

#Save the model and the tokenizer to your PC
model.save_pretrained(base_model_path, from_pt=True) 
tokenizer.save_pretrained(base_model_path, from_pt=True)

print("Model saved")

# Now load the model 
from transformers import  LlamaForCausalLM, LlamaTokenizer, pipeline

base_model_path="./huggingface/microsoft/phi-1_5"
model = LlamaForCausalLM.from_pretrained(base_model_path)
tokenizer =LlamaTokenizer.from_pretrained(base_model_path)

print("Model Loaded\n")

# Inference from it 
pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                device_map="auto",
                max_new_tokens = 512,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                trust_remote_code=True
                )
sequences = pipe(
    'I liked "Maneskin" and "Pink Floyd". Do you have any recommendations of other groups I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


print("\nThanks for using llama inference\n")

