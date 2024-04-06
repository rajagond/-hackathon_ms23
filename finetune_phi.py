from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from pynvml import *
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
import time, torch

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

base_model_id = "kaitchup/phi-1_5-safetensors"
#base_model_id = "microsoft/phi-1_5" # uncomment this line if you wish to load the original version of the model

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id  , use_fast=True, max_length=250, Truncation=True)
#Load the model with fp16
model =  AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": 0})
print(print_gpu_utilization())


duration = 0.0
total_length = 0
prompt = []
prompt.append("Write the recipe for a chicken curry with coconut milk.")
prompt.append("Translate into French the following sentence: I love bread and cheese!")
prompt.append("Cite 20 famous people.")
prompt.append("Where is the moon right now?")

for i in range(len(prompt)):
  model_inputs = tokenizer(prompt[i], return_tensors="pt").to("cuda:0")
  start_time = time.time()
  with torch.autocast(model.device.type, dtype=torch.float16, enabled=True): #Recommendation by Microsoft (see the model card) but I'm not sure whether it's very useful...
    output = model.generate(**model_inputs, max_length=500)[0]
  duration += float(time.time() - start_time)
  total_length += len(output)
  tok_sec_prompt = round(len(output)/float(time.time() - start_time),3)
  print("Prompt --- %s tokens/seconds ---" % (tok_sec_prompt))
  print(print_gpu_utilization())
  print(tokenizer.decode(output, skip_special_tokens=True))

tok_sec = round(total_length/duration,3)
print("Average --- %s tokens/seconds ---" % (tok_sec))
