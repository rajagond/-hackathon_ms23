#!/bin/bash

pip3 install -q -U bitsandbytes transformers xformers peft accelerate datasets trl einops auto-gptq optimum nvidia-ml-py3

python3 finetune_phi.py