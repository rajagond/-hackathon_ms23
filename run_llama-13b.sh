#!/bin/bash

sudo apt-get install python3.8
sudo apt-get install python3.8-venv
rm -rf <project_name>
python3.8 -m venv <project_name>
source <project_name>/bin/activate
pip install --upgrade pip # latest version of pip
pip3 install --quiet huggingface_hub ipywidgets
pip3 install --quiet bitsandbytes
pip3 install --quiet transformers 
pip3 install --quiet accelerate
pip3 install --quiet scipy numpy
pip3 install --quiet torch==2.0.1

echo "setup done"
nvidia-smi
python3 llama-13b.py