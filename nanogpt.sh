#!/bin/bash

sudo apt-get install python3.8
sudo apt-get install python3.8-venv

rm -rf <project_name>
python3.8 -m venv <project_name>
source <project_name>/bin/activate

pip install --upgrade pip # latest version of pip
pip3 install torch numpy transformers datasets tiktoken wandb tqdm
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# split the data into train and val
python3 data/shakespeare_char/prepare.py

# train the model on the data on a GPU(T4)
python3 train.py config/train_shakespeare_char.py

# on CPU
# python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0

# run inference on the trained model on a GPU
python3 sample.py --out_dir=out-shakespeare-char

# on CPU
# python3 sample.py --out_dir=out-shakespeare-char --device=cpu
