#!/bin/bash

# Script to run QLoRA fine-tuning on MacBook Pro M3

# Set the Python path if needed
# export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install dependencies if needed
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Create output directory
mkdir -p ./outputs

echo "Starting QLoRA fine-tuning with Gemma..."
python3 qlora_finetune.py \
  --model_name_or_path google/gemma-7b-it \
  --dataset_name databricks/databricks-dolly-15k \
  --max_samples 500 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --max_seq_length 256 \
  --output_dir ./outputs \
  --save_steps 250 \
  --logging_steps 50

echo "Fine-tuning complete!" 