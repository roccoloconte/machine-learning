#!/bin/bash

# Script to run inference with the fine-tuned model

# Default prompt if none provided
DEFAULT_PROMPT="Explain the concept of machine learning to a high school student."
PROMPT="${1:-$DEFAULT_PROMPT}"

# Check if model exists
if [ ! -d "./outputs" ]; then
    echo "Model not found! Please run the fine-tuning script first."
    exit 1
fi

echo "Running inference with prompt: $PROMPT"
python3 qlora_inference.py \
  --model_path ./outputs \
  --prompt "$PROMPT" \
  --max_length 300 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50

echo "Inference complete!" 