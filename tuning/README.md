# QLoRA Fine-tuning for Language Models

This folder contains code for fine-tuning language models using QLoRA (Quantized Low-Rank Adaptation), which allows efficient fine-tuning of large language models on consumer hardware like the MacBook Pro M3.

## Overview

QLoRA is a technique that enables fine-tuning of large language models with significantly less memory requirements by:

1. Quantizing the pre-trained model weights to 4-bit precision (when not on Apple Silicon)
2. Using Low-Rank Adaptation (LoRA) to update only a small number of parameters
3. Keeping the majority of the model weights frozen during training

This approach makes it possible to fine-tune models with billions of parameters on consumer hardware.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Pre-trained Models Suitable for MacBook Pro M3

The implementation defaults to using Google's Gemma model (7B parameters), which can be fine-tuned on MacBook Pro M3 hardware using QLoRA. Other recommended models that work well on this hardware include:

- Mistral-7B-v0.1 (7B parameters)
- Phi-2 (2.7B parameters)
- Gemma-2B (2B parameters)
- OLMo-1B (1B parameters)
- TinyLlama-1.1B (1.1B parameters)

## MacBook Pro M3 Compatibility

The code automatically detects Apple Silicon (M1/M2/M3) and:

1. Uses the MPS (Metal Performance Shaders) backend for GPU acceleration
2. Disables 4-bit quantization (which is not compatible with MPS)
3. Uses the appropriate optimizer instead of 8-bit AdamW

This ensures that the code runs without errors on MacBook Pro M3 hardware.

## Dataset

The code uses the Databricks Dolly 15k dataset for fine-tuning. This dataset consists of instruction-following examples across various tasks including brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization.

## Usage

### Fine-tuning

To fine-tune a model:

```bash
python qlora_finetune.py \
  --model_name_or_path google/gemma-7b-it \
  --dataset_name databricks/databricks-dolly-15k \
  --max_samples 1000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --output_dir ./tuning/outputs
```

### Inference

To generate text using the fine-tuned model:

```bash
python qlora_inference.py \
  --model_path ./tuning/outputs \
  --prompt "What are some interesting aspects of education that you can tell me about?" \
  --max_length 200 \
  --temperature 0.7
```

## Customization Options

### Model Selection

You can choose different pre-trained models based on your specific requirements:

```bash
python qlora_finetune.py --model_name_or_path <model_name_or_path>
```

### LoRA Configuration

Adjust the LoRA parameters to control the fine-tuning process:

```bash
python qlora_finetune.py \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05
```

### Dataset

You can use a different dataset or just a portion of the default dataset:

```bash
python qlora_finetune.py \
  --dataset_name <dataset_name> \
  --max_samples <max_samples>
```

## Performance Considerations for MacBook Pro M3

For optimal performance on MacBook Pro M3:

1. Start with smaller models (1-3B parameters)
2. Use smaller batch sizes (1-4) with gradient accumulation (8-32) to simulate larger batches
3. Consider reducing the context length (max_seq_length) if running into memory issues
4. Monitor memory usage and temperature during training
5. For models ≤3B parameters, try using the full model without quantization
6. For larger models (≥7B parameters), reduce the batch size further and increase gradient accumulation

## Advanced Configuration

For more advanced options, see the full list of command-line arguments in the `parse_args()` function within `qlora_finetune.py`. 