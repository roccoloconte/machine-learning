import os
import torch
import argparse
import logging
import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset
import bitsandbytes as bnb

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with QLoRA")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/gemma-7b-it",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="databricks/databricks-dolly-15k",
        help="Dataset name or path for fine-tuning",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to use from the dataset (for testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tuning/outputs",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU/TPU core/CPU for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU/TPU core/CPU for evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to use",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length the model can handle",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Rank of LoRA matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha parameter for LoRA scaling",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--use_4bit_quantization",
        action="store_true",
        help="Whether to use 4-bit quantization (disable for Mac)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logger.info(f"Using model: {args.model_name_or_path}")
    
    # Load the dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    
    # Check if we're using a subset of the dataset
    if args.max_samples > 0:
        logger.info(f"Using a subset of {args.max_samples} samples")
        dataset = load_dataset(args.dataset_name, split=f"train[:{args.max_samples}]")
        # Create a small validation set
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        dataset = load_dataset(args.dataset_name)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"] if "validation" in dataset else None
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    # Check for Apple Silicon and set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
        # 4-bit quantization not compatible with MPS, disable it
        args.use_4bit_quantization = False
        logger.warning("4-bit quantization disabled for Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name_or_path}")
    
    if args.use_4bit_quantization:
        # BitsAndBytes configuration for quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        # Prepare the model for QLoRA fine-tuning
        model = prepare_model_for_kbit_training(model)
    else:
        # For Apple Silicon M chips, use standard loading without quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if device.type == "mps":
            model = model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set up LoRA configuration - adjust target_modules based on model architecture
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Add additional modules if present in the model
    if hasattr(model, "get_decoder"):
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])
    elif any("gate_proj" in name for name, _ in model.named_modules()):
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    
    # Apply LoRA to the model
    logger.info("Applying LoRA to the model")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocess the dataset - adjust for Dolly dataset format
    def preprocess_function(examples):
        # For Dolly dataset, concatenate instruction and context
        if "instruction" in examples and "context" in examples and "response" in examples:
            # Format: Instruction: [instruction] Context: [context] Response: [response]
            texts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                context = examples["context"][i] if examples["context"][i] else ""
                response = examples["response"][i]
                
                # Check if we're using a Gemma model and format accordingly
                if "gemma" in args.model_name_or_path.lower():
                    # Format for Gemma models
                    if context:
                        text = f"<start_of_turn>user\n{instruction}\n{context}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
                    else:
                        text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
                else:
                    # Standard format for other models
                    if context:
                        text = f"Instruction: {instruction}\nContext: {context}\nResponse: {response}"
                    else:
                        text = f"Instruction: {instruction}\nResponse: {response}"
                texts.append(text)
        else:
            # Fallback for other datasets
            texts = examples["text"] if "text" in examples else examples["content"] if "content" in examples else []
            
        # Tokenize the texts
        tokenized_inputs = tokenizer(
            texts, 
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels (next token prediction)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        return tokenized_inputs

    # Preprocess the datasets
    logger.info("Preprocessing datasets")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    
    # Define training arguments
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        report_to="tensorboard",
        # Use paged_adamw_8bit only if not on MPS
        optim="paged_adamw_8bit" if not device.type == "mps" else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True if not device.type == "mps" else False,
        # bf16 for newer GPUs if available, not for MPS
        bf16=False,
    )
    
    # Initialize the Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving the final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main() 