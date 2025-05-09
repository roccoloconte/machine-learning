import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with a fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs",
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=300,
        help="Maximum length of generated text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"Loading fine-tuned model from: {args.model_path}")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Get the configuration of the fine-tuned model
    config = PeftConfig.from_pretrained(args.model_path)
    
    # Load the base model
    logger.info(f"Loading base model: {config.base_model_name_or_path}")
    
    # For MPS, need special handling
    if device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map=None,  # Don't use device_map with MPS
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the fine-tuned model
    logger.info("Loading PEFT adapter")
    model = PeftModel.from_pretrained(model, args.model_path)
    
    # Format the prompt in the same way as during training - the model was trained on instruction format
    # For Gemma, use the model's preferred format
    formatted_prompt = f"Instruction: {args.prompt}\nResponse:"
    
    if "gemma" in config.base_model_name_or_path.lower():
        formatted_prompt = f"<start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    logger.info(f"Formatted prompt: {formatted_prompt}")
    
    # Prepare input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Generate text
    logger.info("Generating response...")
    
    # Generation configuration that works with MPS
    generation_config = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_length": args.max_length + len(inputs.input_ids[0]),  # Account for prompt length
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Generate
    outputs = model.generate(**generation_config)
    
    # Decode and print responses
    for i, output in enumerate(outputs):
        # Get only the generated part, not including the prompt
        generated_text = tokenizer.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        print(f"\nGenerated Response {i+1}:")
        print(generated_text.strip())
    
    logger.info("Generation complete!")

if __name__ == "__main__":
    main() 