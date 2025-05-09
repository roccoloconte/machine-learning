import os
import torch
import logging
import pickle
from text.core.model import Model
from text.core.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)

class TextGenerator:
    """
    Handles text generation using a trained model
    """
    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else 
                                 "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.tokenizer = self._initialize_tokenizer()

    def _load_model(self, model_path: str) -> Model:
        """Load and initialize the model"""
        model = Model(self.config)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        logger.info(f"Loading model from: {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        model.to(self.device)
        model.eval()
        logger.info("Model loaded successfully")
        return model

    def _initialize_tokenizer(self, tokenizer_path: str = None) -> BPETokenizer:
        """
        Initialize the tokenizer either from a saved file or train a new one.
        
        Args:
            tokenizer_path: Optional path to a saved tokenizer. If not provided,
                          a new tokenizer will be trained using the training data.
        
        Returns:
            Initialized BPETokenizer
        """
        logger.info("Initializing tokenizer")
        tokenizer = BPETokenizer()
        
        if tokenizer_path and os.path.exists(tokenizer_path):
            # Load pre-trained tokenizer
            logger.info(f"Loading pre-trained tokenizer from: {tokenizer_path}")
            with open(tokenizer_path, 'rb') as f:
                tokenizer_data = pickle.load(f)
                tokenizer.vocab = tokenizer_data["vocab"]
                tokenizer.merge_rules = tokenizer_data["merge_rules"]
                tokenizer.id_to_token = tokenizer_data["id_to_token"]
            logger.info(f"Loaded tokenizer with vocabulary size: {len(tokenizer.vocab)}")
        else:
            # This should never happen in inference - we should always have a trained tokenizer
            raise ValueError(
                "No tokenizer path provided or file not found. "
                "Please train a tokenizer first using: "
                "python -m text.main tokenizer --data-path <your_data> --output-path tokenizer.pkl"
            )
            
        return tokenizer

    def generate(self, prompt: str, max_length: int = None, top_k: int = None) -> str:
        """Generate text from a prompt"""
        max_length = max_length or self.config['max_gen_length']
        top_k = top_k or self.config['top_k']
        
        logger.info(f"Generating text with prompt: '{prompt}'")
        generated_text = self.model.generate(
            self.tokenizer,
            prompt,
            max_length=max_length,
            top_k=top_k,
            device=self.device
        )
        return generated_text