import logging
from collections import defaultdict
logger = logging.getLogger(__name__)

class BPETokenizer:
    """
    Implementation of Byte Pair Encoding tokenizer for text compression and tokenization.
    BPE iteratively merges the most frequent pairs of adjacent tokens.
    """
    def __init__(self):
        self.vocab = {}  # Maps tokens to token IDs
        self.merge_rules = []  # List of token pairs to merge in order
        self.id_to_token = {}  # Maps token IDs back to tokens (for detokenization)
        logger.info("Initialized BPETokenizer")

    def train(self, text, desired_vocab_size):
        """
        Train the BPE tokenizer on the input text.
        
        Args:
            text: The input text to train on
            desired_vocab_size: Target size for the vocabulary
            
        Returns:
            vocab: The vocabulary mapping (token -> id)
            merge_rules: List of merge rules learned during training
        """
        # Start with character-level vocabulary
        logger.info(f"Starting tokenizer training with target vocabulary size: {desired_vocab_size}")
        chars = sorted(list(set(text)))
        logger.info(f"Initial character vocabulary size: {len(chars)}")
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        tokens = list(text)
        num_merges = desired_vocab_size - len(self.vocab)
        logger.info(f"Will perform {num_merges} merge operations")
        
        # Iteratively find and merge the most frequent adjacent token pairs
        for i in range(num_merges):
            if i % 1000 == 0:  # Log progress periodically
                logger.info(f"Processing merge operation {i}/{num_merges}")
            
            pair_freq = self.get_pair_freq(tokens)
            if not pair_freq:
                logger.info("No more pairs to merge, stopping early")
                break
                
            most_freq_pair = max(pair_freq, key=pair_freq.get)
            freq = pair_freq[most_freq_pair]
            if i % 1000 == 0:  # Log details of significant merges
                logger.info(f"Merging pair {most_freq_pair} with frequency {freq}")
                
            self.merge_rules.append(most_freq_pair)
            tokens = self.merge_pair(tokens, most_freq_pair)
            new_token = ''.join(most_freq_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
            
        logger.info(f"Tokenizer training completed. Final vocabulary size: {len(self.vocab)}")
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        return self.vocab, self.merge_rules

    def tokenize(self, text):
        """
        Convert text into tokens using the learned BPE merge rules.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        tokens = list(text)  # Start with character-level tokens
        # Apply merge rules sequentially
        for pair in self.merge_rules:
            tokens = self.merge_pair(tokens, pair)
        return tokens

    def detokenize(self, token_ids):
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Reconstructed text
        """
        tokens = [self.id_to_token[id] for id in token_ids]
        return ''.join(tokens)

    def get_pair_freq(self, tokens):
        """
        Count frequencies of adjacent token pairs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary mapping token pairs to their frequencies
        """
        pair_freq = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += 1
        return pair_freq

    def merge_pair(self, tokens, pair):
        """
        Apply a single merge operation on the token list.
        
        Args:
            tokens: List of tokens
            pair: The pair of tokens to merge
            
        Returns:
            New list of tokens with specified pair merged
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
