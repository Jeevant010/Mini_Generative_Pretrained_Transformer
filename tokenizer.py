import os
import json
from pathlib import Path
from typing import List, Optional, Tuple
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class BytePairTokenizer:
    """Production-ready BPE tokenizer wrapper using the HuggingFace tokenizers library."""
    
    def __init__(self, config=None):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        
        # We store these to help find IDs later
        self.special_to_id = {}

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def train(self, files_or_iterator, vocab_size: int = 32000, verbose: bool = True) -> None:
        """High-performance training using Rust backend. 
        Accepts a single string, a list of strings, or an iterator."""
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=verbose
        )
        if isinstance(files_or_iterator, str):
            files_or_iterator = [files_or_iterator]
        self.tokenizer.train_from_iterator(files_or_iterator, trainer)
        self._sync_special_ids()

    def _sync_special_ids(self):
        vocab = self.tokenizer.get_vocab()
        for tok in self.special_tokens:
            if tok in vocab:
                self.special_to_id[tok] = vocab[tok]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not text: return []
        output = self.tokenizer.encode(text)
        ids = output.ids
        
        if add_bos:
            ids = [self.special_to_id["<bos>"]] + ids
        if add_eos:
            ids = ids + [self.special_to_id["<eos>"]]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(str(path))
        instance._sync_special_ids()
        return instance
