import csv
import json
from models.transformers import BertTokenizer


class SentenceTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False, *args, **kwargs):
        super().__init__(vocab_file=str(vocab_file),
                         do_lower_case=do_lower_case, *args, **kwargs)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens
