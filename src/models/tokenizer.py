import torch
from transformers import AutoTokenizer
from typing import List, Dict


class Tokenizer:
    def __init__(self, tag: str, gpu_id: int):
        self._device = f"cuda:{gpu_id}"
        self._tokenizer = AutoTokenizer.from_pretrained(tag)

        self.eos_token = self._tokenizer.eos_token
        self.pad_token = self._tokenizer.pad_token
        self.eos_token_id = self._tokenizer.eos_token_id
        self.pad_token_id = self._tokenizer.pad_token_id
    
    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs):
        message_templated: str = self._tokenizer.apply_chat_template(messages, **kwargs)
        return message_templated

    
    def text_to_ids(self, text: str, max_tokens: int = 1024, return_mask: bool = False):
        assert isinstance(text, str)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_tokens,
            padding=False,
            truncation=True,
        )
        input_ids: torch.LongTensor = inputs["input_ids"][0].to(self._device)
        attention_mask: torch.Tensor = inputs["attention_mask"][0].to(self._device)
        assert torch.all(attention_mask)

        if return_mask:
            return input_ids, attention_mask
        else:
            return input_ids
    
    def ids_to_tokens(self, ids: torch.LongTensor, skip_special_tokens: bool = False):
        tokens: List[str] = self._tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        return tokens

    def ids_to_text(self, ids: torch.LongTensor, skip_special_tokens: bool = False):
        tokens = self.ids_to_tokens(ids, skip_special_tokens)
        text = self._tokenizer.convert_tokens_to_string(tokens)
        return text
