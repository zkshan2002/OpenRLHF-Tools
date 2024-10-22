import torch
from openrlhf.models.model import get_llm_for_sequence_regression
from typing import List

from .tokenizer import Tokenizer
from .utils import process_completions


class RewardModelWrapper:
    def __init__(self, tag: str, gpu_id: int, normalize_reward: bool = True, max_prompt_tokens: int = 1024, max_response_tokens: int = 1024):
        self._device = f"cuda:{gpu_id}"
        self._rm = get_llm_for_sequence_regression(
            tag,
            "reward",
            normalize_reward=normalize_reward,
            use_flash_attention_2=True,
            bf16=True,
            load_in_4bit=False,
            value_head_prefix="value_head",
        )
        self._rm.to(self._device)
        self._rm.eval()

        self._tokenizer = Tokenizer(tag, gpu_id)
        self._rm.config.pad_token_id = self._tokenizer.pad_token_id
        self._max_prompt_tokens = max_prompt_tokens
        self._max_response_tokens = max_response_tokens
    
    @torch.no_grad()
    def inference(self, prompts: List[str], responses: List[str]):
        completion_ids, completion_attention_mask, num_prompt_tokens, num_response_tokens, batch_response_tokens = process_completions(
            prompts, responses, self._tokenizer, self._max_prompt_tokens, self._max_response_tokens, self._device
        )
        
        r = self._rm(completion_ids, completion_attention_mask)
        r = r.to(torch.float32).cpu().numpy()
        return r
