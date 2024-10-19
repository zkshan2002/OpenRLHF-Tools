import numpy as np
import torch
from openrlhf.models.actor import Actor
from typing import List

from .tokenizer import Tokenizer
from .utils import process_completions, compute_log_probs


class DPORewardModel:
    def __init__(self, dpo_tag: str, ref_tag: str, gpu_id: int, max_prompt_tokens: int = 1024, max_response_tokens: int = 1024):
        self._device = f"cuda:{gpu_id}"
        self._dpo_model = Actor(
            dpo_tag,
            use_flash_attention_2=True,
            bf16=True,
            load_in_4bit=False,
        )
        self._dpo_model.to(self._device)
        self._dpo_model.eval()
        
        self._ref_model = Actor(
            ref_tag,
            use_flash_attention_2=True,
            bf16=True,
            load_in_4bit=False,
        )
        self._ref_model.to(self._device)
        self._ref_model.eval()
        
        self._tokenizer = Tokenizer(ref_tag, gpu_id)
        self._max_prompt_tokens = max_prompt_tokens
        self._max_response_tokens = max_response_tokens
    
    @torch.no_grad()
    def inference(self, prompts: List[str], responses: List[str], return_response_tokens: bool = False):
        completion_ids, completion_attention_mask, num_prompt_tokens, num_response_tokens, batch_response_tokens = process_completions(
            prompts, responses, self._tokenizer, self._max_prompt_tokens, self._max_response_tokens, self._device
        )

        dpo_log_probs = compute_log_probs(self._dpo_model, completion_ids, completion_attention_mask, num_prompt_tokens)
        ref_log_probs = compute_log_probs(self._ref_model, completion_ids, completion_attention_mask, num_prompt_tokens)
        token_rewards = dpo_log_probs - ref_log_probs
        
        token_rewards: List[np.ndarray] = [
            token_rewards[i, :num_response_tokens[i]].to(torch.float32).cpu().numpy()
            for i in range(len(prompts))
        ]

        if return_response_tokens:
            return token_rewards, batch_response_tokens
        else:
            return token_rewards
