import torch
from openrlhf.models import Actor
from typing import List

from .tokenizer import Tokenizer
from .utils import process_prompts, count_leading_1

class ActorWrapper:
    def __init__(self, tag: str, gpu_id: int, max_prompt_tokens: int = 1024, max_response_tokens: int = 1024):
        self._device = f"cuda:{gpu_id}"
        self._actor = Actor(
            tag,
            use_flash_attention_2=True,
            bf16=True,
            load_in_4bit=False,
        )
        self._actor.to(self._device)
        self._actor.eval()
        
        self._max_prompt_tokens = max_prompt_tokens
        self._max_response_tokens = max_response_tokens
        
        self._tokenizer = Tokenizer(tag, gpu_id)
        self._generation_kwargs = dict(
            do_sample=True,
            max_new_tokens=self._max_response_tokens,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

    @torch.no_grad()
    def generate(self, prompts: List[str]):
        prompt_ids, prompt_mask = process_prompts(prompts, self._tokenizer, self._max_prompt_tokens)

        batch_completion_ids, attention_mask, action_mask = self._actor.generate(prompt_ids, attention_mask=prompt_mask, **self._generation_kwargs)
        prompt_length = prompt_ids.size(1)

        responses = []
        for completion_ids in batch_completion_ids:
            response_ids = completion_ids[prompt_length:]
            pad_mask = response_ids == self._tokenizer.pad_token_id
            n_trailing_pads = count_leading_1(pad_mask.flip(0))
            if n_trailing_pads > 0:
                response_ids = response_ids[:-n_trailing_pads]
            response = self._tokenizer.ids_to_text(response_ids, skip_special_tokens=False)
            responses.append(response)
        return responses
