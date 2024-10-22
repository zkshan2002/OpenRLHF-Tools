import torch
import torch.nn.functional as F
from openrlhf.models import Actor
from typing import List

from .tokenizer import Tokenizer

def pad(sequences: List[torch.Tensor], side: str, value):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

def count_leading_1(mask: torch.Tensor):
    indices = torch.arange(1, mask.size(0) + 1, device=mask.device)
    n_leading_1 = (mask.cumsum(dim=0) == indices).int().sum().item()
    return n_leading_1

def process_prompts(prompts: List[str], tokenizer: Tokenizer, max_prompt_tokens: int):
    def preprocess(prompt: str, tokenizer: Tokenizer):
        messages = [
            {"content": prompt, "role": "user"},
        ]

        # "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n"
        # + "<|start_header_id|>assistant<|end_header_id|>\n"
        prompt_templated: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt_templated

    def process(prompt: str, tokenizer: Tokenizer, max_prompt_tokens: int):
        prompt_templated = preprocess(prompt, tokenizer)
        prompt_ids, prompt_mask = tokenizer.text_to_ids(prompt_templated, max_tokens=max_prompt_tokens, return_mask=True)
        return prompt_ids, prompt_mask

    batch_prompt_ids = []
    batch_prompt_mask = []
    for prompt in prompts:
        prompt_ids, prompt_mask = process(prompt, tokenizer, max_prompt_tokens)
        batch_prompt_ids.append(prompt_ids)
        batch_prompt_mask.append(prompt_mask)
    batch_prompt_ids = pad(batch_prompt_ids, "left", value=tokenizer.pad_token_id)
    batch_prompt_mask = pad(batch_prompt_mask, "left", value=0)
    return batch_prompt_ids, batch_prompt_mask

def process_completions(prompts: List[str], responses: List[str], tokenizer: Tokenizer, max_prompt_tokens: int, max_response_tokens: int, device: torch.device):
    def preprocess(prompt: str, response: str, tokenizer: Tokenizer):
        messages = [
            {"content": prompt, "role": "user"},
            {"content": response, "role": "assistant"}
        ]

        # "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n"
        # + "<|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>\n"
        messages_templated: str = tokenizer.apply_chat_template(messages, tokenize=False)

        # "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n"
        # + "<|start_header_id|>assistant<|end_header_id|>\n"
        prompt_templated: str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

        # "{response}<|eot_id|>\n"
        response_templated = messages_templated[len(prompt_templated):]

        return prompt_templated, response_templated

    def process(prompt: str, response: str, tokenizer: Tokenizer, max_prompt_tokens: int, max_response_tokens: int):
        prompt_templated, response_templated = preprocess(prompt, response, tokenizer)
        prompt_ids, prompt_mask = tokenizer.text_to_ids(prompt_templated, max_tokens=max_prompt_tokens, return_mask=True)

        response_templated = response_templated.rstrip("\n")
        if response_templated.endswith(tokenizer.eos_token):
            response_templated += f" {tokenizer.eos_token}"
        
        response_ids, response_mask = tokenizer.text_to_ids(response_templated, max_tokens=max_response_tokens, return_mask=True)
        response_ids[-1] = tokenizer.eos_token_id
        response_mask[-1] = True

        return prompt_ids, prompt_mask, response_ids, response_mask

    batch_prompt_ids = []
    batch_prompt_mask = []
    batch_response_ids = []
    batch_response_mask = []
    num_response_tokens = []
    batch_response_tokens = []
    for prompt, response in zip(prompts, responses):
        prompt_ids, prompt_mask, response_ids, response_mask = process(prompt, response, tokenizer, max_prompt_tokens, max_response_tokens)
        batch_prompt_ids.append(prompt_ids)
        batch_prompt_mask.append(prompt_mask)
        batch_response_ids.append(response_ids)
        batch_response_mask.append(response_mask)
        num_response_tokens.append(response_ids.size(0))
        response_tokens = tokenizer.ids_to_tokens(response_ids, skip_special_tokens=False)
        batch_response_tokens.append(response_tokens)
    
    batch_prompt_ids = pad(batch_prompt_ids, "left", value=tokenizer.pad_token_id)
    batch_prompt_mask = pad(batch_prompt_mask, "left", value=0)
    batch_response_ids = pad(batch_response_ids, "right", value=tokenizer.pad_token_id)
    batch_response_mask = pad(batch_response_mask, "right", value=0)

    batch_completion_ids = torch.cat([batch_prompt_ids, batch_response_ids], dim=1).to(device)
    batch_completion_mask = torch.cat([batch_prompt_mask, batch_response_mask], dim=1).to(device)
    num_prompt_tokens = batch_prompt_ids.size(1)

    return batch_completion_ids, batch_completion_mask, num_prompt_tokens, num_response_tokens, batch_response_tokens

def compute_all_logits(
    model: Actor,
    completions: torch.LongTensor,
    attention_mask: torch.Tensor,
):
    # https://github.com/OpenRLHF/OpenRLHF/issues/217
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    output = model.model(completions, attention_mask=attention_mask, position_ids=position_ids)
    all_logits = output["logits"]
    return all_logits

def compute_log_probs(
    model: Actor,
    completions: torch.LongTensor,
    attention_mask: torch.Tensor,
    num_prompt_tokens: int,
):
    all_logits = compute_all_logits(model, completions, attention_mask)

    logits = all_logits[:, num_prompt_tokens - 1:]
    pick_index = completions[:, num_prompt_tokens:]
    
    log_probs = torch.gather(logits.log_softmax(-1), dim=2, index=pick_index.unsqueeze(2)).squeeze(2)

    return log_probs
