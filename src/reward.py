import numpy as np
import os
from typing import Optional, Union, Tuple, List, Dict

from .data import create_loader
from .file import path_handler, read, write
from .models import RewardModelWrapper, DPORewardModel

def _compute_rewards(reward_model: RewardModelWrapper, completions: List[Tuple[str, str]], compute_on: Optional[List[int]], batch_size: int):
    if compute_on is None:
        compute_on = list(range(len(completions)))
    par = create_loader(compute_on, batch_size)
    output = {}
    for indices in par:
        prompts = [completions[i][0] for i in indices]
        responses = [completions[i][1] for i in indices]
        
        rewards = reward_model.inference(prompts, responses)
        rewards = rewards.tolist()
        for i, k in enumerate(indices):
            output[k] = rewards[i]
    return output

def compute_rewards(reward_model: RewardModelWrapper, completions: Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]], data_tag: str, model_tag: str, compute_on: Optional[List[int]] = None, batch_size: int = 32, k: int = 1):
    workdir = path_handler.get("R", data_tag=data_tag, model_tag=model_tag)
    if k == 1:
        output = _compute_rewards(reward_model, completions, compute_on, batch_size)
        file = f"{workdir}.json"
        write(output, file)
    else:
        outputs = []
        for i in range(k):
            output = _compute_rewards(reward_model, completions[i], compute_on, batch_size)
            outputs.append(output)
        file = f"{workdir}.jsonl"
        write(outputs, file)

def _compute_dpo_rewards(dpo_reward_model: DPORewardModel, completions: List[Tuple[str, str]], compute_on: Optional[List[int]] , batch_size: int, ITR_clip: float):
    if compute_on is None:
        compute_on = list(range(len(completions)))
    par = create_loader(compute_on, batch_size)
    output_ITR: Dict[int, np.ndarray] = {}
    output_ISR: Dict[int, float] = {}
    for indices in par:
        prompts = [completions[i][0] for i in indices]
        responses = [completions[i][1] for i in indices]
        
        dpo_rewards = dpo_reward_model.inference(prompts, responses)
        
        for i, k in enumerate(indices):
            ITR = dpo_rewards[i]
            output_ITR[f"{k}"] = ITR

            ISR = ITR.clip(min=-ITR_clip, max=ITR_clip).sum().item()
            output_ISR[f"{k}"] = ISR
    return output_ITR, output_ISR

def compute_dpo_rewards(dpo_reward_model: DPORewardModel, completions: Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]], data_tag: str, model_tag: str, compute_on: Optional[List[int]] = None, batch_size: int = 16, ITR_clip: float = 1.0, k: int = 1):
    workdir = path_handler.get("ITR", data_tag=data_tag, model_tag=model_tag)
    if k == 1:
        ITR, ISR = _compute_dpo_rewards(dpo_reward_model, completions, compute_on, batch_size, ITR_clip)
        write(ITR, os.path.join(workdir, "ITR.npz"))
        write(ISR, os.path.join(workdir, "ISR.json"))
    else:
        for i in range(k):
            ITR, ISR = _compute_dpo_rewards(dpo_reward_model, completions[i], compute_on, batch_size, ITR_clip)
            write(ITR, os.path.join(workdir, f"{i}", "ITR.npz"))
            write(ISR, os.path.join(workdir, f"{i}", "ISR.json"))

def _extract_values(rewards: Dict[int, float], compute_on: Optional[List[int]]):
    if compute_on is not None:
        rewards = [rewards[i] for i in compute_on]
    else:
        rewards = list(rewards.values())
    rewards: np.ndarray = np.asarray(rewards, dtype=np.float32)
    return rewards

def read_rewards(data_tag: str, model_tag: str, compute_on: Optional[List[int]] = None, k: int = 1):
    workdir = path_handler.get("R", data_tag=data_tag, model_tag=model_tag)
    if k == 1:
        rewards = read(f"{workdir}.json")
        rewards = _extract_values(rewards, compute_on)
    else:
        rewards_list = read(f"{workdir}.jsonl")
        output = []
        for rewards in rewards_list:
            rewards = _extract_values(rewards, compute_on)
            output.append(rewards)
        rewards = np.stack(output, axis=-1)
    return rewards

def _read_dpo_rewards(workdir: str, compute_on: Optional[List[int]] = None):
    data = read(os.path.join(workdir, "ITR.npz"))
    data: Dict[int, np.ndarray] = {int(k): v for k, v in data.items()}
    if compute_on is None:
        ITR = list(data.values())
    else:
        ITR = [data[i] for i in compute_on]

    ISR = read(os.path.join(workdir, "ISR.json"))
    ISR = _extract_values(ISR, compute_on)

    return ITR, ISR

def read_dpo_rewards(data_tag: str, model_tag: str, compute_on: Optional[List[int]] = None, k: int = 1, return_ITR: bool = False):
    workdir = path_handler.get("ITR", data_tag=data_tag, model_tag=model_tag)
    if k == 1:
        output_ITR, output_ISR = _read_dpo_rewards(workdir, compute_on)
    else:
        output_ITR: List[List[np.ndarray]] = []
        output_ISR = []
        for i in range(k):
            ITR, ISR = _read_dpo_rewards(os.path.join(workdir, f"{i}"), compute_on)
            output_ITR.append(ITR)
            output_ISR.append(ISR)
        output_ISR = np.stack(output_ISR, axis=-1)
    if return_ITR:
        return output_ISR, output_ITR
    else:
        return output_ISR
