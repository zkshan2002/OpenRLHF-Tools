from datasets import load_dataset
from math import ceil
import os
from tqdm import tqdm
from typing import Any, List, Dict

from .file import path_handler, read, write


def _download_dataset(data_tag: str):
    if data_tag.startswith("ufb"):
        tag = "HuggingFaceH4/ultrafeedback_binarized"
        dataset = load_dataset(tag)["train_prefs"]
        prompts = [data["chosen"][0]["content"] for data in dataset]
        chosen_responses = [data["chosen"][1]["content"] for data in dataset]
        rejected_responses = [data["rejected"][1]["content"] for data in dataset]
        dataset = [{
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        } for prompt, chosen_response, rejected_response in zip(prompts, chosen_responses, rejected_responses)]
        return dataset
    else:
        raise NotImplementedError

def load_preprocessed_dataset(data_tag: str) -> List[Dict[str, str]]:
    file = path_handler.get("dataset", data_tag=data_tag)
    print(file)
    if os.path.exists(file):
        dataset = read(file)
    else:
        dataset = _download_dataset(data_tag)
        write(dataset, file)
    return dataset

def extract_prompts(dataset: List[Dict[str, str]], prompt_key: str = "prompt", amount: int = 0):
    dataset = dataset[:amount] if amount > 0 else dataset
    prompts = [data[prompt_key] for data in dataset]
    return prompts

def extract_completions(dataset: List[Dict[str, str]], prompt_key: str = "prompt", response_key: str = "response", amount: int = 0):
    dataset = dataset[:amount] if amount > 0 else dataset
    completions = [[data[prompt_key], data[response_key]] for data in dataset]
    return completions

def create_loader(dataset: List[Any], batch_size: int):
    grouped_data = [dataset[i * batch_size:(i + 1) * batch_size] for i in range(int(ceil(len(dataset) / batch_size)))]
    par = tqdm(grouped_data, leave=True)
    return par
