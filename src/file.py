import json
import jsonlines
import numpy as np
import os
from typing import Any


def _rm_dir(dir: str):
    if not os.path.isdir(dir):
        return
    for name in os.listdir(dir):
        entry = os.path.join(dir, name)
        if os.path.isfile(entry):
            os.remove(entry)
        elif os.path.isdir(entry):
            _rm_dir(entry)
    os.rmdir(dir)

def _make_parent_dir(file: str):
    os.makedirs(os.path.dirname(file), exist_ok=True)

def read(file: str):
    if file.endswith(".json"):
        with open(file, "r") as f:
            data = json.load(f)
    elif file.endswith(".jsonl"):
        data = []    
        with jsonlines.open(file, "r") as reader:
            for obj in reader:
                data.append(obj)
    elif file.endswith(".npy"):
        data = np.load(file)
    elif file.endswith(".npz"):
        data = np.load(file)
        data = {key: data[key] for key in data}
    else:
        raise NotImplementedError
    return data

def write(data: Any, file: str):
    _make_parent_dir(file)
    if file.endswith(".json"):
        with open(file, "w") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl"):
        with jsonlines.open(file, mode="w") as writer:
            writer.write_all(data)
    elif file.endswith(".npy"):
        np.save(file, data)
    elif file.endswith(".npz"):
        np.savez(file, **data)
    else:
        raise NotImplementedError

class PathHandler:
    def __init__(self, root: str):
        self._root = os.path.realpath(root)
    
    def get(self, mode: str, **kwargs):
        if mode == "dataset":
            return self._get_dataset(**kwargs)
        elif mode == "generation":
            return self._get_generation(**kwargs)
        elif mode in ["R", "ITR"]:
            return self._get_reward(mode, **kwargs)
        elif mode == "temp":
            return self._get_temp(mode, **kwargs)
        else:
            assert False

    def _get_dataset(self, data_tag: str):
        assert isinstance(data_tag, str)
        return os.path.join(self._root, "preprocessed_datasets", f"{data_tag}.jsonl")

    def _get_generation(self, data_tag: str, model_tag: str):
        assert isinstance(data_tag, str)
        assert isinstance(model_tag, str)
        return os.path.join(self._root, "generations", data_tag, f"{model_tag}.jsonl")

    def _get_reward(self, mode: str, data_tag: str, model_tag: str):
        assert isinstance(data_tag, str)
        assert isinstance(model_tag, str)
        return os.path.join(self._root, "rewards", mode, data_tag, model_tag)

    def _get_temp(self, file_name: str):
        assert isinstance(file_name, str)
        return os.path.join(self._root, "temp", file_name)

project_workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_handler = PathHandler(os.path.join(project_workdir, ".cache"))
