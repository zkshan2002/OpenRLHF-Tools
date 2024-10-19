from typing import List, Dict

from .data import create_loader, extract_completions
from .file import path_handler, read, write
from .model_wrappers import ActorWrapper

def play_with_actor(actor: ActorWrapper):
    def gen(prompt: str):
        response = actor.generate([prompt])[0]
        return response
    return gen

def generate(actor: ActorWrapper, prompts: List[str], data_tag: str, model_tag: str, batch_size: int = 32, k: int = 1):
    par = create_loader(prompts, batch_size)

    output_dataset = []
    for prompts in par:
        if k == 1:
            responses = actor.generate(prompts)
            output_dataset.extend([{
                "prompt": prompt,
                "response": response,
            } for prompt, response in zip(prompts, responses)])
        else:
            output = [{"prompt": prompt} for prompt in prompts]
            for i in range(k):
                responses = actor.generate(prompts)
                output = [{
                    **prev_output,
                    f"response-{i}": response,
                } for response, prev_output in zip(responses, output)]
            output_dataset.extend(output)
    
    file = path_handler.get("generation", data_tag=data_tag, model_tag=model_tag, k=k)
    write(output_dataset, file)

def generate_preference(dataset: List[Dict[str, str]], data_tag: str, chosen_key: str = "chosen", rejected_key: str = "rejected"):
    for key in [chosen_key, rejected_key]:
        output_dataset = [{
            "prompt": data["prompt"],
            "response": data[key] + "<|eot_id|>",
        } for data in dataset[:1000]
        ]
        file = path_handler.get("generation", data_tag=data_tag, model_tag=key)
        write(output_dataset, file)

def read_generation(data_tag: str, model_tag: str, k: int = 1):
    file = path_handler.get("generation", data_tag=data_tag, model_tag=model_tag, k=k)
    dataset = read(file)
    if k == 1:
        completions = extract_completions(dataset)
        return completions
    else:
        k_completions = []
        for i in range(k):
            completions = extract_completions(dataset, response_key=f"response-{i}")
            k_completions.append(completions)
        return k_completions
