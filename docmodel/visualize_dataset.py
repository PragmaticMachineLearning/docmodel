import json
import time
from collections import defaultdict
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from dataset import DocModelDataset

MODEL_CONFIG = {
    "doc-model-roberta": {
        "dataset": DocModelDataset,
        "max_length": 2048,
        "tokenizer": RobertaTokenizerFast.from_pretrained,
        "tokenizer_kwargs": {
            "pretrained_model_name_or_path": "roberta-base",
            "local_files_only": True,
        },
    },
}


def load_word_freq(freq_file):
    with open(freq_file, "r") as f:
        data = json.load(f)
    return data


def redundancy(text):
    words = text.split()
    if not words:
        return 0
    return len(words) / len(set(words))


def avg_word_length(text):
    words = text.split()
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)


WORD_FREQ = load_word_freq("word_freq.json")

def timeit(f):
    def modified_f(*args, **kwargs):
        start = time.time()
        output = f(*args, **kwargs)
        end = time.time()
        print(f"Time to run {f.__name__}: {end - start:.3f}")
        return output
    return modified_f


@timeit
def word_freq_per_example(text: str):
    words: list[str] = text.split()

    if not words:
        return 0

    total_words = len(words)
    avg_word_freq = sum(
        WORD_FREQ[word] for word in words if word in WORD_FREQ
    )/total_words
    missing_words: list[str] = [word for word in words if word not in WORD_FREQ]
    if missing_words:
        print(f"MISSING WORDS: {''.join(missing_words)}")
    avg_word_freq += len(missing_words) / total_words
    return avg_word_freq

# What if we had a function that:
# - Takes in a function
# - Returns a function that does everything the original function did
#   but also prints the time it took to run


# def filter_dataset_by_frequency(
#     dataset: list[dict[str, int]],
#     tokenizer: str,
#     frequency_threshold: int,
# ) -> list[dict[str, int]]:
    
#     filtered_dataset: list[dict[str, int]] = []

#     for example in dataset:
#         text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
#         words: list[str] = text.split()

#         filtered_words: list[str] = [
#             word
#             for word in words
#             if word in WORD_FREQ and WORD_FREQ[word] >= frequency_threshold
#         ]
#         filtered_text = " ".join(filtered_words)

#         filtered_example: dict[str, int] = {
#             "input_ids": tokenizer.encode(
#                 filtered_text, add_special_tokens=True, truncation=True
#             )
#         }

#         filtered_dataset.append(filtered_example)

#     return filtered_dataset


SCORE_FNS = {
    "avg-word-length": avg_word_length,
    "redundancy": redundancy,
    "word_frequency": word_freq_per_example,
}


def main(
    data_dir=None,
    base_model: str="doc-model-roberta",
    n: int=5,
    frequency_threshold: int=3,

    
):
    model_config = MODEL_CONFIG[base_model]
    tokenizer = model_config["tokenizer"](
        model_max_length=model_config["max_length"], **model_config["tokenizer_kwargs"]
    )
    dataset_cls = model_config["dataset"]
    dataset = dataset_cls(
        directory=data_dir,
        split="train",
        max_length=model_config["max_length"],
        reading_order="default",
    )
    
    texts_array: list[str] = []
    scores_by_score_fn: dict[str, list[int]] = defaultdict(list)

    for example_idx in tqdm(list(range(len(dataset)))):
        example = dataset[example_idx]
        text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        texts_array.append(text)
        for score_fn_name, score_fn in SCORE_FNS.items():
            if score_fn_name == "word_frequency":
                score = score_fn(text)/len(dataset)
            else:
                score = score_fn(text)
            scores_by_score_fn[score_fn_name].append(score)
    # Save the score info to a JSON (make sure you know which dataset the JSON came from)
    output_path = f"{data_dir}/word_analysis_scores.json"
    with open(output_path, 'w') as f:
        json.dump(dict(scores_by_score_fn), f)


    dataframe: dict[str, list[Any]] = {'text': texts_array, **scores_by_score_fn}
    dataframe = pd.DataFrame(dataframe)
    dataframe.to_csv('scores_by_score_fn.csv')
    
    for score_fn_name, scores in scores_by_score_fn.items():
        print("SCORE FUNCTION: ", score_fn_name)
        # Plot data distribution
        plt.hist(scores, bins=100)
        plt.xlabel(score_fn_name)
        plt.ylabel("count")
        plt.title(score_fn_name)
        plt.savefig(f"{score_fn_name}.png")
        plt.clf()

if __name__ == "__main__":
    fire.Fire(main)
