import json
import time
from typing import Any, Callable

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from dataset import DocModelDataset
from visualize_dataset import redundancy, avg_word_length, WORD_FREQ


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



def timeit(f):
    def modified_f(*args, **kwargs):
        start = time.time()
        output = f(*args, **kwargs)
        end = time.time()
        print(f"Time to run {f.__name__}: {end - start:.3f}")
        return output

    return modified_f


@timeit
def word_freq_per_example(text: str, num_examples: int) -> float:
    
    words: list[str] = text.split()

    if not words:
        return 0

    total_words = len(words)
    avg_word_freq = (
        sum(WORD_FREQ[word] for word in words if word in WORD_FREQ) / total_words
    )
    missing_words: list[str] = [word for word in words if word not in WORD_FREQ]
    if missing_words:
        print(f"MISSING WORDS: {''.join(missing_words)}")
    avg_word_freq += len(missing_words) / total_words

    return avg_word_freq / num_examples




SCORE_FUNCTIONS: dict[str, Callable] = {
    "avg_word_length": avg_word_length,
    "redundancy": redundancy,
    "word_frequency": word_freq_per_example,
}

SCORE_THRESHOLDS: dict[str, list[tuple[float, float]]] = {
    # A dictionary of dummy thresholds to test the filtering function
    "avg_word_length": [(1.5, 10.0)],
    "redundancy": [(0.1, 5.0)],
    "word_frequency": [(2.0, 4.0)],
}


def filter_dataset_by_metrics(
    example, text: str, num_examples = None
) -> list[dict[str, float]]:
    
    filtered_dataset: list[dict[str, float]] = []
    should_include = True
    for metric, thresholds in SCORE_THRESHOLDS.items():
        score_fn = SCORE_FUNCTIONS[metric]
        score = score_fn(text, num_examples) if metric == "word_frequency" else score_fn(text)

        if not any(
            lower_threshold <= score <= upper_threshold
            for lower_threshold, upper_threshold in thresholds
        ):
            should_include = False
            break

    if should_include:
        filtered_example: dict[str, float] = {"input_ids": example["input_ids"]}
        filtered_dataset.append(filtered_example)

    return filtered_dataset


def main(
    data_dir=None,
    base_model: str = "doc-model-roberta",
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
    num_examples = len(dataset)

    filtered_texts = []
    unfiltered_text = []
    for example in dataset:
        text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        filtered_dataset = filter_dataset_by_metrics(example, text, num_examples)
        for ex in filtered_dataset:
            filtered_text = tokenizer.decode(ex['input_ids'], skip_special_tokens = True)        
            filtered_texts.append(filtered_text)

        unfiltered_text.append(text)

    output = {"FILTERED-TEXT": filtered_texts, "UNFILTERED-TEXT": unfiltered_text}
    out_path = f"{data_dir}/filter_comparison.json"

    with open(out_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    fire.Fire(main)
