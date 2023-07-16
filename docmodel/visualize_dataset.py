import json
from collections import defaultdict
from typing import Any

import fire
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from dataset import DocModelDataset
from filtering_fns import redundancy, avg_word_length, word_freq_per_example


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


SCORE_FNS = {
    "avg-word-length": avg_word_length,
    "redundancy": redundancy,
    "word_frequency": word_freq_per_example,
}


def main(
    data_dir=None,
    base_model: str="doc-model-roberta",
    n: int=5,

    
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
