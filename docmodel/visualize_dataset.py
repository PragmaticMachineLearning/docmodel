from dataset import DocModelDataset
import fire
from transformers import RobertaTokenizerFast
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json


MODEL_CONFIG = {
    "doc-model-roberta": {
        "dataset": DocModelDataset,
        "max_length": 2048,
        "tokenizer": RobertaTokenizerFast.from_pretrained,
        "tokenizer_kwargs": {"pretrained_model_name_or_path": "roberta-base", "local_files_only":True},
    },
}
def word_freq(freq_file):

    with open(freq_file, 'r') as f:
        
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


WORD_FREQ = word_freq("word_freq.json")


def word_freq_per_example(text):
    
    words = text.split()

    if not words: 
        return 0

    avg = 0
    for word in words:
        if word not in WORD_FREQ:
            print(f"{word.upper()} IS MISSING!")
            avg += 1/len(words)
        else:
            avg += WORD_FREQ[word] / len(words)
    
    return avg


SCORE_FNS = {
 "avg-word-length": avg_word_length,   
 "redundancy": redundancy,
 "word_frequency": word_freq_per_example
}
    
def main(
    data_dir=None,
    base_model="doc-model-roberta",
    n = 5,
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

    scores_by_score_fn = defaultdict(list)
    for score_fn_name, score_fn in SCORE_FNS.items():
        for example_idx in tqdm(list(range(len(dataset)))):
            example = dataset[example_idx]
            text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            score = score_fn(text)
            scores_by_score_fn[score_fn_name].append(score)
       
    for score_fn_name, scores in scores_by_score_fn.items():
        print("SCORE FUNCTION: ", score_fn_name)
        # Plot data distribution
        plt.hist(scores, bins=100)
        plt.title(score_fn_name)
        plt.savefig(f"{score_fn_name}.png")
        plt.clf()


        num_scores = len(scores)
        sorted_score_idxs = np.argsort(scores)
        top_n_scores = reversed(sorted_score_idxs[-n:])
        bottom_n_scores = sorted_score_idxs[:n]

        # Print bottom 5
        print(f"Bottom {n} {score_fn_name}")
        for rank, idx in enumerate(bottom_n_scores):
            example = dataset[idx]
            text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            print(f"Score: {scores[idx]}, Percentile: {(rank / num_scores) * 100:.3f}")
            print(text)
            print()
        
        # Print top 5
        print(f"Top {n} {score_fn_name}")
        for rank, idx in enumerate(top_n_scores):
            example = dataset[idx]
            text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            print(f"Score: {scores[idx]}, Percentile: {((num_scores - rank) / num_scores) * 100:.3f}")
            print(text)
            print()


if __name__ == "__main__":
    fire.Fire(main)
