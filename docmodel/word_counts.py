import json
from collections import defaultdict

import fire
import tqdm
from transformers import RobertaTokenizerFast

from dataset import DocModelDataset

MODEL_CONFIG = {
    "doc-model-roberta": {
        "dataset": DocModelDataset,
        "max_length": 2048,
        "tokenizer": RobertaTokenizerFast.from_pretrained,
        "tokenizer_kwargs": {"pretrained_model_name_or_path": "roberta-base", "local_files_only":True},
    },
}


def word_counts(text: str):
    
    counts = defaultdict(int)
    words = text.split()

    for word in words:
        counts[word] += 1
    
    output = dict(counts)
    return output


def main(
    data_dir: str = None,
    base_model: str="doc-model-roberta",
    output_file: str = None
) -> None:
    
    model_config = MODEL_CONFIG[base_model]
    
    tokenizer = model_config["tokenizer"](
        model_max_length=model_config["max_length"], **model_config["tokenizer_kwargs"]
    )

    dataset_cls = model_config["dataset"]
   
    splits = ['train', 'test', 'valid']
    for split in splits:
        dataset = dataset_cls(
            directory=data_dir,
            split=split,
            max_length=model_config["max_length"],
            reading_order="default")
        

        global_word_count: dict[str, int] = defaultdict(int)
        
        for example in dataset:
            example_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            word_count = word_counts(example_text)
            for word, count in word_count.items():
                global_word_count[word] += count

        output = f"{output_file}-{split}.json"
        with open(output, "w") as f:
            json.dump(dict(global_word_count), f)


if __name__ == "__main__":
    fire.Fire(main)