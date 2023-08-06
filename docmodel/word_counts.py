import json
from collections import defaultdict

import fire
import tqdm
from transformers import RobertaTokenizerFast
from concurrent.futures import ProcessPoolExecutor
from dataset import DocModelDataset

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import tqdm


def bounded_executor(
    executor,
    func,
    keyword_arguments,
    bound,
):
    """
    Utility for parallel programming that limits the number of concurrent tasks
    so that, if it requires significant memory to store the inputs / outputs of the tasks,
    we don't run out of memory.
    """
    pbar = tqdm.tqdm(total=len(keyword_arguments))
    futures = {}

    for kwargs in keyword_arguments:
        if len(futures) >= bound:
            # Wait for some futures to complete before submitting new ones
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for completed_future in done:
                input_args = futures.pop(completed_future)
                yield {"inputs": input_args, "results": completed_future.result()}
                pbar.update(1)

        future = executor.submit(func, **kwargs)
        futures[
            future
        ] = kwargs  # Keep a reference to the future in a dict so we can pop it later

    # Wait for the remaining futures to complete
    remaining, _ = concurrent.futures.wait(futures)
    for future in remaining:
        input_args = futures.pop(future)
        yield {"inputs": input_args, "results": future.result()}
        pbar.update(1)


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


def word_counts(text: str):
    counts = defaultdict(int)
    words = text.split()

    for word in words:
        counts[word] += 1

    output = dict(counts)
    return output


def word_count_from_idx(dataset, idx, tokenizer):
    example = dataset[idx]
    example_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    word_count = word_counts(example_text)
    return word_count


def main(
    data_dir: str = None, base_model: str = "doc-model-roberta", output_file: str = None
) -> None:
    model_config = MODEL_CONFIG[base_model]

    tokenizer = model_config["tokenizer"](
        model_max_length=model_config["max_length"], **model_config["tokenizer_kwargs"]
    )

    dataset_cls = model_config["dataset"]
    pool = ProcessPoolExecutor(max_workers=4)
    splits = ["train"]
    for split in splits:
        print("Indexing dataset...")
        dataset = dataset_cls(
            directory=data_dir,
            split=split,
            max_length=model_config["max_length"],
            reading_order="default",
        )
        print("Finished indexing")
        global_word_count: dict[str, int] = defaultdict(int)
        print("Computing word counts...")

        for result in bounded_executor(
            executor=pool,
            func=word_count_from_idx,
            keyword_arguments=[
                {"dataset": dataset, "idx": idx, "tokenizer": tokenizer}
                for idx in range(len(dataset))
            ],
            bound=100,
        ):
            for word, count in result["results"].items():
                global_word_count[word] += count

        output = f"{output_file}-{split}.json"
        with open(output, "w") as f:
            json.dump(dict(global_word_count), f)


if __name__ == "__main__":
    fire.Fire(main)
