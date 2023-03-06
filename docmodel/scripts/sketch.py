from transformers import LayoutLMv2TokenizerFast, RobertaTokenizerFast
from transformers import BatchEncoding
import torch
import glob
import fire
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import warnings
import unicodedata
from typing import Any

class TokenConversionError(ValueError):
    pass


<<<<<<< HEAD

=======
>>>>>>> origin
LAYOUTLM_V2_TOKENIZER = LayoutLMv2TokenizerFast.from_pretrained(
    "microsoft/layoutlmv2-base-uncased"
)
ROBERTA_TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")

"""def check_str_items(s1: str, s2: str) -> bool:
    if len(s1) != len(s2):
        return
    not_present = []
    for ele in s1:
        if ele not in s2:
            not_present.append(ele)
    print(not_present)
    n = len(s1)
    for i in range(n):
        if s1[i] != s2[i]:
            return False
    
    return True"""


def find_words_in_text(words: list[str], text: str) -> list[tuple[int, int]]:
    """_summary_

    Args:
        words (list[str]): layoutlmv2 words
        text (str): original text

    Returns:
        tuple[int, int]: start and end offsets into the text
    """
    offset = 0
    offsets = []
    lower_case = text.lower()
    stripped_accents = _run_strip_accents(lower_case)
    assert len(stripped_accents) == len(lower_case)
    for word in words:
        start_offset = stripped_accents.find(word.lower(), offset)
        if start_offset == -1:
            raise AssertionError(f"failed to find {word} in text")
        end_offset = start_offset + len(word)
        offsets.append((start_offset, end_offset))
        offset = end_offset
    return offsets


def _run_strip_accents(text: str) -> str:
    """Strips accents from a piece of text."""
    normalized_text = unicodedata.normalize("NFD", text)
    # assert len(normalized_text) == len(text)
    output = []
    for char in normalized_text:
        cat = unicodedata.category(char)
        
        if cat == "Mn":
            print(char)
            print("character is non printable")
            continue
        output.append(char)
    assert len(output) == len(text)
    return "".join(output)


def text_from_layoutlmv2_token_ids(token_ids: list[int]) -> str:
    """
    Make sure to deal with pad tokens

    Args:
        token_ids (list[int]): list of layoutlmv2 token IDs

    Returns:
        str: original document text
    """
    return LAYOUTLM_V2_TOKENIZER.decode(token_ids, skip_special_tokens=True)


def roberta_info_from_text(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """
    TODO: deal with padding tokens
    (is there an argument you can pass to disable this behavior?)

    Args:
        text (str): original text

    Returns:
        list[int]: roberta token ids
        list[tuple[int, int]]: offsets into original text
    """
    encoded: dict = ROBERTA_TOKENIZER(text, return_offsets_mapping=True)
    return encoded["input_ids"], encoded["offset_mapping"]


def layoutlmv2_tokens_from_ids(ids: list[int]) -> list[str]:
    # Deal with padding and "##" symbols in the output tokens

    layoutlmv2_tokens: list = LAYOUTLM_V2_TOKENIZER.convert_ids_to_tokens(
            ids, skip_special_tokens=True
        )

    layoutlmv2_tokens = [t.replace("##", "") for t in layoutlmv2_tokens]
    return layoutlmv2_tokens


def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """
    WARNING: be careful about boundary conditions (e.g. < vs. <=)
    TODO: read this function until you understand it

    Examples:
        a: []
        b: ()

        [ ( ] ): overlap(a=(1, 3), b=(2, 4)) -> True
        [ ( ) ]: overlap(a=(1, 5), b=(2, 4)) -> True
        [  ]( ): overlap(a=(1, 5), b=(5, 6)) -> False

    Args:
        a (tuple[int, int]): start and end offset
        b (tuple[int, int]): start and end offset

    Returns:
        bool: do the two elements overlap
    """
    return a[0] <= b[1] and b[0] <= a[1]


def bbox_from_offset_alignment(
    layoutlmv2_offsets: list[tuple[int, int]],
    roberta_offsets: list[tuple[int, int]],
    layoutlmv2_bbox_info: list[list[int]],
) -> list[list[int]]:
    """
    NOTE: Try first with sample data

    Args:
        layoutlmv2_offests (list[tuple[int, int]]): _description_
        roberta_offests (list[tuple[int, int]]): _description_
        layoutlmv2_bbox_info (list[list[int]]): _description_

    Returns:
        list[list[int]]: _description_
    """
    results = []
    layoutlmv2_search_start = 0
    for offset in roberta_offsets:
        # TODO: make me more efficient by not checking offsets that we
        # know aren't valid and taking advantage of the fact that
        # these two lists are ordered
        if offset == (0, 0):
            # NOTE: this depends on what X-doc suggests
            results.append([0, 0, 0, 0])

            continue

        for idx, layoutlmv2_offset in enumerate(
            layoutlmv2_offsets[layoutlmv2_search_start:]
        ):
            adjusted_idx = idx + layoutlmv2_search_start
            if overlap(offset, layoutlmv2_offset):
                bbox_info = layoutlmv2_bbox_info[adjusted_idx]
                results.append(bbox_info)
                layoutlmv2_search_start = adjusted_idx
                break
        else:
            warnings.warn(f"failed to find a match for {offset}")
            results.append([0, 0, 0, 0])
    return results


def translate_bbox_info(
    text: str,
    layoutlmv2_token_ids: list[int],
    layoutlmv2_bbox_info: list[list[int]],
    roberta_offsets: list[tuple[int, int]],
) -> list[list[int]]:
    """_summary_

    Args:
        text (str): original text
        layoutlmv2_tokens (list[int]): _description_
        layoutlmv2_bbox_info (list[list[int]]): _description_
        roberta_offsets (list[tuple[int, int]]): start and end offsets into the text
    Returns:
        list[list[int]]: roberta bbox info
    """

    try:
        layoutlmv2_tokens: list[str] = layoutlmv2_tokens_from_ids(layoutlmv2_token_ids)
    except TokenConversionError:
        layoutlmv2_tokens: list[str] = layoutlmv2_tokens_from_ids(layoutlmv2_token_ids[0])

    try:
        layoutlmv2_offsets: list[tuple[int, int]] = find_words_in_text(
            layoutlmv2_tokens, text
        )
    except Exception as e:
        raise e
    try:
        roberta_bbox_info: list[list[int]] = bbox_from_offset_alignment(
            layoutlmv2_offsets, roberta_offsets, layoutlmv2_bbox_info
        )
    except Exception():
        raise AssertionError

    return roberta_bbox_info


def translate_to_roberta(
    layoutlmv2_tokens_ids: list[int], layoutlmv2_bbox_info: list[list[int]], text: str
) -> tuple[list[int], list[list[int]]]:
    """

    Args:
        layoutlmv2_tokens (list[int]):
        layoutlmv2_bbox_info (list[list[int]]): _description_

    Returns:
        list[int]: roberta tokens ids
        list[list[int]]: roberta bounding box info
    """
    roberta_tokens_ids, roberta_offsets = roberta_info_from_text(text)
    roberta_bbox = translate_bbox_info(
            text, layoutlmv2_tokens_ids, layoutlmv2_bbox_info, roberta_offsets
        )
    return roberta_tokens_ids, roberta_bbox


def get_text_from_file(unique_id: str, page_num: int, split: str) -> str:

    file_name = f"docrep-tiny2/{split}/{unique_id}-{page_num}.txt"
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path) as f:
        text = f.read()
    return text


def get_file_meta_from_file_path(file_path: str) -> dict[str, str]:
    splits = ["train", "test", "valid"]
    file_parts = file_path.split("/")
    data_split = None
    for split in splits:
        if split in file_parts:
            data_split = split
            break
    unique_id, rest = os.path.basename(file_path).split("-")
    page_num = rest.split(".")[0]
    return {"unique_id": unique_id, "page_num": page_num, "data_split": data_split}

def standardize_data(data):
    for key in data.keys():
        if len(data[key]) == 1:
            flattened = torch.flatten(data[key])
            data[key] = flattened
    return data

def ignore_file(data: dict[str, Any])-> bool:
    return isinstance(data['input_ids'], list) 

def convert_to_new_dataset(old_filepath: str, new_filepath: str, override=False):
    """
    New dataset contains:
    - RoBERTa token IDs
    - bbox info per RoBERTa token ID
    - image

    Args:
        old_filepath (str): filepath of layoutlmv2 dataset file
        new_filepath (str): filepath of roberta-based dataset file
    """
    if os.path.exists(new_filepath) and not override:
        return

    data = torch.load(old_filepath)
    
    if ignore_file(data):
        return
    
    meta = get_file_meta_from_file_path(old_filepath)
    text = get_text_from_file(meta["unique_id"], meta["page_num"], meta["data_split"])
    

    roberta_tokens, roberta_bbox_info = translate_to_roberta(
        data['input_ids'][0], data['bbox'][0], text=text
        )
    torch.save(
        {
            "input_ids": roberta_tokens,
            "bbox": roberta_bbox_info,
            "image": data["image"],
        },
        new_filepath,
    )


def main(
    pattern: str, dest_folder: str = "Roberta_files", n_cores: int = 1, split="test"
):
    pool = ThreadPoolExecutor(n_cores)
    sample_files = glob.glob(pattern, recursive=True) * 12
    start = time.time()
    futures = {}
    destination_folder = os.path.join(
        os.path.dirname(__file__), f"{dest_folder}/{split}"
    )
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    for file in sample_files:
        future = pool.submit(
            convert_to_new_dataset,
            file,
            f"{destination_folder}/{os.path.basename(file)}",
            override=True,
        )
        futures[future] = file

    for future in as_completed(futures):
        future.result()
        file = futures[future]
        print(f"Completed processing {file}")

    end = time.time()
    total_time = end - start
    avg_time = total_time / len(sample_files)
    full_dataset_time = (avg_time * 40_000_000) / 3600
    print(f"Total time: {total_time:.2f}")
    print(f"Average time: {avg_time:.2f}")
    print(f"Estimated time for 40M files: {full_dataset_time:0.2f} hours")


if __name__ == "__main__":
    fire.Fire(main)
