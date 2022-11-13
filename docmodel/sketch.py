from re import L
from transformers import LayoutLMv2TokenizerFast, RobertaTokenizerFast
from transformers import BatchEncoding
import torch

LAYOUTLM_V2_TOKENIZER = LayoutLMv2TokenizerFast.from_pretrained(
    "microsoft/layoutlmv2-base-uncased"
)
ROBERTA_TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")


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
    for word in words:
        start_offset = text.find(word, offset)
        end_offset = len(word)
        offsets.append((start_offset, end_offset))
        offset = end_offset
    return offsets


def text_from_layoutlmv2_token_ids(token_ids: list[int]) -> str:
    """
    Make sure to deal with pad tokens

    Args:
        token_ids (list[int]): list of layoutlmv2 token IDs

    Returns:
        str: original document text
    """
    return LAYOUTLM_V2_TOKENIZER.decode(token_ids)


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
    layoutlmv2_tokens = LAYOUTLM_V2_TOKENIZER.convert_ids_to_tokens(ids)
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
    return a[0] < b[1] and b[0] < a[1]


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
    for offset in roberta_offsets:
        # TODO: make me more efficient by not checking offsets that we
        # know aren't valid and taking advantage of the fact that
        # these two lists are ordered
        for idx, layoutlmv2_offset in enumerate(layoutlmv2_offsets):
            if overlap(offset, layoutlmv2_offset):
                bbox_info = layoutlmv2_bbox_info[idx]
                results.append(bbox_info)
                break
        else:
            raise AssertionError("Failed to find a match -- this should never happen")
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
        roberta_tokens (list[int]): _description_
        roberta_offsets (list[tuple[int, int]]): start and end offsets into the text
    Returns:
        list[list[int]]: roberta bbox info
    """

    layoutlmv2_tokens: list[str] = layoutlmv2_tokens_from_ids(layoutlmv2_token_ids)
    layoutlmv2_offsets: list[tuple[int, int]] = find_words_in_text(
        layoutlmv2_tokens, text
    )
    roberta_bbox_info: list[list[int]] = bbox_from_offset_alignment(
        layoutlmv2_offsets, roberta_offsets, layoutlmv2_bbox_info
    )
    return roberta_bbox_info


def translate_to_roberta(
    layoutlmv2_tokens: list[int], layoutlmv2_bbox_info: list[list[int]]
) -> tuple[list[int], list[list[int]]]:
    """

    Args:
        layoutlmv2_tokens (list[int]):
        layoutlmv2_bbox_info (list[list[int]]): _description_

    Returns:
        list[int]: roberta tokens
        list[list[int]]: roberta bounding box info
    """
    text = text_from_layoutlmv2_token_ids(layoutlmv2_tokens)
    roberta_tokens, roberta_offsets = roberta_info_from_text(text)
    roberta_bbox = translate_bbox_info(
        text, layoutlmv2_tokens, layoutlmv2_bbox_info, roberta_offsets
    )
    return roberta_tokens, roberta_bbox


def convert_to_new_dataset(old_filepath: str, new_filepath: str):
    """
    New dataset contains:
    - RoBERTa token IDs
    - bbox info per RoBERTa token ID
    - image

    Args:
        old_filepath (str): filepath of layoutlmv2 dataset file
        new_filepath (str): filepath of roberta-based dataset file
    """
    data = torch.load(old_filepath)
    roberta_tokens, roberta_bbox_info = translate_to_roberta(
        data["input_ids"], data["bbox"]
    )
    torch.save(
        {
            "input_ids": roberta_tokens,
            "bbox": roberta_bbox_info,
            "image": data["image"],
        },
        new_filepath,
    )


if __name__ == "__main__":
    convert_to_new_dataset(
        "docrep-tiny/test/bf5852f352d070465fa0aac4e3f8373d2c04844de4e4c82feece5325-0.pt",
        "sample-roberta-output.pt",
    )
