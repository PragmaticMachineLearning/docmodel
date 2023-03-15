from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Union
import random
import warnings
import torch


def visualize_inputs(tokenizer, result):
    for i in range(len(result["input_ids"])):
        input_text = tokenizer.decode(result["input_ids"][i])
        print(f"\nMasked Input\n{input_text}")
        label_mask = result["labels"][i] != -100
        result["input_ids"][i][label_mask] = result["labels"][i][label_mask]
        expected = tokenizer.decode(result["input_ids"][i])
        print(
            f"\nUnmasked Target\n{expected}",
        )
        print("\n--------------")


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)
    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (
        pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
    ):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    if len(examples[0].shape) == 1:
        # Maybe we need to change this to 0 for bounding box?
        result = examples[0].new_full(
            [len(examples), max_length], tokenizer.pad_token_id
        )
    elif len(examples[0].shape) == 2:
        result = examples[0].new_full(
            [len(examples), max_length, examples[0].shape[1]], tokenizer.pad_token_id
        )

    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[
                i,
                -example.shape[0] :,
            ] = example
    return result


@dataclass
class DataCollatorForWholeWordMask:
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    .. note::

        This collator relies on details of the implementation of subword tokenization by
        :class:`~transformers.BertTokenizer`, specifically that subword tokens are prefixed with `##`. For tokenizers
        that do not adhere to this scheme, this collator will produce an output that is roughly equivalent to
        :class:`.DataCollatorForLanguageModeling`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    position_mask_probability: float = 0.0
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    include_2d_data: Optional[bool] = True

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        input_ids = [e["input_ids"] for e in examples]
        bbox = [e["bbox"] for e in examples]

        batch_input = _torch_collate_batch(
            input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch_bbox = _torch_collate_batch(
            bbox, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )

        mask_labels = []
        mask_position_idxs = []
        for e in examples:
            ref_tokens = []
            for id in e["input_ids"].tolist():
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)
            mask_labels.append(self._whole_word_mask(ref_tokens))
            mask_position_idxs.append(
                self._whole_word_mask(ref_tokens, proba=self.position_mask_probability)
            )
        batch_mask = _torch_collate_batch(
            mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch_position_mask = _torch_collate_batch(
            mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        inputs, labels, attention_mask = self.torch_mask_tokens(batch_input, batch_mask)

        if self.position_mask_probability > 0.0:
            print(f"POSITION MASK PROBA: {self.position_mask_probability}")
            print(
                "WARNING: PLEASE CHECK THIS CODE BEFORE USING IT -- IT PROBABLY NEEDS A SEPARATE MASK"
            )
            bbox_inputs, bbox_labels = self.torch_mask_positions(
                inputs=batch_bbox, tokens=batch_input, mask_labels=batch_position_mask
            )
        else:
            bbox_inputs, bbox_labels = batch_bbox, batch_bbox

        result = {
            "input_ids": inputs,
            "labels": labels,
            "bbox": bbox_inputs,
            "bbox_labels": bbox_labels,
            "attention_mask": attention_mask,
        }

        if not self.include_2d_data:
            del result["bbox"]
            del result["bbox_labels"]

        # print(
        #     "Input IDs",
        #     torch.min(result["input_ids"]).item(),
        #     torch.max(result["input_ids"]).item(),
        # )
        # print(
        #     "BBox", torch.min(result["bbox"]).item(), torch.max(result["bbox"]).item()
        # )

        # visualize_inputs(self.tokenizer, result)

        return result

    def _whole_word_mask(
        self, input_tokens: List[str], max_percentage: float = 0.2, proba: float = 0.15
    ):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy

        Works for both whole word masking and position masking.
        """
        proba = self.mlm_probability if proba is None else proba
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        max_predictions = int(len(input_tokens) * max_percentage)
        num_to_predict = min(
            max_predictions,
            max(1, int(round(len(input_tokens)))),
        )
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))
        ]
        return mask_labels

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()

        # Attention mask is 0 only where padding is present
        attention_mask = torch.ones_like(masked_indices, dtype=torch.float32).float()
        attention_mask.masked_fill_(padding_mask, value=0.0)

        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, attention_mask

    def torch_mask_positions(
        self,
        inputs: Any,
        tokens: Any,
        mask_labels: Any,
        position_mask_value: int = 1000,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked positions for masked language modeling
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        # Need token values to determine special tokens mask
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in tokens.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        padding_mask = tokens.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        inputs[masked_indices] = position_mask_value

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def test_collator(tokenizer):
    # Handles masking out words for the MLM loss
    # Collate examples into a batch

    collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, pad_to_multiple_of=64)
    test_inputs = [
        {
            "input_ids": np.asarray([0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 2, 1, 1, 1, 1]),
            "bbox": np.random.randint(low=0, high=1000, size=[16, 4]),
        },
        {
            "input_ids": np.asarray([0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 2]),
            "bbox": np.random.randint(low=0, high=1000, size=[14, 4]),
        },
    ]
    output = collator(test_inputs)
    print(output)


if __name__ == "__main__":
    from transformers import RobertaTokenizerFast
    import numpy as np

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    test_collator(tokenizer)
