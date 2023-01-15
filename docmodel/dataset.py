from base64 import encode
import glob
import io
import os
import random

import torch
import PIL
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset

# Stackoverflow magic number -- not sure if this is the absolute max or not
PIL.Image.MAX_IMAGE_PIXELS = 933120000

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)



def preprocess(page, max_length=512, chunk_overlap=True, shrink_dtype=True):
    stride = max_length // 3 if (chunk_overlap and max_length) else 0
    truncation = False if max_length is None else True
    add_special_tokens = truncation

    encoded_inputs = tokenizer(
        page["tokens"],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        return_overflowing_tokens=True,
        stride=stride,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    bbox = page["boxes"]

    bbox_inputs = []
    previous_word_idx = None
    for word_idx in encoded_inputs.word_ids():
        # Special tokens don't have position info
        if word_idx is None:
            bbox_inputs.append([0, 0, 0, 0])
        # We set the label for the first token of each word.
        elif word_idx != previous_word_idx:
            bbox_inputs.append(bbox[word_idx])
        # For the other tokens in a word, we set the label to either the current label or -100, depending on
        # the label_all_tokens flag.
        else:
            bbox_inputs.append(bbox[word_idx])
        previous_word_idx = word_idx

    encoded_inputs["bbox"] = torch.tensor(bbox_inputs)

    if shrink_dtype:
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"].type(torch.int16)
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].type(
            torch.bool
        )
        encoded_inputs["bbox"] = encoded_inputs["bbox"].type(torch.int16)

    if truncation:
        assert encoded_inputs.input_ids.shape[-1:] == torch.Size([512])
        assert encoded_inputs.attention_mask.shape[-1:] == torch.Size([512])
        assert encoded_inputs.bbox.shape[-2:] == torch.Size([512, 4])

    return encoded_inputs

class DocModelDataset(Dataset):
    def __init__(
        self,
        directory,
        split="train",
        max_length=2048,
        per_chunk_length=512,
        dataset_size=None,
        stride=None,
        reading_order="default",
        seed=42,
    ):
        if isinstance(directory, str):
            self.filepaths = list(
                glob.glob(os.path.join(directory, split, "**", "*.pt"), recursive=True)
            )
        elif isinstance(directory, list):
            self.filepaths = []
            for dir in directory:
                self.filepaths += list(
                    glob.glob(os.path.join(dir, split, "**", "*.pt"), recursive=True)
                )
        random.seed(seed)
        random.shuffle(self.filepaths)
        if dataset_size is not None:
            self.filepaths = self.filepaths[:dataset_size]
        self.max_length = max_length
        self.stride = stride or self.max_length // 3
        self.num_special_tokens = tokenizer.num_special_tokens_to_add()
        self.reading_order = reading_order
        self.per_chunk_length = per_chunk_length
        self.seen_filepaths = []

    def __len__(self):
        return len(self.filepaths)

    def convert_dtype(self, encoded_inputs):
        # Accidentally overflowed int16
        encoded_inputs["input_ids"] = (
            encoded_inputs["input_ids"].type(torch.int32).squeeze(0)
        )
        correction = torch.iinfo(torch.int16).max * 2 + 2
        encoded_inputs["input_ids"] = torch.where(
            encoded_inputs["input_ids"] < 0,
            encoded_inputs["input_ids"] + correction,
            encoded_inputs["input_ids"],
        )
        assert (encoded_inputs["input_ids"] >= 0).all()
        encoded_inputs["bbox"] = encoded_inputs["bbox"].type(torch.int32).squeeze(0)
        # Occasionally OCR system returns a value outside of page bounds
        encoded_inputs["bbox"] = torch.clamp(encoded_inputs["bbox"], 0, 1000).type(
            torch.int32
        )
        return encoded_inputs

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        self.seen_filepaths.append(filepath)
        encoded_inputs = torch.load(filepath)
        encoded_inputs = self.convert_dtype(encoded_inputs)
        input_mask = encoded_inputs["input_ids"] != 0

        # Empty pages represented as empty pad token
        # TODO: fix upstream or find alternate solution
        input_mask[0] = True

        # Should ideally fix upstream
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"][input_mask]
        encoded_inputs["bbox"] = encoded_inputs["bbox"][input_mask]

        tokens_per_chunk = self.max_length - self.num_special_tokens
        candidate_start_indices = [0] + list(
            range(
                0,
                len(encoded_inputs["input_ids"]) - tokens_per_chunk,
                self.stride,
            )
        )
        start_index = random.choice(candidate_start_indices)
        end_index = start_index + tokens_per_chunk

        # Find first pad token and truncate here since we're re-padding
        first_pad_idx = (encoded_inputs["input_ids"] == 1).nonzero()
        if first_pad_idx.size(0) > 0 and first_pad_idx[0] < end_index:
            end_index = first_pad_idx[0]

        # This probably drops our bbox info
        # Just needs to handle padding / truncation / attention mask?
        # Does not need to handle converting from words to tokens like the LayoutLMv2 version
        # since we have already dealt with that during download
        sliced_input = tokenizer.prepare_for_model(
            ids=encoded_inputs["input_ids"][start_index:end_index].tolist(),
            # Should never really matter -- we have manually taken care of
            max_length=self.max_length,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        bbox = torch.tensor(encoded_inputs["bbox"][start_index:end_index])
        # Add bbox info for start and end chars
        bbox = torch.cat(
            (
                torch.zeros(1, 4, dtype=torch.int32),
                bbox,
                torch.zeros(1, 4, dtype=torch.int32),
            ),
            dim=0,
        )
        sliced_input["bbox"] = bbox
        assert (
            bbox.shape[0] == sliced_input["input_ids"].shape[0]
        ), f"{bbox.shape} {sliced_input['input_ids'].shape}\n{sliced_input['input_ids']}"
        sliced_input["attention_mask"] = sliced_input["attention_mask"].type(
            torch.float32
        )
        # sliced_input['filename'] = filepath
        return sliced_input

    def save(self): 
        pass


if __name__ == "__main__":
    dataset = DocModelDataset(
        directory="~/doc-model/docmodel/document-data",
    )
    for i in range(len(dataset)):
        dataset[i]
