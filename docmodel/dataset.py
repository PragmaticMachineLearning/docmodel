import glob
import io
import os
import random

import torch
import PIL
from transformers.models.layoutlmv2 import LayoutLMv2Processor
from PIL import Image
from torch.utils.data import Dataset

from docmodel.etl_utils import use_reading_order
from docmodel.tokenization_layoutlmv2 import CustomLayoutLMv2Tokenizer

# Stackoverflow magic number -- not sure if this is the absolute max or not
PIL.Image.MAX_IMAGE_PIXELS = 933120000

processor = LayoutLMv2Processor.from_pretrained(
    "microsoft/layoutlmv2-base-uncased", revision="no_ocr"
)
tokenizer = CustomLayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")



def decode(token):
    if isinstance(token, str):
        return token
    else:
        return token.decode("utf-8")


def preprocess(page, max_length=512, chunk_overlap=True, shrink_dtype=True):

    image = Image.open(io.BytesIO(page[b"image"])).convert("RGB")
    stride = max_length // 3 if (chunk_overlap and max_length) else 0
    truncation = False if max_length is None else True
    add_special_tokens = truncation
    encoded_inputs = processor(
        image,
        [decode(token) for token in page[b"tokens"]],
        boxes=page[b"boxes"],
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=truncation,
        return_overflowing_tokens=True,
        add_special_tokens=add_special_tokens,
        stride=stride,
    )
    encoded_inputs["image"] = encoded_inputs["image"].squeeze(0)

    if shrink_dtype:
        encoded_inputs["image"] = encoded_inputs["image"].type(torch.uint8)
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"].type(torch.int16)
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].type(
            torch.bool
        )
        encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"].type(
            torch.bool
        )
        encoded_inputs["bbox"] = encoded_inputs["bbox"].type(torch.int16)

    if truncation:
        assert encoded_inputs.input_ids.shape[-1:] == torch.Size([512])
        assert encoded_inputs.attention_mask.shape[-1:] == torch.Size([512])
        assert encoded_inputs.token_type_ids.shape[-1:] == torch.Size([512])
        assert encoded_inputs.bbox.shape[-2:] == torch.Size([512, 4])
        assert encoded_inputs.image.shape == torch.Size([3, 224, 224])

    return encoded_inputs


class PageChunkDataset(Dataset):
    def __init__(
        self,
        directory,
        split="train",
        max_length=2048,
        per_chunk_length=512,
        dataset_size=None,
        stride=None,
        reading_order="default",
        include_2d_data=True,
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
        self.include_2d_data = include_2d_data

    def __len__(self):
        return len(self.filepaths)

    def convert_dtype(self, encoded_inputs):
        encoded_inputs["input_ids"] = (
            encoded_inputs["input_ids"].type(torch.int32).squeeze(0)
        )
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
        (encoded_inputs["input_ids"], encoded_inputs["bbox"]), _ = use_reading_order(
            encoded_inputs["input_ids"],
            encoded_inputs["bbox"],
            order=self.reading_order,
        )

        input_mask = encoded_inputs["input_ids"] != 0

        # Empty pages represented as empty pad token
        # TODO: fix upstream or find alternate solution
        input_mask[0] = True

        # Should ideally fix upstream
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"][input_mask]
        if self.include_2d_data:
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
        
        sliced_input = tokenizer.prepare_for_model(
            encoded_inputs["input_ids"][start_index:end_index].tolist(),
            boxes=encoded_inputs["bbox"][start_index:end_index].tolist(),
            max_length=self.max_length,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )

        sliced_input["attention_mask"] = sliced_input["attention_mask"].type(
            torch.float32
        )
        sliced_input["image"] = encoded_inputs["image"]
        return sliced_input

    def save(self):
        pass


if __name__ == "__main__":
    dataset = PageChunkDataset(
        directory="test-page-format",
    )
    for i in range(len(dataset)):
        dataset[i]
