from bdb import set_trace
import glob
import io
import os
import random

import torch
import PIL
from transformers.models.roberta import RobertaTokenizerFast
from PIL import Image
from torch.utils.data import Dataset
import zlib
from docmodel.etl_utils import use_reading_order
import msgpack
from etl_utils import normalize_bbox
# Stackoverflow magic number -- not sure if this is the absolute max or not
PIL.Image.MAX_IMAGE_PIXELS = 933120000



tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

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
                glob.glob(os.path.join(directory, split, "**", "*.zlib"), recursive=True)
            )
        elif isinstance(directory, list):
            self.filepaths = []
            for dir in directory:
                self.filepaths += list(
                    glob.glob(os.path.join(dir, split, "**", "*.zlib"), recursive=True)
                )
        print(self.filepaths)
        random.seed(seed)
        random.shuffle(self.filepaths)

        if dataset_size is not None:
            self.filepaths = self.filepaths[:dataset_size]

        self.max_length = max_length
        self.stride = stride or self.max_length // 3
        self.num_special_tokens = tokenizer.num_special_tokens_to_add()
        self.per_chunk_length = per_chunk_length
        self.seen_filepaths = []

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
    
    def load(self, filepath:str):
        with open(filepath, 'rb') as f:
            decompressed_file = zlib.decompress(f.read())
            data = msgpack.unpackb(decompressed_file, unicode_errors = 'ignore')
        return data  

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        self.seen_filepaths.append(filepath)
        raw_files = self.load(filepath)
        raw_page = random.choice(raw_files)
        payload = {
                    "tokens": [token["text"] for token in raw_page["tokens"]],
                    "boxes": [
                        normalize_bbox(
                            bbox=[
                                token["position"]["bbLeft"],
                                token["position"]["bbTop"],
                                token["position"]["bbRight"],
                                token["position"]["bbBot"],
                            ],
                            width=raw_page['pages'][0]["size"]["width"],
                            height=raw_page['pages'][0]["size"]["height"],
                        )
                        for token in raw_page["tokens"]
                    ],
                }
        
        encoded_inputs = tokenizer(
                payload['tokens'],
                padding="max_length",
                max_length=None,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
                return_overflowing_tokens=True,
                stride=0,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
        input_mask = encoded_inputs['input_ids'] != 0
        
        bbox = payload["boxes"]
        previous_word_idx = None
        bbox_inputs = []
        
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


        # Empty pages represented as empty pad token
        # TODO: fix upstream or find alternate solution
        input_mask = input_mask.squeeze(0)
        input_mask[0] = True
        encoded_inputs['input_ids'] = encoded_inputs['input_ids'].squeeze(0)
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
        return sliced_input

    def save(self):
        pass



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
        import ipdb; ipdb.set_trace()
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
    dataset = DocModelDataset(
        directory="document-data",
    )
    for i in range(len(dataset)):
        dataset[i]
