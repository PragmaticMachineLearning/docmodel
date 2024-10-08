from typing import Any
import torch
from PIL import Image
from torch.utils.data import Dataset
from docmodel.etl_utils import normalize_bbox, use_reading_order
from transformers import RobertaTokenizerFast



tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space = True)

def preprocess(page: dict[str, Any], max_length: int =512, chunk_overlap: bool = True, shrink_dtype: bool = True, truncation: bool = True):
    add_special_tokens = truncation

    encoded_inputs = tokenizer(
        page["tokens"],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        return_overflowing_tokens=False,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    bbox = page["boxes"]

    bbox_inputs = []
    for word_idx in encoded_inputs.word_ids():
        # Special tokens don't have position info
        if word_idx is None:
            bbox_inputs.append([0, 0, 0, 0])
        # We set the label for the first token of each word.
        else:
            bbox_inputs.append(bbox[word_idx])

    if "labels" in page:
        # For token classification task
        labels = []
        for word_idx in encoded_inputs.word_ids():
            if word_idx is None:
                labels.append(-100)
            else:
                labels.append(page["labels"][word_idx])
        encoded_inputs["labels"] = torch.tensor(labels)

    encoded_inputs["bbox"] = torch.tensor(bbox_inputs)

    if shrink_dtype:
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"].type(torch.int16)
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].type(
            torch.bool
        )
        encoded_inputs["bbox"] = encoded_inputs["bbox"].type(torch.int16)

    if truncation:
        assert encoded_inputs.input_ids.shape[-1:] == torch.Size([max_length])
        assert encoded_inputs.attention_mask.shape[-1:] == torch.Size([max_length])
        assert encoded_inputs.bbox.shape[-2:] == torch.Size([max_length, 4])

    return encoded_inputs


class DocModelSpatialIEDataset(Dataset):
    def __init__(
        self,
        annotations,
        tokenizer=None,
        max_length: int =2048,
        label2id=None,
        reading_order: str ="default",
        doc_info: bool =True,
    ):
        self.words, self.boxes, self.labels, self.images = annotations
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in label2id.items()}
        self.reading_order = reading_order
        self.max_length = max_length
        self.doc_info = doc_info

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get word-level annotations
        words = self.words[idx]
        boxes = self.boxes[idx]
        word_labels = self.labels[idx]
        (words, boxes, word_labels), _ = use_reading_order(
            words, boxes, word_labels, order=self.reading_order
        )

        assert len(words) == len(boxes) == len(word_labels)
        word_label_ids: list[Any] = [self.label2id[label] for label in word_labels]

        payload: dict[str, list[list[int]] ] = {
            "tokens": words,
            "boxes": [
                normalize_bbox(
                    bbox,
                    width=1000,
                    height=1000,
                )
                for bbox in boxes
            ],
            "labels": word_label_ids,
        }
        encoded_inputs = preprocess(
            payload, max_length=self.max_length, shrink_dtype=False
        )
        
        # del encoded_inputs["overflow_to_sample_mapping"]

        # remove batch dimension
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze(0)


        assert encoded_inputs.input_ids.shape == torch.Size([self.max_length])
        assert encoded_inputs.attention_mask.shape == torch.Size([self.max_length])
        assert encoded_inputs.bbox.shape == torch.Size([self.max_length, 4])
        assert encoded_inputs.labels.shape == torch.Size([self.max_length])

        if not self.doc_info:
            del encoded_inputs["bbox"]

        return encoded_inputs


class SpatialIEDataset(Dataset):
    def __init__(
        self,
        annotations,
        processor=None,
        max_length=512,
        label2id=None,
        reading_order="default",
        doc_info=True,
    ):
        self.words, self.boxes, self.labels, self.images = annotations
        self.processor = processor
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in label2id.items()}
        self.reading_order = reading_order
        self.max_length = max_length
        self.doc_info = doc_info

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # first, take an image
        item = self.images[idx]
        image = Image.open(item).convert("RGB")

        # get word-level annotations
        words = self.words[idx]
        boxes = self.boxes[idx]
        word_labels = self.labels[idx]

        # Sequence hasn't been truncated yet to 512 tokens
        # If this dataset is going to end up being >512 tokens (different than >512 words)
        # the act of shuffling to random reading order could affect which tokens
        # are presented to the model and which fall off the end.
        # Assertion: we don't care too much if 1/50 documents gets truncated but we do care if
        # the tokens shown to the document are different based on setting of reading order.

        # In order to ensure a fair comparison -- we may need to shuffle after tokenization.
        # But!  We need to be careful because shuffling words ensure that multi-token
        # words have sequential position IDs.  Shuffling tokens does not produce the same result.

        # If we're going to make the argument that models are still sensitive to 1D reading order
        # and this causes a dependency on the OCR provider used during pre-training,
        # we should shuffle after tokenization.

        # WARNING: If we shuffle after tokenization -- we need to again be careful because we
        # have to care about pad tokens and need to make sure that pad tokens
        # are not included in the random shuffle.

        # We could achieve this by either:
        # A) padding after the random shuffle
        # B) ensuring our random shuffle ignores padding indexes
        (words, boxes, word_labels), _ = use_reading_order(
            words, boxes, word_labels, order=self.reading_order
        )

        assert len(words) == len(boxes) == len(word_labels)
        word_label_ids = [self.label2id[label] for label in word_labels]
        # use processor to prepare everything
        encoded_inputs = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_label_ids,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        encoded_inputs["bbox"] = torch.clamp(encoded_inputs.bbox, min=0, max=1000)

        # remove batch dimension
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze()

        assert encoded_inputs.input_ids.shape == torch.Size([self.max_length])
        assert encoded_inputs.attention_mask.shape == torch.Size([self.max_length])
        # assert encoded_inputs.token_type_ids.shape == torch.Size([self.max_length])
        assert encoded_inputs.bbox.shape == torch.Size([self.max_length, 4])

        assert encoded_inputs.pixel_values.shape == torch.Size([3, 224, 224])
        assert encoded_inputs.labels.shape == torch.Size([self.max_length])

        if not self.doc_info:
            del encoded_inputs["bbox"]
            del encoded_inputs["image"]

        return encoded_inputs
