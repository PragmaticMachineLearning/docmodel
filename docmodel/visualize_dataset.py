from dataset import DocModelDataset
from collator import DataCollatorForWholeWordMask
import fire
import torch
from docmodel.doc_model import RobertaDocModelForMLM, DocModelConfig
from transformers import RobertaTokenizerFast
from transformers import Trainer, AutoConfig


MODEL_CONFIG = {
    "doc-model-roberta": {
        "model": RobertaDocModelForMLM,
        "config": DocModelConfig,
        "dataset": DocModelDataset,
        "max_length": 2048,
        "gradient_accumulation_steps": 8,
        "tokenizer": RobertaTokenizerFast.from_pretrained,
        "tokenizer_kwargs": {"pretrained_model_name_or_path": "roberta-base"},
        "collator_kwargs": {"include_2d_data": True, "pad_to_multiple_of": 128},
        "pretrained_checkpoint": "roberta-base",
    },
}


def main(
    data_dir=None,
    mlm_proba=0.15,
    max_length=None,
    batch_size=8,
    pretrained_checkpoint=None,
    base_model="doc-model-roberta",
):
    model_config = MODEL_CONFIG[base_model]
    pretrained_checkpoint = pretrained_checkpoint or model_config.get(
        "pretrained_checkpoint"
    )
    print("Training from pre-trained model")
    tokenizer = model_config["tokenizer"](
        model_max_length=model_config["max_length"], **model_config["tokenizer_kwargs"]
    )
    collator_kwargs = model_config.get("collator_kwargs", {})
    collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm_probability=mlm_proba, **collator_kwargs
    )
    dataset_cls = model_config["dataset"]
    dataset = dataset_cls(
        directory=data_dir,
        split="train",
        max_length=(max_length or model_config["max_length"]),
        reading_order="default",
    )
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = batch_start + batch_size
        batch = [dataset[i] for i in range(batch_start, batch_end)]
        collator(batch)


if __name__ == "__main__":
    fire.Fire(main)
