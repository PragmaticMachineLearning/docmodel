import os
import random
import warnings
import faulthandler
from typing import Any

faulthandler.enable()

warnings.filterwarnings("ignore")
import os
import time
import wandb
import fire
import numpy as np
import pandas as pd
from transformers import LayoutLMv3ForTokenClassification
from transformers import LayoutLMv3Config, LayoutLMv3Tokenizer
from transformers import LayoutLMv3Processor
from transformers import TrainingArguments
from docmodel.benchmarks.trainer import CustomTrainer as Trainer
from docmodel.benchmarks.visualizer import visualize_prediction
from docmodel.benchmarks.dataset import DocModelSpatialIEDataset, SpatialIEDataset
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from ray import tune

from transformers import RobertaTokenizerFast
from docmodel.doc_model import RobertaDocModelForTokenClassification, DocModelConfig
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
docmodel_tokenizer = RobertaTokenizerFast.from_pretrained(
    "roberta-base", add_prefix_space=True
)


MODEL_CONFIG = {
    "docmodel": {
        "model": RobertaDocModelForTokenClassification,
        "config": DocModelConfig,
        "dataset": DocModelSpatialIEDataset,
        "max_length": 2048,
        "gradient_accumulation_steps": 8,
        "tokenizer": docmodel_tokenizer,
        "tokenizer_kwargs": {"pretrained_model_name_or_path": "roberta-base"},
        "collator_kwargs": {"include_2d_data": True, "pad_to_multiple_of": 128},
        "pretrained_checkpoint": "../docmodel_weights",
    },
    "layoutlmv3": {
        "model": LayoutLMv3ForTokenClassification,
        "config": LayoutLMv3Config,
        "dataset": SpatialIEDataset,
        "processor": LayoutLMv3Processor,
        "max_length": 2048,
        "gradient_accumulation_steps": 8,
        "tokenizer": tokenizer,
        "tokenizer_kwargs": {
            "pretrained_model_name_or_path": "microsoft/layoutlmv3-base"
        },
        "collator_kwargs": {"include_2d_data": True, "pad_to_multiple_of": 128},
        "pretrained_checkpoint": "microsoft/layoutlmv3-base",
    },
}


class LabelMap:
    label_map = None


def metrics(eval_preds):
    preds = eval_preds.predictions
    out_label_ids = eval_preds.label_ids
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(LabelMap.label_map[out_label_ids[i][j]])
                preds_list[i].append(LabelMap.label_map[preds[i][j]])

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(classification_report(out_label_list, preds_list))
    return results


def text_from_words(words, bbox):
    # TODO: improve the algo for finding newline breaks
    # to clean this up a bit
    output_text = ""
    prev_midpoint = 0
    for word, box in zip(words, bbox):
        left, top, right, bottom = box
        if top > prev_midpoint:
            output_text += "\n"
        else:
            output_text += " "
        output_text += word
        prev_midpoint = (top + bottom) / 2
    return output_text


def main(
    base_model:str="layoutlmv3",
    checkpoint: str|Any = None,
    dataset: str="SROIE",
    model_name: str|Any = None,
    reading_order: str = "default",
    sweep_parameters: bool = False,
    n_trials: int = 10,
    n_epochs: int = 5,
    batch_size: int = 2,
    max_length: int = 512,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int =1,
    doc_info: bool =True,
    from_scratch: bool =False,
    ocr: str ="default",
    seed: int|Any =None,
    only_label_first_subword: bool =False,
):
    if seed:
        # Doesn't actually do what it should, but just want to get something running
        random.seed(seed)

    if model_name is None:
        model_name = str(int(time.time()))

    model_config: dict[str, Any] = MODEL_CONFIG[base_model]
    pretrained_checkpoint: str | Any | None = model_config.get(
        "pretrained_checkpoint"
    )
    dataset_cls = model_config["dataset"]
    model_cls = model_config["model"]
    cfg = model_config["config"]

    print(f"Using base model: {pretrained_checkpoint}")
    print(f"Training model `{model_name}` on dataset {dataset}")

    _, _, train_labels, _ = train = pd.read_pickle(f"{dataset}/data/{ocr}/train.pkl")
    validation_path = f"{dataset}/data/{ocr}/validation.pkl"
    if not os.path.exists(validation_path):
        validation_path = f"{dataset}/data/{ocr}/test.pkl"
    _, _, test_labels, _ = test = pd.read_pickle(validation_path)
    all_labels = [item for sublist in train_labels for item in sublist] + [
        item for sublist in test_labels for item in sublist
    ]
    labels = list(set(all_labels))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}

    # Global state is a bit gross but I'm not sure how to inject
    # this into the compute_metrics function otherwise
    LabelMap.label_map = id2label
    

    if base_model != 'docmodel':
        processor_cls = model_config.get("processor")
        processor = processor_cls.from_pretrained(
            pretrained_checkpoint,
            apply_ocr=False,
            only_label_first_subword=only_label_first_subword,
        )

        train_dataset = dataset_cls(
            annotations=train,
            processor=processor,
            label2id=label2id,
            reading_order=reading_order,
            max_length=max_length,
            doc_info=doc_info,
        )
        test_dataset = dataset_cls(
            annotations=test,
            processor=processor,
            label2id=label2id,
            reading_order=reading_order,
            max_length=max_length,
            doc_info=doc_info,
        )
    else:
        train_dataset = dataset_cls(
            annotations=train,
            label2id=label2id,
            reading_order=reading_order,
            max_length=max_length,
            doc_info=doc_info,
        )
        test_dataset = dataset_cls(
            annotations=test,
            label2id=label2id,
            reading_order=reading_order,
            max_length=max_length,
            doc_info=doc_info,
        )

    def model_init():

        if from_scratch:
            print("Training from random init")
            model = model_cls(config=cfg, num_labels=len(labels))
        else:
            print(f"Training from pretrained model -- {base_model.upper()}")
            model = model_cls.from_pretrained(
                pretrained_checkpoint, num_labels=len(labels)
            )

        model.id2label = id2label
        model.label2id = label2id
        return model

    args = TrainingArguments(
        output_dir=os.path.join("models", model_name),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        do_eval=False,
        evaluation_strategy="no",
        learning_rate=learning_rate,
        num_train_epochs=n_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=1,
        save_strategy="no",
        fp16=True,
    )
    trainer_kwargs = dict(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        # WARNING: using test set as validation set.
        # Official splits are hidden though so this isn't a true test set.
        eval_dataset=test_dataset,
        compute_metrics=metrics,
    )

    trainer = Trainer(**trainer_kwargs)

    if sweep_parameters:

        def hp_search_space(*args, **kwargs):
            return {
                "learning_rate": tune.loguniform(1e-6, 1e-4),
                "num_train_epochs": tune.uniform(5, 20),
                "seed": tune.uniform(1, 40),
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": tune.choice([1, 2, 4]),
            }

        trainer.hyperparameter_search(
            hp_space=hp_search_space,
            direction="maximize",
            backend="ray",
            n_trials=n_trials,
            resources_per_trial={"gpu": 1},
        )
    else:
        trainer.train()

    trainer.evaluate()

    encoded_inputs = trainer.get_eval_dataloader(trainer.eval_dataset)

    predictions = trainer.predict(trainer.eval_dataset)
    pred_folder = os.path.join("models", model_name, "predictions")
    os.makedirs(pred_folder, exist_ok=True)

    for example, image_path, prediction, words in zip(
        encoded_inputs, trainer.eval_dataset.images, predictions.predictions, test[0]
    ):
        img = visualize_prediction(
            example, image_path, LabelMap.label_map, prediction, tokenizer
        )

        img.save(os.path.join(pred_folder, os.path.basename(image_path)))
        wandb.log({"sample-result": wandb.Image(img)})

        # TODO: add newlines where appropriate

        text = text_from_words(words, bbox=example["bbox"].squeeze(0))
        wandb.log(
            {"sample-text": text_from_words(words, bbox=example["bbox"].squeeze(0))}
        )


if __name__ == "__main__":
    fire.Fire(main)
