import os
import random
import warnings
import faulthandler


faulthandler.enable()

# from docrep.modeling_layoutlmv2 import (
#     FusionInEncoderForTokenClassification,
#     DocRepForTokenClassification,
# )
# from docrep.configuration_layoutlmv2 import DocRepConfig

warnings.filterwarnings("ignore")
import os
import time
import wandb
import fire
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification
from transformers import LayoutLMv3Config, LayoutLMv3Tokenizer
from transformers import LayoutLMv3Processor
from transformers import TrainingArguments
# from docrep.processing_layoutlmv2 import LayoutLMv2Processor
# from docrep.modeling_layoutlmv2 import (
#     LayoutLMv2ForTokenClassification,
# )
from docmodel.benchmarks.trainer import CustomTrainer as Trainer
from docmodel.benchmarks.visualizer import visualize_prediction
from docmodel.benchmarks.dataset import DocModelSpatialIEDataset, SpatialIEDataset
from transformers import RobertaForTokenClassification
from seqeval.metrics import (
    classification_report,
    f1_score,
   
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from ray import tune

# from docrep.etl_utils import use_reading_order
from transformers import LayoutLMv2Tokenizer, RobertaTokenizerFast
from docmodel.doc_model import RobertaDocModelForTokenClassification

tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
docmodel_tokenizer = RobertaTokenizerFast.from_pretrained(
    "roberta-base", add_prefix_space=True
)


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
    base_model="layoutlmv3",
    checkpoint="microsoft/layoutlmv3-base",
    dataset="SROIE",
    model_name=None,
    reading_order="default",
    sweep_parameters=False,
    n_trials=10,
    n_epochs=5,
    batch_size=2,
    max_length=512,
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    doc_info=True,
    from_scratch=False,
    ocr="default",
    seed=None,
    only_label_first_subword=False,
):
    if seed:
        # Doesn't actually do what it should, but just want to get something running
        random.seed(seed)

    if model_name is None:
        model_name = str(int(time.time()))
    print(f"Using base model: {checkpoint}")
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

    if base_model != "docmodel":
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,
            only_label_first_subword=only_label_first_subword,
        )

        train_dataset = SpatialIEDataset(
            annotations=train,
            processor=processor,
            label2id=label2id,
            reading_order=reading_order,
            max_length=max_length,
            doc_info=doc_info,
        )
        test_dataset = SpatialIEDataset(
            annotations=test,
            processor=processor,
            label2id=label2id,
            reading_order=reading_order,
            max_length=max_length,
            doc_info=doc_info,
        )
    else:
        train_dataset = DocModelSpatialIEDataset(
            annotations=train,
            tokenizer=docmodel_tokenizer,
            label2id=label2id,
            reading_order=reading_order,
            max_length=max_length,
            doc_info=doc_info,
        )
        test_dataset = DocModelSpatialIEDataset(
            annotations=test,
            tokenizer=docmodel_tokenizer,
            label2id=label2id,
            reading_order=reading_order,
            max_length=max_length,
            doc_info=doc_info,
        )

    # import ipdb; ipdb.set_trace()
    def model_init():
        if base_model == "layoutlmv3":
            if from_scratch:
                model = LayoutLMv3ForTokenClassification(
                    config=LayoutLMv3Config(), num_labels=len(labels)
                )
            else:
                model = LayoutLMv3ForTokenClassification.from_pretrained(
                    checkpoint, num_labels=len(labels)
                )
        elif base_model == "fusion-in-encoder":
            if from_scratch:
                model = FusionInEncoderForTokenClassification(
                    config=LayoutLMv3Config(max_length=max_length),
                    num_labels=len(labels),
                )
            else:
                model = FusionInEncoderForTokenClassification.from_pretrained(
                    checkpoint, num_labels=len(labels)
                )

        elif base_model == "docrep":
            if from_scratch:
                model = DocRepForTokenClassification(
                    config=DocRepConfig(max_length=max_length)
                )
            else:
                model = DocRepForTokenClassification.from_pretrained(
                    checkpoint, num_labels=len(labels)
                )
        elif base_model == "roberta":
            model = RobertaForTokenClassification.from_pretrained(
                checkpoint, num_labels=len(labels)
            )
        elif base_model == "docmodel":
            model = RobertaDocModelForTokenClassification.from_pretrained(
                checkpoint, num_labels=len(labels)
            )
        else:
            raise ValueError(f"Unknown base_model setting: {base_model}")

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
