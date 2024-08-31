import os
import random
import warnings
import faulthandler
from typing import Any, Optional, Dict
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import optuna
from accelerate import Accelerator
import torch
faulthandler.enable()
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Config,
    LayoutLMv3Tokenizer,
    LayoutLMv3Processor,
    TrainingArguments,
    RobertaTokenizerFast,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
)
from docmodel.benchmarks.trainer import CustomTrainer as Trainer
from docmodel.benchmarks.visualizer import visualize_prediction
from docmodel.benchmarks.dataset import DocModelSpatialIEDataset, SpatialIEDataset
from docmodel.doc_model import RobertaDocModelForTokenClassification, DocModelConfig
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
docmodel_tokenizer = RobertaTokenizerFast.from_pretrained(
    "roberta-base", add_prefix_space=True
)

MODEL_CONFIG = {
    "docmodel": {
        "model": RobertaDocModelForTokenClassification,
        "config": DocModelConfig,
        "dataset": DocModelSpatialIEDataset,
        "max_length": 512,
        "gradient_accumulation_steps": 8,
        "tokenizer": docmodel_tokenizer,
        "tokenizer_kwargs": {"pretrained_model_name_or_path": "../model_weights"},
        "collator_kwargs": {"include_2d_data": True, "pad_to_multiple_of": 128},
        "pretrained_checkpoint": "../model_weights",
    },
    "xdoc": {
        "model": AutoModel,
        "config": AutoConfig,
        "dataset": DocModelSpatialIEDataset,
        "max_length": 512,
        "gradient_accumulation_steps": 8,
        "tokenizer": AutoTokenizer,
        "tokenizer_kwargs": {"pretrained_model_name_or_path": "microsoft/xdoc-base-funsd"},
        "collator_kwargs": {"include_2d_data": True, "pad_to_multiple_of": 128},
        "pretrained_checkpoint": "microsoft/xdoc-base-funsd",
    }
}


MODELS_TO_TRAIN = ["docmodel", "xdoc"]
class LabelMap:
    label_map: Optional[Dict[int, str]] = None

def metrics(eval_preds):
    preds = eval_preds.predictions
    out_label_ids = eval_preds.label_ids
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100 and LabelMap.label_map is not None:
                out_label_list[i].append(LabelMap.label_map.get(out_label_ids[i][j], "UNKNOWN"))
                preds_list[i].append(LabelMap.label_map.get(preds[i][j], "UNKNOWN"))

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(classification_report(out_label_list, preds_list))
    return results

def objective(trial):

    with mlflow.start_run(nested=True):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
        per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [2, 4, 8])
        gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 32)
        num_train_epochs = trial.suggest_int('num_train_epochs', 20, 160)

        args = TrainingArguments(
            output_dir=f"models/trial_{trial.number}",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=8,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy="no",
            evaluation_strategy="epoch",
            logging_dir=f'logs/trial_{trial.number}',
            logging_steps=100,
            warmup_steps=100,
            weight_decay=0.01,
            fp16=True,
            dataloader_num_workers=4,
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate()

        mlflow.log_params({
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "num_train_epochs": num_train_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        })
        mlflow.log_metrics(eval_result)

        return eval_result["eval_f1"]

def main(
    base_models: List[str] = MODELS_TO_TRAIN,
    dataset: str = "FUNSD",
    ocr: str = "read",
    n_trials: int = 100,
    seed: Optional[int] = None,
):
    global model_init, train_dataset, test_dataset

    if seed:
        random.seed(seed)
        np.random.seed(seed)

    mlflow.set_experiment("docmodel-xdoc-funsd-finetune")
    for base_model in base_models
        model_config: dict[str, Any] = MODEL_CONFIG[base_model]
        pretrained_checkpoint: Optional[str] = model_config.get("pretrained_checkpoint")
        dataset_cls = model_config["dataset"]
        model_cls = model_config["model"]

        print(f"Using base model: {pretrained_checkpoint}")
        print(f"Training on dataset {dataset}")

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

        LabelMap.label_map = id2label

        train_dataset = dataset_cls(
            annotations=train,
            label2id=label2id,
            max_length=model_config["max_length"],
        )
        test_dataset = dataset_cls(
            annotations=test,
            label2id=label2id,
            max_length=model_config["max_length"],
        )

        def model_init():
            model = model_cls.from_pretrained(
                pretrained_checkpoint, num_labels=len(labels)
            )
            model.id2label = id2label
            model.label2id = label2id
            return model

        mlflow.pytorch.autolog(log_models=False)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # Train the final model with the best hyperparameters
        with mlflow.start_run(run_name="best_model"):
            best_per_device_batch_size = trial.params['per_device_train_batch_size']
            best_gradient_accumulation_steps = trial.params['gradient_accumulation_steps']
            best_num_train_epochs = trial.params['num_train_epochs']

            best_args = TrainingArguments(
                output_dir="models/best_model",
                per_device_train_batch_size=best_per_device_batch_size,
                per_device_eval_batch_size=8,
                learning_rate=trial.params['learning_rate'],
                num_train_epochs=best_num_train_epochs,
                gradient_accumulation_steps=best_gradient_accumulation_steps,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                logging_dir='logs/best_model',
                logging_steps=100,
                warmup_ratio=0.1,
                weight_decay=0.01,
                fp16=True,
                dataloader_num_workers=4,
                gradient_checkpointing=True,
            )

            best_trainer = Trainer(
                model_init=model_init,
                args=best_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=metrics,
            )

            best_trainer.train()
            final_eval_result = best_trainer.evaluate()

            mlflow.log_params({
                "best_per_device_train_batch_size": best_per_device_batch_size,
                "best_gradient_accumulation_steps": best_gradient_accumulation_steps,
                "learning_rate": trial.params['learning_rate'],
                "num_train_epochs": best_num_train_epochs,
            })
            mlflow.log_metrics(final_eval_result)

            # Save the full model for Hugging Face
            best_model = best_trainer.model
            best_model.save_pretrained("best_model_for_hf")

            # Save the tokenizer if you're using a custom one
            if base_model == "docmodel":
                docmodel_tokenizer.save_pretrained("best_model_for_hf")
            else:
                AutoTokenizer.from_pretrained(pretrained_checkpoint).save_pretrained("best_model_for_hf")

            # Log the model directory as an artifact
            mlflow.log_artifact("best_model_for_hf")
            # Optionally, log model configuration
            model_config = {
                "model_type": base_model,
                "num_labels": len(labels),
                "id2label": id2label,
                "label2id": label2id,
            }
            mlflow.log_dict(model_config, "model_config.json")

        print(f"Finished training {base_model} on {dataset}")

if __name__ == "__main__":
    main()
