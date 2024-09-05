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
import torch.utils.data
from typing import List
from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler
from pathlib import Path

faulthandler.enable()
warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

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
    LayoutLMForTokenClassification,
    DataCollatorForTokenClassification

)
from docmodel.benchmarks.trainer import CustomTrainer as Trainer
from docmodel.benchmarks.visualizer import visualize_prediction
from docmodel.benchmarks.dataset import DocModelSpatialIEDataset, SpatialIEDataset
from docmodel.doc_model import RobertaDocModelForTokenClassification, DocModelConfig
from docmodel.layout_model import Layoutlmv1ForTokenClassification
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from docmodel.custom_split import custom_train_test_split

docmodel_tokenizer = RobertaTokenizerFast.from_pretrained(
    "roberta-base", add_prefix_space=True
)

xdoc_tokenizer = AutoTokenizer.from_pretrained("microsoft/xdoc-base")

MODEL_CONFIG = {
    "docmodel": {
        "model": RobertaDocModelForTokenClassification,
        "config": DocModelConfig,
        "dataset": DocModelSpatialIEDataset,
        "max_length": 512,
        "gradient_accumulation_steps": 8,
        "tokenizer": docmodel_tokenizer,
        "tokenizer_kwargs": {"pretrained_model_name_or_path": "../model_weights"},
        "collator_kwargs": {"pad_to_multiple_of": 128},
        "pretrained_checkpoint": "../model_weights",
    },
    "xdoc": {
        "model": Layoutlmv1ForTokenClassification,
        "config": AutoConfig,
        "dataset": DocModelSpatialIEDataset,
        "max_length": 512,
        "gradient_accumulation_steps": 8,
        "tokenizer": xdoc_tokenizer,
        "tokenizer_kwargs": {"pretrained_model_name_or_path": "microsoft/xdoc-base"},
        "collator_kwargs": {"pad_to_multiple_of": 128},
        "pretrained_checkpoint": "microsoft/xdoc-base",
    },
}


MODELS_TO_TRAIN = ["docmodel","xdoc"]
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

def objective(trial, train_dataset, val_dataset, collator, resume):

    with mlflow.start_run(nested=True):
        learning_rate = trial.suggest_loguniform('learning_rate', 5e-5, 1e-4)
        print(f"optuna suggested learning rate: {learning_rate}")
        per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [32, 64])
        gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 4)
        num_train_epochs = trial.suggest_int('num_train_epochs', 20, 150)

        args = TrainingArguments(
            output_dir=f"models/trial_{trial.number}",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=32,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy="steps",
            save_steps=500,
            save_total_limit = 3,
            evaluation_strategy="epoch",
            eval_steps=1000,
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
            eval_dataset=val_dataset,
            compute_metrics=metrics,
            data_collator = collator
        )

        if resume:
            trainer.train(resume_from_checkpoint=resume)
        else:
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
    dataset: str = "CORD",
    ocr: str = "default",
    n_trials: int = 50,
    seed: Optional[int] = 42,
    resume: bool = False
):
    global model_init

    model_performances = {}
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sampler=TPESampler(seed=seed)
    else:
        sampler = None

    mlflow.set_experiment("docmodel-xdoc-CORD-sweep")
    for base_model in base_models:
        print(f"\n--- Starting training for {base_model} ---")
        model_config: dict[str, Any] = MODEL_CONFIG[base_model]
        pretrained_checkpoint: Optional[str] = model_config.get("pretrained_checkpoint")
        dataset_cls = model_config["dataset"]
        model_cls = model_config["model"]
        tokenizer = model_config['tokenizer']
        collator_kwargs = model_config.get("collator_kwargs", {})
        collator = DataCollatorForTokenClassification(tokenizer=tokenizer, **collator_kwargs)


        print(f"Using base model: {pretrained_checkpoint}")
        print(f"Training on dataset {dataset}")

        train_data = pd.read_pickle(f"{dataset}/data/{ocr}/train.pkl")
        test_data = pd.read_pickle(f"{dataset}/data/{ocr}/test.pkl")
        validation_path = f"{dataset}/data/{ocr}/validation.pkl"

        if not Path(validation_path).exists():
        # get the validation set from the training data
            train_data, val_data = custom_train_test_split(train_data, test_size=0.2, random_state=42)
        else:
            val_data = pd.read_pickle(validation_path)
        # unpack the data
        _, _, train_labels, _ = train_data
        _, _, val_labels, _ = val_data
        _, _, test_labels, _ = test_data

        all_labels = [item for sublist in train_labels for item in sublist] + [
            item for sublist in test_labels for item in sublist
        ]
        labels = list(set(all_labels))
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for idx, label in enumerate(labels)}

        LabelMap.label_map = id2label

        train_dataset = dataset_cls(
            annotations=train_data,
            label2id=label2id,
            max_length=model_config["max_length"],
            tokenizer = tokenizer
        )

        val_dataset = dataset_cls(
            annotations=val_data,
            label2id=label2id,
            max_length=model_config["max_length"],
            tokenizer=tokenizer
        )

        test_dataset = dataset_cls(
            annotations=test_data,
            label2id=label2id,
            max_length=model_config["max_length"],
            tokenizer = tokenizer
        )

        full_train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

        def model_init():
            model = model_cls.from_pretrained(
                pretrained_checkpoint, num_labels=len(labels)
            )
            model.id2label = id2label
            model.label2id = label2id
            return model

        mlflow.pytorch.autolog(log_models=False)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trail: objective(trail, train_dataset, val_dataset, collator, resume), n_trials=n_trials)

        best_trial = study.best_trial
        model_performances[base_model]={
            "f1_score": best_trial.value,
            "params": best_trial.params,
        }

        print(f"Best trial for {base_model}:")
        print(f"  Value: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # Train the final model with the best hyperparameters
        with mlflow.start_run(run_name=f"{base_model}_training"):
            mlflow.log_param("model_type", base_model)

            best_per_device_batch_size = best_trial.params['per_device_train_batch_size']
            best_gradient_accumulation_steps = best_trial.params['gradient_accumulation_steps']
            best_num_train_epochs = best_trial.params['num_train_epochs']



            best_args = TrainingArguments(
                output_dir="models/best_model",
                per_device_train_batch_size=best_per_device_batch_size,
                per_device_eval_batch_size=32,
                learning_rate=best_trial.params['learning_rate'],
                num_train_epochs=best_num_train_epochs,
                gradient_accumulation_steps=best_gradient_accumulation_steps,
                save_strategy="steps",
                evaluation_strategy="steps",
                eval_steps=len(full_train_dataset) // (best_per_device_batch_size * 10),
                save_steps=len(full_train_dataset) // (best_per_device_batch_size * 5),  # Save roughly every 20% of an epoch
                save_total_limit=5,
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
                train_dataset=full_train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=metrics,
                data_collator = collator
            )

            if resume:
                best_trainer.train(resume_from_checkpoint = resume)
            else:
                best_trainer.train()
            final_eval_result = best_trainer.evaluate()

            mlflow.log_params({
                "best_per_device_train_batch_size": best_per_device_batch_size,
                "best_gradient_accumulation_steps": best_gradient_accumulation_steps,
                "learning_rate": best_trial.params['learning_rate'],
                "num_train_epochs": best_num_train_epochs,
            })
            mlflow.log_metrics(final_eval_result)

            # Save the full model for Hugging Face
            best_model = best_trainer.model
            best_model.save_pretrained(f"best_{base_model}_model_for_hf")
            # Save the tokenizer if you're using a custom one
            if base_model == "docmodel":
                docmodel_tokenizer.save_pretrained(f"best_{base_model}_model_for_hf")
            else:
                AutoTokenizer.from_pretrained(pretrained_checkpoint).save_pretrained(f"best_{base_model}_model_for_hf")

            # Log the model directory as an artifact
            mlflow.log_artifact(f"best_{base_model}_model_for_hf")
            # Optionally, log model configuration
            model_config = {
                "model_type": base_model,
                "num_labels": len(labels),
                "id2label": id2label,
                "label2id": label2id,
            }
            mlflow.log_dict(model_config, "model_config.json")

        print(f"Finished training {base_model} on {dataset}")
    # Compare model performances
    best_model = max(model_performances, key=lambda x: model_performances[x]["f1_score"])

    print("\nModel Performance Comparison:")
    for model, performance in model_performances.items():
        print(f"{model}: F1 Score = {performance['f1_score']:.4f}")

    print(f"\nBest performing model: {best_model}")
    print(f"Best F1 Score: {model_performances[best_model]['f1_score']:.4f}")
    print("Best hyperparameters:")
    for param, value in model_performances[best_model]["params"].items():
        print(f"  {param}: {value}")

    # Log the comparison results with MLflow
    with mlflow.start_run(run_name="model_comparison"):
        for model, performance in model_performances.items():
            mlflow.log_metric(f"{model}_best_f1", performance["f1_score"])
            for param, value in performance["params"].items():
                mlflow.log_param(f"{model}_{param}", value)

        mlflow.log_param("best_model", best_model)
        mlflow.log_metric("best_f1_score", model_performances[best_model]["f1_score"])

if __name__ == "__main__":
    main()
