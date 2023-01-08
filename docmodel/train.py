from dataset import PageChunkDataset
from transformers import TrainingArguments
from collator import DataCollatorForWholeWordMask
import fire
import torch
from docmodel.doc_model import DocModelForMLM, DocModelConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast, RobertaConfig
from trainer import CustomTrainer


MODEL_CONFIG = {
    "doc_model": {
        "model": DocModelForMLM,
        "config": DocModelConfig,
        "dataset": PageChunkDataset,
        "batch_size": 16,
        "eval_batch_size": 4,
        "max_length": 512,
        "gradient_accumulation_steps": 1,
        "tokenizer": RobertaTokenizerFast.from_pretrained,
        "tokenizer_kwargs": {
            "pretrained_model_name_or_path": "roberta-base"
        },
        "collator_kwargs": {
            "include_2d_data": True
        },
        'pretrained_checkpoint': 'roberta-base'
    },
    "roberta": {
        "model": RobertaForMaskedLM,
        "config": RobertaConfig,
        "dataset": PageChunkDataset,
        "batch_size": 16,
        "eval_batch_size": 4,
        "max_length": 512,
        "gradient_accumulation_steps": 1,
        "collator_kwargs": {
            "include_2d_data": False
        },
        "tokenizer": RobertaTokenizerFast.from_pretrained,
        "tokenizer_kwargs": {
            "pretrained_model_name_or_path": "roberta-base",
        } 
    }
}


def main(
    experiment_name,
    data_dir=None,
    dataloader_num_workers=0,
    mlm_proba=0.15,
    max_length=None,
    eval_steps=5000,
    eval_examples=500,
    batch_size=None,
    eval_batch_size=None,
    gradient_checkpointing=True,
    pretrained_checkpoint=None,
    base_model="doc_model",
    from_scratch=False,
    num_train_epochs=1.0,
    learning_rate=1e-5,  # 2e-5
    weight_decay=0.01,  # 0.01
    warmup_ratio=0.1,  # 0.1
    gradient_accumulation_steps=1,
    resume=False,
    max_steps=10000,
    **kwargs,
):
    if kwargs:
        raise AssertionError(f"Unexpected arguments: {kwargs}")
    # TODO: start training from random initialization
    # TODO: incorporate other objectives
    model_config = MODEL_CONFIG[base_model]
    pretrained_checkpoint = pretrained_checkpoint or model_config.get('pretrained_checkpoint')
    model_cls = model_config['model']

    if from_scratch:
        print("Training from random init")
        cfg = model_config["config"]
        model = model_cls(
            config=cfg(
                gradient_checkpointing=gradient_checkpointing,
            )
        )
    else:
        print("Training from pre-trained model")
        model = model_cls.from_pretrained(pretrained_checkpoint)

    per_device_batch_size = batch_size or model_config['batch_size']
    gradient_accumulation_steps = gradient_accumulation_steps or model_config['gradient_accumulation_steps']

    tokenizer = model_config['tokenizer'](
        model_max_length=model_config['max_length'], 
        **model_config["tokenizer_kwargs"]
    )
    args = TrainingArguments(
        output_dir=experiment_name,
        run_name=experiment_name,
        dataloader_num_workers=dataloader_num_workers,
        per_device_train_batch_size=per_device_batch_size,
        do_eval=False,
        evaluation_strategy="no",
        num_train_epochs=num_train_epochs,
        prediction_loss_only=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ignore_data_skip=False,
        save_steps=1000,
        save_total_limit=2,
        save_strategy="steps",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_steps=max_steps,
        report_to='wandb'
    )
    collator_kwargs = model_config.get('collator_kwargs', {})
    collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm_probability=mlm_proba,
        **collator_kwargs
    )
    dataset_cls = model_config['dataset']

    train_dataset = dataset_cls(
        directory=data_dir,
        split="train",
        max_length=(max_length or model_config['max_length']),
    )
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    eval_dataset = dataset_cls(
        directory=data_dir,
        split="valid",
        max_length=(max_length or model_config["max_length"]),
        dataset_size=eval_examples,
    )
    trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = CustomTrainer(**trainer_kwargs)

    if not from_scratch and resume and pretrained_checkpoint is not None:
        trainer.train(pretrained_checkpoint)
    else:
        trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    fire.Fire(main)
