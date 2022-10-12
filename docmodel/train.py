from dataset import PageChunkDataset
from transformers import TrainingArguments,LayoutLMv2Tokenizer
from collator import DataCollatorForWholeWordMask
import fire
import torch
from doc_model import DocModelForMLM, DocModelConfig
from trainer import CustomTrainer

MODEL_CONFIG = {
    "doc_model": {
        "model": DocModelForMLM,
        "config": DocModelConfig,
        "dataset": PageChunkDataset,
        "batch_size": 8,
        "eval_batch_size": 4,
        "max_length": 512,
        "from_scratch": True,
        "gradient_accumulation_steps": 1,
    },
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
    reading_order="default",
    num_train_epochs=1.0,
    learning_rate=1e-5,  # 2e-5
    weight_decay=0.01,  # 0.01
    warmup_ratio=0.1,  # 0.1
    gradient_accumulation_steps=16,
    resume=False,
    **kwargs,
):
    if kwargs:
        raise AssertionError(f"Unexpected arguments: {kwargs}")
    # TODO: start training from random initialization
    # TODO: incorporate other objectives
    model_config = MODEL_CONFIG[base_model]
    pretrained_checkpoint = pretrained_checkpoint or model_config['pretrained_checkpoint']
    model_cls = model_config['model']
    if from_scratch:
        cfg = model_config["config"]
        model = model_cls(
            config=cfg(
                gradient_checkpointing=gradient_checkpointing,
            )
        )
    else:
        model = model_cls.from_pretrained(pretrained_checkpoint)

    per_device_batch_size = batch_size or model_config['batch_size']
    gradient_accumulation_steps = gradient_accumulation_steps or model_config['gradient_accumulation_steps']

    # TODO: verify that tokenizer has desired properties
    tokenizer = LayoutLMv2Tokenizer.from_pretrained(
        "microsoft/layoutlmv2-base-uncased",
        model_max_length=model_config['max_length'],
    )
    args = TrainingArguments(
        output_dir=experiment_name,
        dataloader_num_workers=dataloader_num_workers,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=(
            eval_batch_size or model_config['eval_batch_size']
        ),
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        num_train_epochs=num_train_epochs,
        prediction_loss_only=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ignore_data_skip=False,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
    )
    collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm_probability=mlm_proba,
    )
    dataset_cls = model_config['dataset']
    train_dataset = dataset_cls(
        directory=data_dir,
        split="train",
        max_length=(max_length or model_config['max_length']),
        reading_order=reading_order,
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
        reading_order=reading_order,
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
