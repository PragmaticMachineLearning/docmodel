import re

from dataset import PageDataset, NetworkPageDataset, FullPageDataset
from transformers import TrainingArguments, TrainerCallback
from transformers import LayoutLMv2TokenizerFast
from modeling.layoutlmv3 import LayoutLMv3ForDualMasking
from modeling.layoutlmv2 import LayoutLMv2ForDualMasking
from modeling.fusion_in_encoder import FusionInEncoderForDualMasking
from modeling.luna import LunaForDualMasking
from configuration_layoutlmv2 import LayoutLMv2Config, DocRepConfig
from collator import DataCollatorForWholeWordMask
import fire
import torch
from trainer import CustomTrainer

MODEL_CONFIG = {
    "luna": {
        "model": LunaForDualMasking,
        "config": DocRepConfig,
        "dataset": FullPageDataset,
        "batch_size": 8,
        "eval_batch_size": 4,
        "max_length": 2048,
        "summary_ffn": False,
        "dropout_unpack_attn": False,
        "luna_attention_shape": "V",
        "from_scratch": True,
        "gradient_accumulation_steps": 1,
    },
    "layoutlmv2": {
        "model": LayoutLMv2ForDualMasking,
        "config": LayoutLMv2Config,
        "dataset": FullPageDataset,
        "pretrained_checkpoint": "microsoft/layoutlmv2-base-uncased",
        "batch_size": 16,
        "eval_batch_size": 32,
        "max_length": 512,
        "gradient_accumulation_steps": 1,
    },
    "fusion-in-encoder": {
        "model": FusionInEncoderForDualMasking,
        "config": LayoutLMv2Config,
        "dataset": FullPageDataset,
        "pretrained_checkpoint": "microsoft/layoutlmv2-base-uncased",
        "batch_size": 2,
        "eval_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "max_length": 2048,
        "num_fusion_heads": 4,
    },
}


def main(
    experiment_name,
    data_dir=None,
    dataloader_num_workers=0,
    mlm_proba=0.15,
    position_mask_proba=0.0,
    max_length=None,
    eval_steps=5000,
    eval_examples=500,
    batch_size=None,
    eval_batch_size=None,
    gradient_checkpointing=True,
    pretrained_checkpoint=None,
    base_model="luna",
    from_scratch=False,
    reading_order="default",
    visualization_dir=None,
    visualize_inputs=False,
    visualize_preds=False,
    num_train_epochs=1.0,
    has_rotary_embeddings=False,
    has_relative_attention_bias=True,
    has_spatial_attention_bias=True,
    learning_rate=2e-5,  # 2e-5
    weight_decay=0.01,  # 0.01
    warmup_ratio=0.1,  # 0.1
    fast_attention=False,
    gradient_accumulation_steps=16,
    resume=False,
    local_rank=None,
    **kwargs,
):
    if kwargs:
        raise AssertionError(f"Unexpected arguments: {kwargs}")
    # TODO: start training from random initialization
    # TODO: incorporate other objectives
    model_config = MODEL_CONFIG.get(
        base_model,
    )
    pretrained_checkpoint = pretrained_checkpoint or model_config.get(
        "pretrained_checkpoint"
    )
    model_cls = model_config.get("model")
    if from_scratch:
        cfg = model_config.get("config")
        model = model_cls(
            config=cfg(
                gradient_checkpointing=gradient_checkpointing,
                has_rotary_embeddings=has_rotary_embeddings,
                has_relative_attention_bias=has_relative_attention_bias,
                has_spatial_attention_bias=has_spatial_attention_bias,
                fast_attention=fast_attention,
            )
        )
    else:
        model = model_cls.from_pretrained(pretrained_checkpoint)

    per_device_batch_size = batch_size or model_config.get("batch_size")
    gradient_accumulation_steps = gradient_accumulation_steps or model_config.get(
        "gradient_accumulation_steps"
    )
    model.config.visualize_preds = visualize_preds
    model.config.position_loss = position_mask_proba > 0.0

    # Logic for settings that don't apply to every model type
    for key in [
        "summary_ffn",
        "luna_attention_shape",
        "dropout_unpack_attn",
        "num_fusion_heads",
        "from_scratch",
        "has_rotary_embeddings",
        "has_relative_attention_bias",
        "has_spatial_attention_bias",
    ]:
        if key in model_config:
            setattr(model.config, key, model_config[key])

    # TODO: verify that tokenizer has desired properties
    tokenizer = LayoutLMv2TokenizerFast.from_pretrained(
        "microsoft/layoutlmv2-base-uncased",
        model_max_length=model_config.get("max_length"),
    )
    args = TrainingArguments(
        output_dir=experiment_name,
        dataloader_num_workers=dataloader_num_workers,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=(
            eval_batch_size or model_config.get("eval_batch_size")
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
        position_mask_probability=position_mask_proba,
    )
    dataset_cls = model_config.get("dataset")
    train_dataset = dataset_cls(
        directory=data_dir,
        split="train",
        max_length=(max_length or model_config.get("max_length")),
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
        max_length=(max_length or model_config.get("max_length")),
        dataset_size=eval_examples,
        reading_order=reading_order,
    )
    trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = CustomTrainer(**trainer_kwargs)
    trainer.visualization_dir = visualization_dir
    trainer.visualize_inputs = visualize_inputs

    # # Only needed if using distributed training
    # model.layoutlmv2.visual.synchronize_batch_norm()
    if not from_scratch and resume and pretrained_checkpoint is not None:
        trainer.train(pretrained_checkpoint)
    else:
        trainer.train()
    trainer.save_model()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    fire.Fire(main)


if __name__ == "__main__":
    fire.Fire(main)
