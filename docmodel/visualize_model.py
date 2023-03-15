from dataset import DocModelDataset
from collator import DataCollatorForWholeWordMask
import fire
import random
import torch
import copy
from pprint import pprint
from docmodel.doc_model import RobertaDocModelForMLM, DocModelConfig
from transformers import RobertaTokenizerFast
from transformers import AutoConfig


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
    batch_size=1,
    pretrained_checkpoint="one-full-epoch",
    base_model="doc-model-roberta",
    k=5,
):
    model_config = MODEL_CONFIG[base_model]
    pretrained_checkpoint = pretrained_checkpoint or model_config.get(
        "pretrained_checkpoint"
    )
    model_cls = model_config["model"]
    config = AutoConfig.from_pretrained(pretrained_checkpoint)
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    model = model_cls.from_pretrained(pretrained_checkpoint, config=config).to(
        torch.device("cuda")
    )
    model.eval()
    tokenizer = model_config["tokenizer"](
        model_max_length=model_config["max_length"], **model_config["tokenizer_kwargs"]
    )
    dataset_cls = model_config["dataset"]
    dataset = dataset_cls(
        directory=data_dir,
        split="valid",
        max_length=(max_length or model_config["max_length"]),
        reading_order="default",
    )
    mask_token_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0]

    def top_k_at_idx(output, index, k=5):
        token_output = output[0][0, index, :]
        probs = torch.softmax(token_output, -1)
        top_k_indices = torch.argsort(probs, descending=True)[:k]
        top_k_scores = probs[top_k_indices]
        return list(
            zip(
                tokenizer.convert_ids_to_tokens(top_k_indices),
                [score.item() for score in top_k_scores],
            )
        )

    def rank_of_correct_token(output, index, gt_token_id):
        token_output = output[0][0, index, :]
        probs = torch.softmax(token_output, -1)
        sorted_indices = torch.argsort(probs, descending=True)
        return sorted_indices.tolist().index(gt_token_id)

    for i in range(len(dataset)):
        example = dataset[i]
        masked_index = random.choice(list(range(1, example["input_ids"].shape[0] - 1)))
        example_copy = copy.deepcopy(example)
        gt_token = example_copy["input_ids"][masked_index].item()
        example["input_ids"][masked_index] = mask_token_id
        input_text = tokenizer.decode(example["input_ids"])
        for key, value in example.items():
            example[key] = torch.unsqueeze(value, 0).to(torch.device("cuda"))

        output = model(**example)

        rank = rank_of_correct_token(output, masked_index, gt_token)
        if rank != 0:
            print("Inputs", input_text)
            print("Masked", tokenizer.convert_ids_to_tokens([gt_token]))
            pprint(top_k_at_idx(output, masked_index))
            print("Rank of correct token", rank)
            import ipdb

            ipdb.set_trace()
            pass


if __name__ == "__main__":
    fire.Fire(main)
