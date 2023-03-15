import chunk
import boto3
import fire
import os
import io
import itertools
import zlib
import glob
import msgpack
from numpy.random import choice
from rich.progress import track
from rich.console import Console
from multiprocessing.pool import Pool, ThreadPool
import functools
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from docmodel.etl_utils import normalize_bbox
import traceback
from transformers import RobertaTokenizerFast
import tqdm
from typing import Union

from docmodel.dataset import preprocess

import json

TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space = True)

console = Console()
s3_resource = boto3.resource(
    "s3",
    endpoint_url="https://s3.us-east-2.wasabisys.com",
    aws_access_key_id="MILREV7S20K4HJNBPLDL",
    aws_secret_access_key="pbenp4u3CuX1FmzPwC3DzaGQjKMLH0nAtK34OHD6",
)


def download_file_to_dir(
    key: str,
    bucket: str,
    output_dir: str,
    p_train: float = 0.89,
    p_valid: float = 0.01,
    p_test: float = 0.1,
    upload_bucket: str = None,
    shrink_dtype=True,
):
    dir = key[0]
    assert p_train + p_valid + p_test == 1.0
    split = choice(["train", "valid", "test"], size=1, p=[p_train, p_valid, p_test])[0]
    bucket = s3_resource.Bucket(name=bucket)
    obj = bucket.Object(key)
    upload_bucket = s3_resource.Bucket(name=upload_bucket) if upload_bucket else None

    with io.BytesIO() as byte_data:
        try:
            obj.download_fileobj(byte_data)
            byte_data.seek(0)
            data = zlib.decompress(byte_data.read())
            pages = msgpack.unpackb(data, unicode_errors="ignore")
            basename = key.partition(".")[0]
            for page_num, page in enumerate(pages):
                page_meta = page["pages"][0]
                payload = {
                    "tokens": [token["text"] for token in page["tokens"]],
                    "boxes": [
                        normalize_bbox(
                            bbox=[
                                token["position"]["bbLeft"],
                                token["position"]["bbTop"],
                                token["position"]["bbRight"],
                                token["position"]["bbBot"],
                            ],
                            width=page_meta["size"]["width"],
                            height=page_meta["size"]["height"],
                        )
                        for token in page["tokens"]
                    ],
                }

                # Preprocess as full page of tokens, no trunctation
                encoded = preprocess(
                    payload,
                    max_length=None,
                    shrink_dtype=shrink_dtype,
                )
                chunk_data = {
                    "input_ids": encoded.input_ids,
                    "bbox": encoded.bbox,
                }
                if output_dir is not None:
                    # Save data to disk
                    partitioned_output_dir = os.path.join(output_dir, split, dir)
                    if not os.path.exists(partitioned_output_dir):
                        os.mkdir(partitioned_output_dir)

                    torch.save(
                        chunk_data,
                        os.path.join(
                            partitioned_output_dir, f"{basename}-{page_num}.pt"
                        ),
                    )

                # Upload data to S3
                if upload_bucket is not None:
                    obj_up = upload_bucket.Object(
                        os.path.join(split, f"{basename}-{page_num}.pt")
                    )
                    bytes_handle = io.BytesIO()
                    torch.save(chunk_data, bytes_handle)
                    bytes_handle.seek(0)
                    obj_up.upload_fileobj(bytes_handle)

        except Exception as e:
            console.print(f"{key} failed with error: {e}")

    return key


def existing_keys_from_directory(output_dir):
    keys = set()
    for key in glob.glob(os.path.join(output_dir, "**", "*.pt"), recursive=True):
        # Ex: sample/train/019a3b0155deac149392c347a28b3c3b9ecea01e38a19ddeae0c9989-2.pt
        basename = os.path.basename(key).rpartition("-")[0]
        keys.add(basename)
    return keys


def existing_keys_from_bucket(bucket):
    bucket = s3_resource.Bucket(name=bucket)
    keys = set(
        [os.path.basename(obj.key).partition("-")[0] for obj in bucket.objects.all()]
    )
    return keys


def sample(
    bucket_name: str = "general-business-documents-results",
    k: int = 1000000,
    cache_dirs: list = None,
    output_dir: str = None,
    concurrency: int = 4,
    upload_bucket: str = None,
    shrink_dtype=True,
):
    """
    Sample files from an s3 bucket
    """
    if output_dir is not None and not os.path.exists(output_dir):
        os.mkdir(output_dir)

        for split in ["train", "valid", "test"]:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    print("Determining already downloaded files")
    if cache_dirs is not None:
        already_downloaded = set()
        for dir in cache_dirs:
            print("Adding keys from dir", dir)
            already_downloaded |= existing_keys_from_directory(dir)

    k_remaining = k - len(already_downloaded)
    console.print(
        f"{len(already_downloaded)} files downloaded, {k_remaining} files remaining"
    )
    bucket = s3_resource.Bucket(name=bucket_name)
    pool = Pool(concurrency)
    download_page_files = functools.partial(
        download_file_to_dir,
        bucket=bucket_name,
        output_dir=output_dir,
        upload_bucket=upload_bucket,
        shrink_dtype=shrink_dtype,
    )
    sample_page_keys = itertools.islice(
        (
            obj.key
            for obj in bucket.objects.all()
            if obj.key.split(".")[0] not in already_downloaded
        ),
        k_remaining,
    )

    for doc in tqdm.tqdm(
        pool.imap_unordered(
            download_page_files,
            sample_page_keys,
            chunksize=1,
        ),
        total=k_remaining,
    ):
        pass

    pool.close()
    pool.join()


if __name__ == "__main__":
    fire.Fire(sample)