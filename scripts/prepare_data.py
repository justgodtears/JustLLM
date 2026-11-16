"""Prepare FineWeb-Edu for pretraining.

Streams documents from the Hugging Face dataset, tokenizes them with the GPT-2
BPE tokenizer and writes one continuous token stream to two binary files:
data/val.bin (first VAL_TOKENS tokens) and data/train.bin (the rest).
Tokens are stored as uint16 (the GPT-2 vocab has 50257 entries, so it fits).
"""

import os
import pathlib

import numpy as np
import tiktoken
from datasets import load_dataset

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
TARGET_TOKENS = 200_000_000
VAL_TOKENS = 2_000_000
BATCH_DOCS = 1000
NUM_THREADS = 12

train_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "train.bin"
val_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "val.bin"


def tokenize_batch(enc, texts: list[str]) -> np.ndarray:
    """Tokenize a batch of documents into one flat array of token ids.

    Each document gets an end-of-text token appended, so the model can tell
    where one document ends and the next begins after concatenation.

    Args:
        enc: tiktoken encoding (GPT-2).
        texts: raw document texts.

    Returns:
        1-D uint16 array with all documents' tokens concatenated.
    """
    encoded_texts = enc.encode_ordinary_batch(texts, num_threads=NUM_THREADS)

    for encoded_text in encoded_texts:
        encoded_text.append(enc.eot_token)

    doc_array = [np.array(ids, dtype=np.uint16) for ids in encoded_texts]
    flat_array = np.concatenate(doc_array)
    return flat_array


def write_block(block: np.ndarray, total: int, val_file, train_file) -> int:
    """Write a block of tokens to the val file first, then to the train file.

    The first VAL_TOKENS tokens overall go to val. A block that crosses the
    boundary is split: its beginning completes val, the rest goes to train.

    Args:
        block: 1-D uint16 array of tokens to write.
        total: number of tokens written so far (both files combined).
        val_file: binary file opened for writing.
        train_file: binary file opened for writing.

    Returns:
        Updated total number of tokens written.
    """
    total_val_left = VAL_TOKENS - total
    n_val = min(len(block), max(0, total_val_left))

    block[:n_val].tofile(val_file)
    block[n_val:].tofile(train_file)

    return total + len(block)


def main():
    """Stream the dataset, tokenize it in batches and write val.bin / train.bin.

    Documents are collected into a buffer of BATCH_DOCS texts, tokenized
    together (faster than one by one) and written out. Stops once
    TARGET_TOKENS have been written. The last partial batch is flushed
    after the loop. The dataset is not shuffled, so val.bin always contains
    the same documents between runs.
    """
    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(DATASET, name=CONFIG, streaming=True, split="train")

    with (
        open(train_path, "wb") as train_file,
        open(val_path, "wb") as val_file,
    ):
        total = 0
        buffer = []

        for doc in dataset:
            buffer.append(doc["text"])

            if len(buffer) == BATCH_DOCS:
                block = tokenize_batch(enc, buffer)
                total = write_block(block, total, val_file, train_file)
                buffer = []

            if total >= TARGET_TOKENS:
                break

        if buffer:
            block = tokenize_batch(enc, buffer)
            total = write_block(block, total, val_file, train_file)

    print(f"val: {min(total, VAL_TOKENS)}")
    print(f"train:{total - VAL_TOKENS}")


if __name__ == "__main__":
    main()