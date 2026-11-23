import numpy as np
import tiktoken

from justllm.data import DATA_DIR


def main():
    enc = tiktoken.get_encoding("gpt2")
    data = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    print(f"tokens: {len(data):,}")

    for name, start in (("start", 0), ("middle", len(data) // 2), ("end", len(data) - 100)):
        snippet = enc.decode(data[start : start + 100].tolist())
        print(f"--- {name} ---\n{snippet}\n")


if __name__ == "__main__":
    main()