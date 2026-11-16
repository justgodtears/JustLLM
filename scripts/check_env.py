import datasets
import numpy
import tiktoken
import torch

def main():
    print(f"Torch version: {torch.__version__}")
    print(f"GPU Cuda available: {torch.cuda.is_available()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3} GB")

    device = "cuda"
    tensor_a = torch.rand(4096, 4096, device=device, dtype=torch.bfloat16)
    tensor_b = torch.rand(4096, 4096, device=device, dtype=torch.bfloat16)
    test = torch.matmul(tensor_a, tensor_b)
    torch.cuda.synchronize()
    print(test)

    print(tiktoken)
    print(datasets)
    print(numpy)


if __name__ == "__main__":
    main()