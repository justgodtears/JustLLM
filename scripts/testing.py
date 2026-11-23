import math
from justllm.data import get_batch
from justllm.model import GPT, GPTConfig

model = GPT(GPTConfig())
x, y = get_batch("train", batch_size=4, block_size=128, device="cpu")
logits, loss = model(x, y)

print(logits.shape)
print(loss.item())
print(math.log(50304))