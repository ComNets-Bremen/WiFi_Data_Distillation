import torch
free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
print(f"Free GPU memory: {free/1e9:.2f} GB")
torch.cuda.empty_cache()
print(f"Free GPU memory: {free/1e9:.2f} GB")
