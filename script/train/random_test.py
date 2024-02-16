import torch

# Initialize a CUDA device
device = torch.device('cuda')

# Run some code that uses GPU memory
tensor = torch.randn(1000, 1000, device=device)
result = tensor @ tensor

# Now, to clear GPU memory, you can use either of the following methods:

# Method 1: Delete the tensor explicitly
del tensor
del result

# Method 2: Clear the cache using torch.cuda.empty_cache()
torch.cuda.empty_cache()
