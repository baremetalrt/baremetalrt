import torch

print('torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Name:', torch.cuda.get_device_name(0))
else:
    print('No CUDA GPU detected.')
