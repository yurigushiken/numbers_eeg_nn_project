import torch, os

print(f"torch: {torch.__version__}")
print(f"cuda libs: {torch.version.cuda}")
print(f"is_available: {torch.cuda.is_available()}")
print(f"device_count: {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"device_name: {torch.cuda.get_device_name(0)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n") 