import os
import random
import numpy as np
import torch

def seed_everything(seed: int | None):
    """
    Seeds all relevant random number generators for full reproducibility.
    If seed is None, the function does nothing and allows for non-deterministic behavior.
    """
    if seed is None:
        print("--- Running with no fixed seed (non-deterministic) ---")
        # Ensure cudnn settings are non-deterministic for speed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return
        
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # Crucial settings for reproducibility on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Running with fixed seed: {seed} ---")
