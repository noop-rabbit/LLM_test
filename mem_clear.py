import torch
import gc

# Delete large variables or your model
if 'model' in locals():
    del model

# Force Python to collect remaining garbage fragments
gc.collect()

# Clear the cached GPU memory
torch.cuda.empty_cache()