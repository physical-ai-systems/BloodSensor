import yaml
import torch

class Config(dict):
    """Configuration YAML wrapper."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        """Load configuration from a YAML file."""
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls(config)
    

def get_device():
    """
    Determines the optimal available computing device.

    Returns:
        torch.device: Returns a CUDA device if a compatible GPU is detected, 
        otherwise falls back to the CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    print("CUDA not available, falling back to CPU")
    return torch.device('cpu')