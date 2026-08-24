"""
Compatibility patch for PyTorch 1.8.1.

Add methods and modules that are missing from older PyTorch versions.
"""
import torch
import numpy as np


def frombuffer(buffer, dtype, count=-1, offset=0):
    """Provide a compatible implementation of ``frombuffer`` for PyTorch 1.8.1."""
    if isinstance(buffer, (bytes, bytearray)):
        # Convert the byte buffer to a NumPy array.
        buffer = np.frombuffer(buffer, dtype=np.uint8)

    # Apply the offset and element count.
    if count > 0:
        buffer = buffer[offset:offset+count]
    else:
        buffer = buffer[offset:]

    # Convert the array to a PyTorch tensor.
    return torch.from_numpy(buffer)


# Provide the parametrizations module missing from PyTorch 1.8.1.
class DummyParametrizations:
    """Provide a minimal stand-in for the parametrizations module."""

    @staticmethod
    def weight_norm(*args, **kwargs):
        # Return a no-op weight_norm function.
        def dummy_weight_norm(module, name='weight', dim=0):
            return module
        return dummy_weight_norm


# Apply the compatibility patches.
if not hasattr(torch, 'frombuffer'):
    torch.frombuffer = frombuffer
    print("Added torch.frombuffer for PyTorch 1.8.1")

if not hasattr(torch.nn.utils, 'parametrizations'):
    torch.nn.utils.parametrizations = DummyParametrizations()
    print("Added nn.utils.parametrizations for PyTorch 1.8.1")
