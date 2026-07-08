import torch


def mask_probability_from_outputs(outputs):
    """Return mask-head probability for prediction and metrics."""
    return torch.sigmoid(outputs["mask_logits"])
