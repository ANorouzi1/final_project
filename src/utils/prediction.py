import torch


def mask_probability_from_outputs(
    outputs,
    use_sdf=False,
    sdf_weight=0.35,
    sdf_scale=4.0,
):
    """Return the mask probability used for final prediction and metrics."""
    mask_prob = torch.sigmoid(outputs["mask_logits"])
    if not use_sdf or "distance_logits" not in outputs:
        return mask_prob

    sdf = torch.tanh(outputs["distance_logits"])
    sdf_prob = torch.sigmoid(sdf_scale * sdf)
    return ((1.0 - sdf_weight) * mask_prob + sdf_weight * sdf_prob).clamp(0.0, 1.0)
