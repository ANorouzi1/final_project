import matplotlib.pyplot as plt
import torch
from scipy.ndimage import label


def _display_image(image):
    """Return a displayable HWC RGB/grayscale tensor from CHW model input."""
    image = image.detach().cpu().float()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")

    if image.shape[0] == 1:
        preview = image[0]
    else:
        # FTW inputs are [window_b RGBNIR, window_a RGBNIR]. For display, show
        # the RGB channels from the first temporal window available.
        preview = image[:3].permute(1, 2, 0)

    low = torch.quantile(preview, 0.02)
    high = torch.quantile(preview, 0.98)
    if float(high - low) > 1e-6:
        preview = (preview - low) / (high - low)
    return preview.clamp(0, 1)


def _error_overlay(target, pred):
    target = target.bool()
    pred = pred.bool()
    overlay = torch.zeros((*target.shape, 3), dtype=torch.float32)
    overlay[target & pred] = torch.tensor([0.15, 0.75, 0.25])
    overlay[~target & pred] = torch.tensor([0.1, 0.35, 1.0])
    overlay[target & ~pred] = torch.tensor([1.0, 0.15, 0.1])
    return overlay


def _remove_small_components(mask, min_area):
    if min_area <= 0:
        return mask
    labels, n_labels = label(mask.detach().cpu().numpy())
    keep = torch.zeros_like(mask, dtype=torch.bool)
    for component_id in range(1, n_labels + 1):
        component = labels == component_id
        if int(component.sum()) >= min_area:
            keep |= torch.from_numpy(component).to(keep.device)
    return keep


def show_batch(batch, max_items=4):
    images = batch["image"][:max_items].detach().cpu()
    masks = batch["mask"][:max_items].detach().cpu()
    distances = batch["distance"][:max_items].detach().cpu()
    n = images.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        sample_id = batch.get("id", [""] * n)[i]
        axes[i, 0].imshow(_display_image(images[i]), cmap="gray" if images[i].shape[0] == 1 else None)
        axes[i, 0].set_title(f"image\n{sample_id}" if sample_id else "image")
        axes[i, 1].imshow(masks[i, 0], cmap="gray")
        axes[i, 1].set_title("mask")
        axes[i, 2].imshow(distances[i, 0], cmap="coolwarm", vmin=-1, vmax=1)
        axes[i, 2].set_title("SDF target")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    return fig


@torch.no_grad()
def show_predictions(model, batch, device=None, threshold=0.5, max_items=4, min_area=0):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    images = batch["image"][:max_items].to(device)
    outputs = model(images)
    pred_mask = torch.sigmoid(outputs["mask_logits"]).detach().cpu()
    if "distance_logits" in outputs:
        pred_distance = torch.tanh(outputs["distance_logits"]).detach().cpu()
    else:
        pred_distance = torch.zeros_like(batch["distance"][:max_items].detach().cpu())
    batch = {key: value[:max_items].detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()}
    n = images.shape[0]
    fig, axes = plt.subplots(n, 7, figsize=(21, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        sample_id = batch.get("id", [""] * n)[i]
        target = batch["mask"][i, 0] > threshold
        pred = _remove_small_components(pred_mask[i, 0] > threshold, min_area=min_area)
        axes[i, 0].imshow(_display_image(batch["image"][i]), cmap="gray" if batch["image"][i].shape[0] == 1 else None)
        axes[i, 0].set_title(f"image\n{sample_id}" if sample_id else "image")
        axes[i, 1].imshow(target, cmap="gray")
        axes[i, 1].set_title("target mask")
        axes[i, 2].imshow(pred_mask[i, 0], cmap="viridis", vmin=0, vmax=1)
        axes[i, 2].set_title(f"pred prob max {pred_mask[i, 0].max():.2f}")
        axes[i, 3].imshow(pred, cmap="gray")
        axes[i, 3].set_title("pred mask" if min_area <= 0 else f"pred mask >= {min_area}px")
        axes[i, 4].imshow(_error_overlay(target, pred))
        axes[i, 4].set_title("green ok / blue fp / red fn")
        axes[i, 5].imshow(batch["distance"][i, 0], cmap="coolwarm", vmin=-1, vmax=1)
        axes[i, 5].set_title("target SDF")
        axes[i, 6].imshow(pred_distance[i, 0], cmap="coolwarm", vmin=-1, vmax=1)
        axes[i, 6].set_title("pred SDF" if "distance_logits" in outputs else "no SDF head")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    return fig
