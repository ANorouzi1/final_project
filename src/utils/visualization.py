import matplotlib.pyplot as plt
import torch


def show_batch(batch, max_items=4):
    images = batch["image"][:max_items].detach().cpu()
    masks = batch["mask"][:max_items].detach().cpu()
    distances = batch["distance"][:max_items].detach().cpu()
    n = images.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).clamp(0, 1))
        axes[i, 0].set_title("image")
        axes[i, 1].imshow(masks[i, 0], cmap="gray")
        axes[i, 1].set_title("mask")
        axes[i, 2].imshow(distances[i, 0], cmap="magma")
        axes[i, 2].set_title("distance target")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    return fig


@torch.no_grad()
def show_predictions(model, batch, device=None, threshold=0.5, max_items=4):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    images = batch["image"][:max_items].to(device)
    outputs = model(images)
    pred_mask = torch.sigmoid(outputs["mask_logits"]).detach().cpu()
    pred_distance = torch.sigmoid(outputs["distance_logits"]).detach().cpu()
    batch = {key: value[:max_items].detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()}
    n = images.shape[0]
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(batch["image"][i].permute(1, 2, 0).clamp(0, 1))
        axes[i, 0].set_title("image")
        axes[i, 1].imshow(batch["mask"][i, 0], cmap="gray")
        axes[i, 1].set_title("target mask")
        axes[i, 2].imshow(pred_mask[i, 0] > threshold, cmap="gray")
        axes[i, 2].set_title("pred mask")
        axes[i, 3].imshow(batch["distance"][i, 0], cmap="magma")
        axes[i, 3].set_title("target distance")
        axes[i, 4].imshow(pred_distance[i, 0], cmap="magma")
        axes[i, 4].set_title("pred distance")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    return fig
