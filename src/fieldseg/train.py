from __future__ import annotations

from pathlib import Path


def add_train_args(parser):
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-masks", required=True)
    parser.add_argument("--val-images")
    parser.add_argument("--val-masks")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--out", default="outputs/checkpoints/dual_head_unet.pt")
    parser.set_defaults(func=run_train)


def main(parser=None):
    if parser is not None:
        return add_train_args(parser)
    import argparse

    parser = argparse.ArgumentParser(description="Train dual-head field segmentation model")
    add_train_args(parser)
    return run_train(parser.parse_args())


def run_train(args) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("Training requires PyTorch. Install requirements.txt first.") from exc

    from .data import FieldFolderDataset
    from .losses import combined_loss
    from .model import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = FieldFolderDataset(args.train_images, args.train_masks, image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_loader = None
    if args.val_images and args.val_masks:
        val_dataset = FieldFolderDataset(args.val_images, args.val_masks, image_size=args.image_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        message = f"epoch={epoch:03d} train_loss={train_loss:.4f}"

        if val_loader is not None:
            val_loss = _run_epoch(model, val_loader, None, device)
            message += f" val_loss={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(out_path, model, args, epoch, val_loss)
        else:
            _save_checkpoint(out_path, model, args, epoch, train_loss)

        print(message)


def _run_epoch(model, loader, optimizer, device: str) -> float:
    import torch

    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
        with torch.set_grad_enabled(is_train):
            outputs = model(batch["image"])
            loss, _ = combined_loss(outputs, batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.detach().cpu()) * batch["image"].shape[0]
        count += batch["image"].shape[0]
    return total / max(count, 1)


def _save_checkpoint(path: Path, model, args, epoch: int, loss: float) -> None:
    import torch

    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "base_channels": args.base_channels,
            "image_size": args.image_size,
        },
        path,
    )
