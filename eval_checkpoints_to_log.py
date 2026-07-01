"""Evaluate trained checkpoints on data and write per-config logs.

For each config it builds the model from the config, loads the checkpoint, runs
the config's metrics on the validation (heldout) loader, and appends one
``epoch: 1 | eval_*: ...`` line to ``Logs/<config>.log`` — exactly the format
``compare_experiments.py`` parses. So after running this, compare_experiments
shows every model, even ones trained elsewhere (only the .pth was copied in).

Examples:
    # default models, austria, first 20 val batches:
    python eval_checkpoints_to_log.py
    # a single model on the full val set:
    python eval_checkpoints_to_log.py --configs ftw_dual_head --max-batches 0
"""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from cfgs import field_segmentation

METRIC_ORDER = ["miou", "boundary_iou"]


def _find_checkpoint(save_dir, name):
    d = Path(save_dir) / name
    best = d / "best_model.pth"
    if best.exists():
        return best
    cands = sorted(d.glob("*.pth"))
    if not cands:
        raise FileNotFoundError(f"No .pth checkpoint under {d}")
    return cands[0]


def _load_model(config, checkpoint, device):
    model = config["model_arch"](**config["model_args"]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_config(name, args, device):
    config = deepcopy(getattr(field_segmentation, name))
    config["data_args"]["data_dir"] = args.data_dir or config["data_args"]["data_dir"]
    config["data_args"]["shuffle"] = False
    config["data_args"]["num_workers"] = 0
    if args.countries:
        config["data_args"]["countries"] = args.countries

    checkpoint = Path(args.checkpoint) if args.checkpoint else _find_checkpoint(
        config["trainer_config"]["save_dir"], name)
    model = _load_model(config, checkpoint, device)

    data_module = config["datamodule"](**config["data_args"])
    loader = data_module.get_heldout_loader()
    metrics = config["metrics"]

    sums = {k: 0.0 for k in metrics}
    n_batches = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(batch["image"])
        for k, fn in metrics.items():
            sums[k] += float(fn.compute(outputs, batch))
        n_batches += 1
        if args.max_batches and n_batches >= args.max_batches:
            break
    means = {k: sums[k] / max(n_batches, 1) for k in metrics}
    return checkpoint, means, n_batches * config["data_args"]["batch_size"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["ftw_mask_baseline", "ftw_dual_head"])
    ap.add_argument("--data-dir", default=None, help="override data dir (else config's)")
    ap.add_argument("--countries", nargs="+", default=["austria"])
    ap.add_argument("--checkpoint", default=None,
                    help="explicit .pth (only valid with a single --configs)")
    ap.add_argument("--max-batches", type=int, default=20, help="0 = whole val set")
    ap.add_argument("--log-dir", default="Logs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for name in args.configs:
        ckpt, m, n = evaluate_config(name, args, device)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,000")
        line = (f"{ts}:INFO:epoch: 1"
                + "".join(f" | eval_{k}: {m[k]:.4f}" for k in METRIC_ORDER))
        with open(log_dir / f"{name}.log", "a") as fh:
            fh.write(line + "\n")
        print(f"{name:<18} ({n} chips, {ckpt.name}): "
              + "  ".join(f"{k}={m[k]:.3f}" for k in METRIC_ORDER))


if __name__ == "__main__":
    main()
