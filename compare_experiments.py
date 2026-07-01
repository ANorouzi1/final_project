import argparse
import re
from pathlib import Path


METRICS = [
    "eval_miou",
    "eval_boundary_iou",
]


def _parse_log(path):
    rows = []
    for line in path.read_text().splitlines():
        if "epoch:" not in line:
            continue
        # allow digits in the key (e.g. instance_f1) so it isn't dropped
        pairs = dict(re.findall(r"([a-zA-Z_][a-zA-Z_0-9]*): ([+-]?(?:nan|inf|[0-9.]+))", line))
        if "epoch" not in pairs:
            continue
        rows.append({key: float(value) for key, value in pairs.items()})
    return rows


def _best_row(rows, metric):
    if not rows:
        return None
    return max(rows, key=lambda row: row.get(metric, float("-inf")))


def main():
    parser = argparse.ArgumentParser(description="Compare trained experiment logs.")
    parser.add_argument("--configs", nargs="+", default=["ftw_mask_baseline", "ftw_dual_head"])
    parser.add_argument("--log-dir", default="Logs")
    parser.add_argument("--select", default="eval_miou")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    header = ["config", "best_epoch", *METRICS]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))

    for name in args.configs:
        path = log_dir / f"{name}.log"
        if not path.exists():
            print(f"{name} | missing log | " + " | ".join([""] * len(METRICS)))
            continue
        row = _best_row(_parse_log(path), args.select)
        if row is None:
            print(f"{name} | no epochs | " + " | ".join([""] * len(METRICS)))
            continue
        values = [name, str(int(row["epoch"]))]
        values.extend(f"{row.get(metric, float('nan')):.4f}" for metric in METRICS)
        print(" | ".join(values))


if __name__ == "__main__":
    main()
