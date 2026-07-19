import argparse
import re
from pathlib import Path


METRICS = [
    "eval_miou",
    "eval_boundary_iou",
]


def _parse_log_sessions(path):
    sessions = []
    rows = []
    for line in path.read_text().splitlines():
        if "New Training Session" in line:
            if rows:
                sessions.append(rows)
                rows = []
            continue
        if "epoch:" not in line:
            continue
        # allow digits in the key (e.g. instance_f1) so it isn't dropped
        pairs = dict(re.findall(r"([a-zA-Z_][a-zA-Z_0-9]*): ([+-]?(?:nan|inf|[0-9.]+))", line))
        if "epoch" not in pairs:
            continue
        rows.append({key: float(value) for key, value in pairs.items()})
    if rows:
        sessions.append(rows)
    return sessions


def _best_row(rows, metric):
    eval_rows = [row for row in rows if metric in row]
    if not eval_rows:
        return None
    return max(eval_rows, key=lambda row: row.get(metric, float("-inf")))


def main():
    parser = argparse.ArgumentParser(description="Compare trained experiment logs.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "ftw_mask_baseline",
            "ftw_dual_head",
            "ftw_dual_head_boundary_bce_w20_s012",
            "ftw_three_head_boundary_bce_w20_s012",
        ],
    )
    parser.add_argument("--log-dir", default="Logs")
    parser.add_argument("--select", default="eval_boundary_iou")
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Select the best row across every logged training session instead of only the latest session.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    header = ["config", f"best_epoch_by_{args.select}", *METRICS]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))

    for name in args.configs:
        path = log_dir / f"{name}.log"
        if not path.exists():
            print(f"{name} | missing log | " + " | ".join([""] * len(METRICS)))
            continue
        sessions = _parse_log_sessions(path)
        rows = [row for session in sessions for row in session] if args.all_sessions else (sessions[-1] if sessions else [])
        row = _best_row(rows, args.select)
        if row is None:
            print(f"{name} | no eval rows | " + " | ".join([""] * len(METRICS)))
            continue
        values = [name, str(int(row["epoch"]))]
        values.extend(f"{row.get(metric, float('nan')):.4f}" for metric in METRICS)
        print(" | ".join(values))


if __name__ == "__main__":
    main()
