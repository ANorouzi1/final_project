import argparse
import re
from pathlib import Path


NUMBER_RE = r"[+-]?(?:nan|inf|\d+(?:\.\d+)?)"
PAIR_RE = re.compile(rf"([A-Za-z_][A-Za-z_0-9]*): ({NUMBER_RE})")
THRESHOLD_RE = re.compile(r"_(t\d{2})$")


def _parse_log(path, latest_session=True):
    rows = []
    for line in path.read_text().splitlines():
        if latest_session and "New Training Session" in line:
            rows = []
            continue
        if "eval_" not in line:
            continue
        pairs = dict(PAIR_RE.findall(line))
        if "epoch" not in pairs:
            continue
        row = {key: float(value) for key, value in pairs.items()}
        row["epoch"] = int(row["epoch"])
        rows.append(row)
    return rows


def _metric_threshold(metric_key):
    match = THRESHOLD_RE.search(metric_key)
    if not match:
        return "0.50"
    return f"0.{match.group(1)[1:]}"


def _display_metric(metric_key):
    return THRESHOLD_RE.sub("", metric_key.removeprefix("eval_"))


def _best_rows(rows):
    metric_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("eval_")
            and not key.endswith("_t50")
            and (
                key.startswith("eval_miou")
                or key.startswith("eval_boundary_iou")
            )
        }
    )
    for metric_key in metric_keys:
        yield metric_key, max(rows, key=lambda row: row.get(metric_key, float("-inf")))


def main():
    parser = argparse.ArgumentParser(
        description="Compare best validation mIoU and Boundary IoU from log files."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        default=[
            "Logs/ftw_mask_baseline.log",
            "Logs/ftw_dual_head.log",
            "Logs/ftw_dual_head_boundary_bce.log",
        ],
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Use every session in each log instead of only the latest session.",
    )
    args = parser.parse_args()

    print("log | best_by | threshold | epoch | value | miou | boundary_iou")
    print("--- | --- | --- | --- | --- | --- | ---")
    for log_path in args.logs:
        path = Path(log_path)
        if not path.exists():
            print(f"{path.name} | missing log |  |  |  |  | ")
            continue
        rows = _parse_log(path, latest_session=not args.all_sessions)
        if not rows:
            print(f"{path.name} | no eval rows |  |  |  |  | ")
            continue
        for metric_key, row in _best_rows(rows):
            print(
                f"{path.name} | {_display_metric(metric_key)} | "
                f"{_metric_threshold(metric_key)} | {int(row['epoch'])} | "
                f"{row[metric_key]:.4f} | "
                f"{row.get('eval_miou', float('nan')):.4f} | "
                f"{row.get('eval_boundary_iou', float('nan')):.4f}"
            )


if __name__ == "__main__":
    main()
