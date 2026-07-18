import argparse
import csv
import math
import re
import sys
from pathlib import Path


NUMBER_RE = (
    r"[+-]?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
PAIR_RE = re.compile(rf"([A-Za-z_][A-Za-z_0-9]*): ({NUMBER_RE})")
WEIGHT_RE = re.compile(r"_w(?P<weight>\d+(?:\.\d+)?)(?:_|$)")
DISTANCE_WEIGHT_RE = re.compile(r"_d(?P<weight>\d+(?:\.\d+)?)(?:_|$)")

DEFAULT_METRICS = [
    "pixel_iou",
    "miou",
    "boundary_iou",
    "loss",
    "bce",
    "raw_bce",
    "dice_loss",
    "distance_loss",
    "raw_distance_loss",
    "weighted_distance_loss",
    "distance_loss_fraction",
    "boundary_head_loss",
    "weighted_boundary_head_loss",
    "boundary_head_loss_fraction",
    "boundary_target_fraction",
]

EXPECTED_CONFIGS = [
    "ftw_three_head_boundary_bce_w20_s012_headw2_other3",
    "ftw_three_head_boundary_bce_w20_s012_headw2_other3_nosdf",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from cfgs import field_segmentation
except Exception:
    field_segmentation = None


def _parse_log_sessions(path):
    sessions = []
    rows = []
    for line in path.read_text().splitlines():
        if "New Training Session" in line:
            if rows:
                sessions.append(rows)
                rows = []
            continue
        if "eval_" not in line:
            continue
        pairs = dict(PAIR_RE.findall(line))
        if "epoch" not in pairs:
            continue
        row = {}
        for key, value in pairs.items():
            row[key] = float(value)
        row["epoch"] = int(row["epoch"])
        rows.append(row)
    if rows:
        sessions.append(rows)
    return sessions


def _discover_log_configs(log_dir):
    return sorted({path.stem for path in log_dir.glob("*.log")} | set(EXPECTED_CONFIGS))


def _config_sort_key(config):
    if config == "ftw_mask_baseline":
        return (-1.0, -1.0, config)
    weight = _boundary_weight(config)
    distance_weight = _distance_weight(config)
    return (
        math.inf if weight is None else weight,
        math.inf if distance_weight is None else distance_weight,
        config,
    )


def _boundary_weight(config):
    configured = _criterion_arg(config, "boundary_weight")
    if configured is not None:
        return configured
    if not config.startswith("ftw_dual_head_boundary_bce") and not config.startswith(
        "ftw_dual_head_boundary_both"
    ):
        return None
    match = WEIGHT_RE.search(config)
    if not match:
        return None
    return float(match.group("weight"))


def _distance_weight(config):
    configured = _criterion_arg(config, "distance_weight")
    if configured is not None:
        return configured
    if not config.startswith("ftw_dual_head_boundary_bce") and not config.startswith(
        "ftw_dual_head_boundary_both"
    ):
        return None
    match = DISTANCE_WEIGHT_RE.search(config)
    if not match:
        return None
    return float(match.group("weight"))


def _distance_boundary_weight(config):
    return _criterion_arg(config, "distance_boundary_weight")


def _boundary_head_weight(config):
    return _criterion_arg(config, "boundary_head_weight")


def _criterion_arg(config, key):
    if field_segmentation is None or not hasattr(field_segmentation, config):
        return None
    value = getattr(field_segmentation, config).get("criterion_args", {}).get(key)
    return float(value) if value is not None else None


def _rows_for_path(path, all_sessions):
    sessions = _parse_log_sessions(path)
    if all_sessions:
        return [row for session in sessions for row in session]
    if not sessions:
        return []
    return sessions[-1]


def _select_row(rows, select, selection):
    eval_select = _eval_key(select)
    eval_rows = [row for row in rows if eval_select in row]
    if not eval_rows:
        return None
    if selection == "last":
        return eval_rows[-1]
    return max(eval_rows, key=lambda row: row.get(eval_select, float("-inf")))


def _eval_key(metric):
    return metric if metric.startswith("eval_") else f"eval_{metric}"


def _display_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows, columns):
    widths = {
        column: max(
            len(column),
            *(len(_display_value(row.get(column))) for row in rows),
        )
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = " | ".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(
            _display_value(row.get(column)).ljust(widths[column])
            for column in columns
        )
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare boundary-BCE weight sweep logs against baseline logs."
        )
    )
    parser.add_argument("--log-dir", default="Logs")
    parser.add_argument(
        "--configs",
        nargs="+",
        help=(
            "Explicit config names to compare. Defaults to every *.log file "
            "in --log-dir, ordered by filename."
        ),
    )
    parser.add_argument(
        "--select",
        default="boundary_iou",
        help="Metric used for best-row selection, with or without eval_.",
    )
    parser.add_argument(
        "--selection",
        choices=["best", "last"],
        default="best",
        help="Use the best validation row by --select, or the last validation row.",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Compare across every session in each log instead of only the latest one.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path for writing the same summary as CSV.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    configs = args.configs or _discover_log_configs(log_dir)
    sort_metric = args.select.removeprefix("eval_")
    epoch_column = (
        f"best_epoch_by_{sort_metric}"
        if args.selection == "best"
        else "last_eval_epoch"
    )
    metric_columns = [
        "config",
        "boundary_weight",
        "distance_weight",
        "distance_boundary_weight",
        "boundary_head_weight",
        epoch_column,
        *DEFAULT_METRICS,
    ]

    summary_rows = []
    for config in configs:
        path = log_dir / f"{config}.log"
        if not path.exists():
            summary_rows.append(
                {
                    "config": config,
                    "boundary_weight": _boundary_weight(config),
                    "distance_weight": _distance_weight(config),
                    "distance_boundary_weight": _distance_boundary_weight(config),
                    "boundary_head_weight": _boundary_head_weight(config),
                    "status": "missing log",
                }
            )
            continue

        rows = _rows_for_path(path, all_sessions=args.all_sessions)
        selected_row = _select_row(rows, select=args.select, selection=args.selection)
        if selected_row is None:
            summary_rows.append({"config": config, "status": "no eval rows"})
            continue

        summary_row = {
            "config": config,
            "boundary_weight": _boundary_weight(config),
            "distance_weight": _distance_weight(config),
            "distance_boundary_weight": _distance_boundary_weight(config),
            "boundary_head_weight": _boundary_head_weight(config),
            epoch_column: selected_row["epoch"],
        }
        for metric in DEFAULT_METRICS:
            summary_row[metric] = selected_row.get(_eval_key(metric))
        summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: row.get("config", ""))

    columns = [
        column
        for column in [*metric_columns, "status"]
        if any(column in row for row in summary_rows)
    ]
    print(_markdown_table(summary_rows, columns))

    if args.csv:
        _write_csv(args.csv, summary_rows, columns)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
