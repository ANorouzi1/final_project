from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fieldseg.evaluate import main as evaluate_main
from fieldseg.train import main as train_main


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Polygon-aware field segmentation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the dual-head U-Net")
    train_main(train_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluate_main(eval_parser)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
