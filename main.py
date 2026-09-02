"""Main entry point for Part A — Telco Customer Churn."""

import argparse

from src.telco_churn.scripts.train_pipeline_classification import (
    main as train_classification,
)
from src.telco_churn.scripts.train_pipeline_regression import (
    main as train_regression,
)
from src.telco_churn.scripts.train_pipeline_clustering import (
    main as train_clustering,
)
from src.telco_churn.scripts.predict import (
    main as predict,
)


def train(stage: str) -> None:
    """Run the selected training stage."""

    if stage == "classification":
        print("\n=== Training classification ===")
        train_classification()

    elif stage == "regression":
        print("\n=== Training regression ===")
        train_regression()

    elif stage == "clustering":
        print("\n=== Training clustering ===")
        train_clustering()

    elif stage == "all":
        print("\n=== Training classification ===")
        train_classification()

        print("\n=== Training regression ===")
        train_regression()

        print("\n=== Training clustering ===")
        train_clustering()

        print("\n✓ Part A completed successfully.")
        print("Artifacts saved to artifacts/")


def main():

    parser = argparse.ArgumentParser(
        description="Part A — Telco Customer Churn"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train models",
    )

    train_parser.add_argument(
        "--stage",
        choices=[
            "all",
            "classification",
            "regression",
            "clustering",
        ],
        default="all",
        help="Training stage",
    )

    subparsers.add_parser(
        "predict",
        help="Run inference",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args.stage)

    elif args.command == "predict":
        predict()


if __name__ == "__main__":
    main()