"""Main entry point for the Final Project."""

import sys

from src.telco_churn.scripts.train import main as train_part_a
from src.telco_churn.scripts.predict import main as predict_part_a


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python main.py part_a train")
        print("  python main.py part_a predict")
        return

    part = sys.argv[1]
    command = sys.argv[2]

    if part == "part_a":

        if command == "train":
            train_part_a()

        elif command == "predict":
            predict_part_a()

        else:
            print(f"Unknown command: {command}")

    else:
        print(f"Unknown part: {part}")


if __name__ == "__main__":
    main()