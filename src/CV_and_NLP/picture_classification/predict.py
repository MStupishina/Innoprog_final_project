import argparse

from configs.cv_and_nlp_config import Config
from predictor import load_model, predict_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model",
                        default="artifacts/cv_classification/resnet18/best_model.pt"
                        )
    args = parser.parse_args()
    config = Config()
    model = load_model(config, args.model)
    classes, probs = predict_image(model, args.image, config)

    print("Predicted classes:")
    for c in classes:
        print("-", c)

if __name__ == "__main__":
    main()
