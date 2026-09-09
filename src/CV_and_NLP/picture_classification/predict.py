import argparse

from configs.cv_and_nlp_config import Config
from predictor import load_model, predict_image, load_threshold


def main():
    parser = argparse.ArgumentParser(description="Инференс для классификации изображений")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model",
                        default="artifacts/cv_classification/resnet18/best_model.pt"
                        )
    args = parser.parse_args()
    config = Config()
    model = load_model(config, args.model)
    threshold = load_threshold(config=config, model_path=args.model)
    classes, probs = predict_image(
        model=model,
        image_path=args.image,
        config=config,
        threshold=threshold,
    )

    print(f"Model: ResNet18")
    print(f"Threshold: {threshold:.3f}")

    print("\nPredictions:")

    if not classes:
        print("No classes detected above threshold.")
    else:
        for idx, probability in enumerate(probs):
            if probability > threshold:
                class_name = config.B1["classes"][idx]
                print(f"- {class_name}: {probability:.3f}")


if __name__ == "__main__":
    main()
