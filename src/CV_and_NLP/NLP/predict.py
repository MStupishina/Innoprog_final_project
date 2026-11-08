import argparse
from pathlib import Path
import pandas as pd


from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.NLP.inference_utils import load_baseline_model, load_transformer_model, predict_baseline, \
    predict_transformer

LABEL_NAMES = {0: "negative", 1: "positive"}



def make_results(texts, predictions, probabilities, confidence, model_name):
    """Формирует таблицу предсказаний."""
    return pd.DataFrame({
        "text": texts,
        "model": model_name,
        "prediction": [LABEL_NAMES[int(x)] for x in predictions],
        "probability_negative": probabilities[:, 0],
        "probability_positive": probabilities[:, 1],
        "confidence": confidence,
    })


def run_inference(texts, model_name, config):
    """Запускает одну или обе модели."""
    texts = [str(text) for text in texts]
    if not texts:
        raise ValueError("Не передано ни одного отзыва.")

    if any(not text.strip() for text in texts):
        raise ValueError("Список содержит пустые отзывы.")

    results = []
    if model_name in ("baseline", "both"):
        model, vectorizer = load_baseline_model(config)
        predictions, probabilities, confidence = predict_baseline(
            model, vectorizer, texts
        )
        results.append(
            make_results(
                texts,
                predictions,
                probabilities,
                confidence,
                "tfidf_logreg",
            )
        )
    if model_name in ("transformer", "both"):
        model, tokenizer = load_transformer_model(config)
        predictions, probabilities, confidence = predict_transformer(
            model,
            tokenizer,
            texts,
            config,
        )
        results.append(
            make_results(
                texts,
                predictions,
                probabilities,
                confidence,
                "distilbert",
            )
        )
    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="B4 IMDb sentiment inference")
    parser.add_argument(
        "--model",
        choices=["baseline", "transformer", "both"],
        default="both",
        help="Модель для инференса.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--text",
        nargs="+",
        help="Один или несколько отзывов в кавычках.",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="Путь к CSV-файлу с отзывами.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Название колонки с текстом в CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Путь для сохранения результатов в CSV.",
    )
    args = parser.parse_args()
    config = Config()
    if args.text is not None:
        texts = args.text
    else:
        if not args.input.is_file():
            raise FileNotFoundError(
                f"Входной файл не найден: {args.input}"
            )
        input_df = pd.read_csv(args.input)
        if args.text_column not in input_df.columns:
            raise ValueError(
                f"В CSV нет колонки '{args.text_column}'. "
                f"Доступные колонки: {list(input_df.columns)}"
            )
        if input_df[args.text_column].isna().any():
            raise ValueError(
                f"Колонка '{args.text_column}' содержит пустые значения."
            )
        texts = input_df[args.text_column].astype(str).tolist()
    results = run_inference(
        texts=texts,
        model_name=args.model,
        config=config,
    )
    # Читаемый вывод в консоль.
    for _, row in results.iterrows():
        print(f"\nModel: {row['model']}")
        print(f"Text: {row['text'][:300]}")
        print(f"Sentiment: {row['prediction']}")
        print(f"P(negative): {row['probability_negative']:.4f}")
        print(f"P(positive): {row['probability_positive']:.4f}")
        print(f"Confidence: {row['confidence']:.4f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(
            args.output,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
