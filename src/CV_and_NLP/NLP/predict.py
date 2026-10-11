import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configs.cv_and_nlp_config import Config

LABEL_NAMES = {0: "negative", 1: "positive"}

def load_baseline_model(config):
    """Загружает обученные TF-IDF и Logistic Regression."""
    model_dir = config.artifacts_B4 / "tfidf_logreg"
    model_path = model_dir / "model.pkl"
    vectorizer_path = model_dir / "vectorizer.pkl"
    for path in (model_path, vectorizer_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Не найден файл {path}. Сначала запусти baseline.py."
            )
    with model_path.open("rb") as file:
        model = pickle.load(file)
    with vectorizer_path.open("rb") as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


def load_transformer_model(config):
    """Загружает дообученный DistilBERT и его токенизатор."""
    model_dir = config.artifacts_B4 / "distilbert"
    required_files = ("config.json", "tokenizer_config.json")
    for filename in required_files:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Не найден {path}. Сначала запусти transformer.py."
            )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True
                                                               ).to(config.device)
    model.eval()
    return model, tokenizer


def predict_baseline(model, vectorizer, texts):
    """Возвращает предсказания и вероятности baseline."""
    texts = [str(text) for text in texts]
    features = vectorizer.transform(texts)
    probabilities = model.predict_proba(features)
    # Явно учитываем порядок классов обученной модели.
    class_indices = [
        int(np.flatnonzero(model.classes_ == label)[0])
        for label in (0, 1)
    ]
    probabilities = probabilities[:, class_indices]
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    return predictions, probabilities, confidence

@torch.inference_mode()
def predict_transformer(
        model,
        tokenizer,
        texts,
        config,
        batch_size=None,
):
    """Пакетный инференс DistilBERT без вычисления градиентов."""
    if batch_size is None:
        batch_size = config.B4["transformer_batch_size"]

    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным.")

    texts = [str(text) for text in texts]
    all_probabilities = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=config.B4["max_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(config.device) for key, value in encoded.items()}
        logits = model(**encoded)
        probabilities = torch.softmax(logits.logits, dim=-1)
        # Стандартный порядок выходов для модели с двумя классами:
        # 0 — negative, 1 — positive.
        all_probabilities.append(probabilities.cpu().numpy())
    if not all_probabilities:
        return (
            np.array([], dtype=int),
            np.empty((0, 2), dtype=float),
            np.array([], dtype=float),
        )
    probabilities = np.concatenate(all_probabilities, axis=0)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    return predictions, probabilities, confidence


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
