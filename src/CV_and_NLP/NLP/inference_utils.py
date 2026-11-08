import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configs.cv_and_nlp_config import Config


def load_baseline_model(config: Config):
    """Загружает TF-IDF + LogReg."""
    save_dir = config.artifacts_B4 / "tfidf_logreg"
    model_path = save_dir / "model.pkl"
    vectorizer_path = save_dir / "vectorizer.pkl"
    for path in (model_path, vectorizer_path):
        if not path.exists():
            raise FileNotFoundError(f"Не найден артефакт: {path}. Сначала запусти baseline.py.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@torch.no_grad()
def load_transformer_model(config: Config):
    """Загружает DistilBERT."""
    save_dir = config.artifacts_B4 / "distilbert"
    required_files = ("config.json","tokenizer_config.json")
    for filename in required_files:
        path = save_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Не найден артефакт: {path}. Проверь сохранение модели и токенизатора "
                "в transformer.py и запусти его повторно.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            save_dir
        ).to(config.device)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Не удалось загрузить модель или токенизатор из {save_dir}. "
            "Проверь артефакты и при необходимости запусти transformer.py повторно."
        ) from exc
    model.eval()
    return model, tokenizer


def predict_baseline(model, vectorizer, texts):
    """Батч-предикт для TF-IDF+LogReg."""
    X = vectorizer.transform(texts)
    probabilities = model.predict_proba(X)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    return predictions, probabilities, confidence


@torch.inference_mode()
def predict_transformer(
        model,
        tokenizer,
        texts,
        config: Config,
        batch_size=None,

):
    """Батч-предикт для DistilBERT."""
    if batch_size is None:
        batch_size = config.B4["transformer_batch_size"]
    all_probs = []
    model.eval()
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=config.B4["max_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(config.device) for key, value in encoded.items()}
        logits = model(**encoded).logits
        prob = torch.softmax(logits, dim=-1)
        all_probs.append(prob.cpu().numpy())
    if not all_probs:
        empty_probs = np.empty((0, 2), dtype=float)
        return np.array([], dtype=int), empty_probs, np.array([], dtype=float)
    probabilities = np.concatenate(all_probs, axis=0)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    return predictions, probabilities, confidence
