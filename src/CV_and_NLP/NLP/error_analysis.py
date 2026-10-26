"""
B4: Сравнительный анализ ошибок — TF-IDF+LogReg vs DistilBERT.
Запуск: python analysis.py  (после baseline.py и transformer.py)
Скрипт использует официальный test split IMDb только для итогового сравнения.
"""
import json
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, recall_score, \
    precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configs.cv_and_nlp_config import Config

LABEL_NAMES = {0: "negative", 1: "positive"}

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


def _make_case(text, true, baseline, transformer, baseline_probs, transformer_probs,
               baseline_confidence, transformer_confidence, index):
    text = str(text)
    return {
        "index": int(index),
        "text": text,
        "text_length_chars": len(text),
        "true": int(true),
        "true_label": LABEL_NAMES[int(true)],
        "baseline": int(baseline),
        "baseline_label": LABEL_NAMES[int(baseline)],
        "baseline_correct": bool(baseline == true),
        "baseline_confidence": round(float(baseline_confidence), 6),
        "baseline_probability_positive": round(float(baseline_probs[1]), 6),
        "transformer": int(transformer),
        "transformer_label": LABEL_NAMES[int(transformer)],
        "transformer_correct": bool(transformer == true),
        "transformer_confidence": round(float(transformer_confidence), 6),
        "transformer_probability_positive": round(float(transformer_probs[1]), 6),
    }


def find_disagreements(texts, y_true, y_baseline, y_transformer,
                       baseline_probs, transformer_probs,
                       baseline_confidence, transformer_confidence,
                       long_text_chars=2000
                       ):
    """
    Группирует ошибки и расхождения; дополнительно выделяет длинные отзывы.

    Длина текста — лишь диагностический признак, а не автоматическое определение
    сарказма или пограничного случая. Такие примеры нужно читать вручную.
    """
    results = defaultdict(list)
    texts = list(texts)
    y_true = np.asarray(y_true)
    y_baseline = np.asarray(y_baseline)
    y_transformer = np.asarray(y_transformer)
    baseline_probs = np.asarray(baseline_probs)
    transformer_probs = np.asarray(transformer_probs)
    baseline_confidence = np.asarray(baseline_confidence)
    transformer_confidence = np.asarray(transformer_confidence)

    arrays = [y_true, y_baseline, y_transformer,
              baseline_probs, transformer_probs,
              baseline_confidence, transformer_confidence]
    lengths = {len(texts), *(len(array) for array in arrays)}
    if len(lengths) != 1:
        raise ValueError(f"Размеры входов не совпадают: {sorted(lengths)}")
    if baseline_probs.shape != (len(texts), 2):
        raise ValueError("baseline_probs должен иметь форму (N, 2)")
    if transformer_probs.shape != (len(texts), 2):
        raise ValueError("transformer_probs должен иметь форму (N, 2)")

    for name, probs in [
        ("baseline_probs", baseline_probs),
        ("transformer_probs", transformer_probs),
    ]:
        if not np.isfinite(probs).all():
            raise ValueError(f"{name} содержит NaN или inf")

        if ((probs < 0) | (probs > 1)).any():
            raise ValueError(f"{name} содержит значения вне [0, 1]")

        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Сумма вероятностей в {name} не равна 1")

    for name, confidence in [
        ("baseline_confidence", baseline_confidence),
        ("transformer_confidence", transformer_confidence),
    ]:
        if not np.isfinite(confidence).all():
            raise ValueError(f"{name} содержит NaN или inf")

        if ((confidence < 0) | (confidence > 1)).any():
            raise ValueError(f"{name} содержит значения вне [0, 1]")

    for name, labels in [
        ("y_true", y_true),
        ("y_baseline", y_baseline),
        ("y_transformer", y_transformer),
    ]:
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{name} содержит метки вне диапазона 0/1")

    for i, values in enumerate(zip(
            texts, y_true, y_baseline, y_transformer, baseline_probs,
            transformer_probs, baseline_confidence, transformer_confidence,
    )):
        text, true, base, trans, base_probs, trans_probs, base_conf, trans_conf = values
        case = _make_case(text, true, base, trans, base_probs, trans_probs,
                          base_conf, trans_conf, i)
        if base != true:
            category = (
                "baseline_false_positive" if base == 1
                else "baseline_false_negative"
            )
            results[category].append(case)

        if trans != true:
            category = (
                "transformer_false_positive" if trans == 1
                else "transformer_false_negative"
            )
            results[category].append(case)

        if base != trans:
            # Модели разошлись
            if base == true:
                results["baseline_only_correct"].append(case)
            else:
                results["transformer_only_correct"].append(case)
        elif base != true:
            results["both_wrong_same"].append(case)
        else:
            results["both_correct"].append(case)

        if len(str(text)) >= long_text_chars:
            results["long_text"].append(case)
        # Уверенность ниже 0.60 — удобный эвристический фильтр для ручной проверки.
        # Порог не является универсальным и не означает, что пример объективно неоднозначен.
        if max(base_conf, trans_conf) < 0.60:
            results["low_confidence_both_models"].append(case)

        # Уверенная ошибка — полезный объект для ручного анализа.
        if base != true and base_conf >= 0.90:
            results["baseline_high_confidence_error"].append(case)

        if trans != true and trans_conf >= 0.90:
            results["transformer_high_confidence_error"].append(case)

    return dict(results)


def save_comparison_plot(save_dir, base_acc, base_f1, trans_acc, trans_f1):
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["TF-IDF + LogReg", "DistilBERT"]
    accs = [base_acc, trans_acc]
    f1s = [base_f1, trans_f1]
    x = np.arange(len(models))
    width = 0.34
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy")
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("TF-IDF + LogReg vs DistilBERT — IMDb test set")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "comparison.png", dpi=120)
    plt.close(fig)


def main():
    config = Config()
    save_dir = config.artifacts_B4 / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    base_model, vectorizer = load_baseline_model(config)
    trans_model, tokenizer = load_transformer_model(config)

    print("Loading test data...")
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=str(config.imdb_cache))
    test_data = dataset["test"]
    texts = test_data["text"]
    labels = test_data["label"]
    print(f"Test size: {len(texts)}")

    # Предикты
    print("\nRunning baseline predictions...")
    y_baseline, baseline_probs, baseline_confidence = predict_baseline(base_model, vectorizer, texts)

    print("Running transformer predictions...")
    y_transformer, transformer_probs, transformer_confidence = predict_transformer(
        trans_model, tokenizer, texts, config, config.B4["transformer_batch_size"])

    # Общие метрики
    base_acc = accuracy_score(labels, y_baseline)
    base_f1 = f1_score(labels, y_baseline, zero_division=0)
    trans_acc = accuracy_score(labels, y_transformer)
    trans_f1 = f1_score(labels, y_transformer, zero_division=0)

    print(f"\n{'=' * 50}")
    print("FINAL COMPARISON")
    print(f"{'=' * 50}")
    print(f"TF-IDF + LogReg:  Acc={base_acc:.4f}  F1={base_f1:.4f}")
    print(f"DistilBERT:       Acc={trans_acc:.4f}  F1={trans_f1:.4f}")

    # Где модели расходятся
    disagreements = find_disagreements(
        texts, labels, y_baseline, y_transformer,
        baseline_probs, transformer_probs,
        baseline_confidence, transformer_confidence,
    )

    print(f"\n{'=' * 50}")
    print("DISAGREEMENT ANALYSIS")
    print(f"{'=' * 50}")
    for category, cases in disagreements.items():
        print(f"\n{category}: {len(cases)} examples")
        # Для ошибок показываем примеры с наименьшей уверенностью модели,
        # для прочих категорий — первые несколько примеров.
        if category.startswith("baseline_false"):
            shown = sorted(cases, key=lambda item: item["baseline_confidence"])[:3]
        elif category.startswith("transformer_false"):
            shown = sorted(cases, key=lambda item: item["transformer_confidence"])[:3]
        else:
            shown = cases[:3]
        for case in shown:
            print(
                f"  true={case['true_label']} | baseline={case['baseline_label']} "
                f"(conf={case['baseline_confidence']:.2f}) | "
                f"transformer={case['transformer_label']} "
                f"(conf={case['transformer_confidence']:.2f}) | "
                f"chars={case['text_length_chars']}"
            )
            print(f"  {case['text'][:250].replace(chr(10), ' ')}...\n")

    # Сохраняем примеры для отчёта

    examples = {}
    for category, cases in disagreements.items():
        if category.endswith("high_confidence_error"):
            confidence_key = ("baseline_confidence" if category.startswith("baseline") else "transformer_confidence")
            selected = sorted(cases, key=lambda case: case[confidence_key], reverse=True)[:20]
        elif category.startswith("baseline_false"):
            selected = sorted(cases, key=lambda case: case["baseline_confidence"])[:20]
        elif category.startswith("transformer_false"):
            selected = sorted(cases, key=lambda case: case["transformer_confidence"])[:20]
        else:
            selected = cases[:20]
        examples[category] = selected

    report = {
        "dataset": "stanfordnlp/imdb",
        "evaluation_split": "test",
        "test_size": len(labels),
        "summary": {
            "baseline_accuracy": round(float(base_acc), 4),
            "baseline_f1": round(float(base_f1), 4),
            "baseline_precision": round(
                precision_score(labels, y_baseline, zero_division=0), 4
            ),
            "baseline_recall": round(
                recall_score(labels, y_baseline, zero_division=0), 4
            ),
            "transformer_accuracy": round(float(trans_acc), 4),
            "transformer_f1": round(float(trans_f1), 4),
            "transformer_precision": round(
                precision_score(labels, y_transformer, zero_division=0), 4
            ),
            "transformer_recall": round(
                recall_score(labels, y_transformer, zero_division=0), 4
            ),
        },
        "classification_reports": {
            "baseline": classification_report(
                labels, y_baseline, target_names=["negative", "positive"],
                output_dict=True, zero_division=0,
            ),
            "transformer": classification_report(
                labels, y_transformer, target_names=["negative", "positive"],
                output_dict=True, zero_division=0,
            ),
        },
        "confusion_matrices": {
            "baseline": confusion_matrix(labels, y_baseline, labels=[0, 1]).tolist(),
            "transformer": confusion_matrix(labels, y_transformer, labels=[0, 1]).tolist(),
        },
        "category_counts": {category: len(cases) for category, cases in disagreements.items()},
        # Ограничиваем размер JSON: по 20 примеров на категорию.
        "examples": examples
    }

    with open(save_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    error_categories = {
        "baseline_false_positive",
        "baseline_false_negative",
        "transformer_false_positive",
        "transformer_false_negative",
        "baseline_only_correct",
        "transformer_only_correct",
        "both_wrong_same",
        "baseline_high_confidence_error",
        "transformer_high_confidence_error",
    }
    error_rows = []
    for category in error_categories:
        for case in disagreements.get(category, []):
            error_rows.append({"category": category, **case})
    if error_rows:
        pd.DataFrame(error_rows).drop_duplicates(
            subset=["category", "index"]
        ).to_csv(save_dir / "error_examples.csv", index=False, encoding="utf-8-sig")

    # График: сравнение метрик
    save_comparison_plot(save_dir, base_acc, base_f1, trans_acc, trans_f1)
    print(f"\nAnalysis saved to: {save_dir}")
    print("Важно: сарказм и пограничность нельзя надёжно определить только длиной/уверенностью;")
    print("отберите сохранённые примеры и проанализируйте их вручную для отчёта.")


if __name__ == "__main__":
    main()
