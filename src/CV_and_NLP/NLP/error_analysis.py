"""
B4: Сравнительный анализ ошибок — TF-IDF+LogReg vs DistilBERT.
Запуск: python analysis.py  (после baseline.py и transformer.py)
Находит: сарказм, пограничные случаи, длинные тексты — где модели расходятся.
"""
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from configs.cv_and_nlp_config import Config


def load_baseline_model(config: Config):
    """Загружает TF-IDF + LogReg."""
    save_dir = config.artifacts_B4 / "tfidf_logreg"
    with open(save_dir / "model.pkl",  "rb") as f:
        model = pickle.load(f)
    with open(save_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@torch.no_grad()
def load_transformer_model(config: Config):
    """Загружает DistilBERT."""
    save_dir = config.artifacts_B4 / "distilbert"
    tokenizer = DistilBertTokenizer.from_pretrained(save_dir)
    model = DistilBertForSequenceClassification.from_pretrained(save_dir).to(config.device)
    model.eval()
    return model, tokenizer


def predict_baseline(model, vectorizer, texts):
    """Батч-предикт для TF-IDF+LogReg."""
    X = vectorizer.transform(texts)
    return model.predict(X)


def predict_transformer(
        model,
        tokenizer,
        texts,
        config: Config,
        batch_size=32,

):
    """Батч-предикт для DistilBERT."""
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            list(batch_texts),
            max_length=config.B4["max_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(config.device)
        attention_mask = enc["attention_mask"].to(config.device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
    return np.array(all_preds)


def find_disagreements(texts, y_true, y_baseline, y_transformer):
    """
    Находит примеры где:
    1) Модели расходятся между собой
    2) Каждая модель ошибается по сравнению с истиной
    """
    results = defaultdict(list)

    for i, (text, true, base, trans) in enumerate(
            zip(texts, y_true, y_baseline, y_transformer)
    ):
        case = {
            "text": text[:300],
            "true": int(true),
            "baseline": int(base),
            "transformer": int(trans),
        }

        if base != trans:
            # Модели разошлись
            if base == true and trans != true:
                results["baseline_right_transformer_wrong"].append(case)
            elif trans == true and base != true:
                results["transformer_right_baseline_wrong"].append(case)
            elif base != true and trans != true:
                results["both_wrong"].append(case)
        elif base != true and trans != true:
            results["both_wrong_same"].append(case)

    return results


def main():
    config=Config()
    print("Loading models...")
    base_model, vectorizer = load_baseline_model(config)
    trans_model, tokenizer = load_transformer_model(config)

    print("Loading test data...")
    dataset = load_dataset(
        "stanfordnlp/imdb",
        cache_dir=str(config.imdb_cache)
    )
    test_data = dataset["test"]
    texts = test_data["text"]
    labels = test_data["label"]

    print(f"Test size: {len(texts)}")

    # Предикты
    print("\nRunning baseline predictions...")
    y_baseline = predict_baseline(base_model, vectorizer, texts)

    print("Running transformer predictions...")
    y_transformer = predict_transformer(trans_model, tokenizer, texts, config, config.B4["transformer_batch_size"])

    # Общие метрики
    base_acc = accuracy_score(labels, y_baseline)
    base_f1 = f1_score(labels, y_baseline)
    trans_acc = accuracy_score(labels, y_transformer)
    trans_f1 = f1_score(labels, y_transformer)

    print(f"\n{'=' * 50}")
    print("FINAL COMPARISON")
    print(f"{'=' * 50}")
    print(f"TF-IDF + LogReg:  Acc={base_acc:.4f}  F1={base_f1:.4f}")
    print(f"DistilBERT:       Acc={trans_acc:.4f}  F1={trans_f1:.4f}")

    # Где модели расходятся
    disagreements = find_disagreements(texts, labels, y_baseline, y_transformer)

    save_dir = config.artifacts_B4 / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print("DISAGREEMENT ANALYSIS")
    print(f"{'=' * 50}")
    for category, cases in disagreements.items():
        print(f"\n{category}: {len(cases)} examples")
        for case in cases[:3]:
            true_label = "POS" if case["true"] == 1 else "NEG"
            base_label = "POS" if case["baseline"] == 1 else "NEG"
            trans_label = "POS" if case["transformer"] == 1 else "NEG"
            print(f"  True: {true_label} | Baseline: {base_label} | Transformer: {trans_label}")
            print(f"  \"{case['text'][:200]}...\"")
            print()

    # Сохраняем примеры для отчёта
    report = {
        "summary": {
            "baseline_accuracy": round(base_acc, 4),
            "baseline_f1": round(base_f1, 4),
            "transformer_accuracy": round(trans_acc, 4),
            "transformer_f1": round(trans_f1, 4),
        },
        "disagreements": {
            cat: [{
                "text": c["text"],
                "true": c["true"],
                "baseline": c["baseline"],
                "transformer": c["transformer"],
            } for c in cases[:20]]  # топ-20 для каждого типа
            for cat, cases in disagreements.items()
        },
    }

    with open(save_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── График: сравнение метрик ──
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["TF-IDF + LogReg", "DistilBERT"]
    accs = [base_acc, trans_acc]
    f1s = [base_f1, trans_f1]

    x = np.arange(len(models))
    width = 0.3
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1", color="coral")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", fontsize=11)

    ax.set_ylabel("Score")
    ax.set_title("TF-IDF+LogReg vs DistilBERT — IMDb Sentiment")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.85, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "comparison.png", dpi=100)
    plt.close()

    print(f"\n✓ Analysis saved to {save_dir}")


if __name__ == "__main__":
    main()
