import json
import pickle

import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from configs.cv_and_nlp_config import Config


def load_imdb_data(sample_size=None):
    """Загружает IMDb датасет через HuggingFace datasets."""
    print("Loading IMDb dataset...")
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=config.cache_dir)
    # ^ HuggingFace сам кэширует

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    if sample_size and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=config.seed)
        # Проверяем баланс
        pos_ratio = train_df["label"].mean()
        print(f"Sampled {sample_size} examples, positive ratio: {pos_ratio:.2f}")

    return train_df, test_df


def main():
    config = Config()
    save_dir = config.artifacts_B4 / "tfidf_logreg"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Загрузка данных ──
    train_df, test_df = load_imdb_data(config.B4["sample_size"])
    X_train, y_train = train_df["text"].values, train_df["label"].values
    X_test, y_test = test_df["text"].values, test_df["label"].values

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── TF-IDF векторизация ──
    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=config.B4["tfidf_max_features"],
        ngram_range=config.B4["tfidf_ngram_range"],
        sublinear_tf=config.B4["tfidf_sublinear_tf"],
        stop_words=None,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # ── Logistic Regression ──
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(X_train_tfidf, y_train)

    # ── Оценка ──
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"TF-IDF + Logistic Regression Results")
    print(f"{'=' * 50}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['neg', 'pos'])}")

    # ── Сохранение артефактов ──
    # Модель и векторизатор
    with open(save_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(save_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Метрики
    metrics = {"accuracy": round(acc, 4), "f1": round(f1, 4)}
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["neg", "pos"])
    ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — TF-IDF + LogReg")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", fontsize=16,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=100)
    plt.close()

    # ── Топ слова (интерпретируемость) ──
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]

    # Топ-20 позитивных и негативных слов
    top_indices = np.argsort(coef)
    top_negative = [(feature_names[i], coef[i]) for i in top_indices[:20]]
    top_positive = [(feature_names[i], coef[i]) for i in top_indices[-20:][::-1]]

    print("\nTop 10 negative words (сигнал негатива):")
    for word, weight in top_negative[:10]:
        print(f"  {word:20s} {weight:+.4f}")
    print("\nTop 10 positive words (сигнал позитива):")
    for word, weight in top_positive[:10]:
        print(f"  {word:20s} {weight:+.4f}")

    # График топ слов
    fig, ax = plt.subplots(figsize=(10, 6))
    words = [w for w, _ in top_negative[:15][::-1]] + [w for w, _ in top_positive[:15]]
    weights = [c for _, c in top_negative[:15][::-1]] + [c for _, c in top_positive[:15]]
    colors = ["red"] * 15 + ["green"] * 15
    ax.barh(words, weights, color=colors)
    ax.set_xlabel("Coefficient weight")
    ax.set_title("Top words — TF-IDF + LogReg")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "top_words.png", dpi=100)
    plt.close()

    print(f"\n✓ Artifacts saved to {save_dir}")


if __name__ == "__main__":
    main()
