from configs.telco_churn_config import Config
from src.telco_churn.clustering import CustomerSegmenter
from src.telco_churn.datasets.dataset_loader import DatasetLoaderRegression
from src.telco_churn.preprocessor import Preprocessor
from src.telco_churn.visualisation import plot_elbow_method, plot_scatter_2d

def main():
    config = Config()

    plot_dir = config.artifacts_dir / "clustering"

    loader = DatasetLoaderRegression(config)
    df = loader.load_data_with_value()

    y_churn = df[config.target_column_classification].replace({"Yes": 1, "No": 0}).astype(int)

    cols_to_drop = [config.target_column_classification, config.target_column_value, "p_churn"]
    if "customerID" in df.columns:
        cols_to_drop.append("customerID")

    X_raw = df.drop(columns=cols_to_drop)

    preprocessor = Preprocessor(config)
    X_processed = preprocessor.fit_transform(X_raw)

    segmenter = CustomerSegmenter(config)

    plot_elbow_method(X_processed, plot_dir, config.random_state, config.kmeans_n_init)

    X_pca = segmenter.fit_transform_pca(X_processed)
    X_tsne = segmenter.fit_transform_tsne(X_processed)

    print("\nОтрисовка проекций (PCA и t-SNE) по churn")
    plot_scatter_2d(X_pca, y_churn, "PCA 2D Projection (hue=Churn)", "pca_churn.png",
                    plot_dir)
    plot_scatter_2d(X_tsne, y_churn, "t-SNE 2D Projection (hue=Churn)", "tsne_churn.png",
                    plot_dir)

    print(f"\nКластеризация на {config.n_clusters} сегментов (K-Means)...")

    cluster_labels = segmenter.fit_predict_clusters(X_processed)

    plot_scatter_2d(X_pca, cluster_labels, f"K-Means Clusters in PCA Space", "kmeans_pca.png",
                    plot_dir)
    plot_scatter_2d(X_tsne, cluster_labels, f"K-Means Clusters in t-SNE Space", "kmeans_tsne.png",
                            plot_dir)

    print("\nПрофили сегментов (что это за клиенты):")
    profiles = segmenter.get_cluster_profiles(df, cluster_labels)
    print(profiles.to_string())

    profiles_path = plot_dir / "cluster_profiles.csv"
    profiles.to_csv(profiles_path)
    print(f"\nТаблица профилей сохранена в {profiles_path}")

if __name__ == "__main__":
    main()