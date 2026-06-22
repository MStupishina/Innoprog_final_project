import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from configs.telco_churn_config import Config


class CustomerSegmenter:
    def __init__(self, config: Config):
        self.config = config
        self.pca = PCA(n_components=config.n_components_pca, random_state=config.random_state)
        # t-SNE лучше ограничить по init='pca' и learning_rate='auto' в свежих версиях sklearn
        self.tsne = TSNE(n_components=2, random_state=config.random_state, init='pca', learning_rate='auto')
        self.kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.random_state, n_init=config.kmeans_n_init)

    def fit_transform_pca(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.pca.fit_transform(X)
        explained_variance = self.pca.explained_variance_ratio_
        print(
            f"PCA ({len(explained_variance)} компоненты): "
            f"{sum(explained_variance) * 100:.1f}% дисперсии"
        )
        return X_pca

    def fit_transform_tsne(self, X: np.ndarray) -> np.ndarray:
        return self.tsne.fit_transform(X)

    def fit_predict_clusters(self, X: np.ndarray) -> np.ndarray:
        return self.kmeans.fit_predict(X)

    def get_cluster_profiles(self, df_original: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        """Считает средние значения признаков для каждого кластера"""
        df_temp = df_original.copy()
        df_temp['Cluster'] = cluster_labels
        total_customers = len(df_temp)
        # Считаем среднее для числовых и моду (самое частое) для категориальных
        profiles = df_temp.groupby('Cluster').agg({
            'MonthlyCharges': 'mean',
            'tenure': 'mean',
            'Churn': lambda x: (x == 'Yes').mean() * 100,  # % оттока в кластере
            'InternetService': lambda x: x.mode()[0],
            'Contract': lambda x: x.mode()[0],
            'customer_value': 'mean'
        }).round(2)

        cluster_size = df_temp.groupby("Cluster").size()
        cluster_share = (cluster_size / total_customers * 100).round(2)

        profiles.insert(0, "Segment Size", cluster_size)
        profiles.insert(1, "Segment Share (%)", cluster_share)

        profiles.rename(columns={'Churn': 'Churn Rate (%)',
                                 'customer_value' : 'Avg Value ($)'}, inplace=True)
        return profiles


