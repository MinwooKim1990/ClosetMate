from sklearn.cluster import KMeans
from tools.saving import save_labels, load_labels

def perform_clustering_with_cache(user_name, features, model_type, n_clusters=13, force_cluster=False):
    if not force_cluster:
        labels = load_labels(user_name, model_type, n_clusters)
        if labels is not None:
            return labels
    
    print("K-means 클러스터링 수행 중...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    save_labels(user_name, labels, model_type, n_clusters)
    
    return labels