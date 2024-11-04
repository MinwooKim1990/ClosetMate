from sklearn.cluster import KMeans
from tools.saving import save_labels, load_labels, save_kmeans_model, load_kmeans_model
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances


def perform_clustering_with_cache(user_name, features, model_type, n_clusters=13, force_cluster=False):
    if not force_cluster:
        labels = load_labels(user_name, model_type, n_clusters)
        kmeans = load_kmeans_model(user_name, model_type, n_clusters)
        
        if labels is not None and kmeans is not None:
            return labels, kmeans
    
    print("K-means 클러스터링 수행 중...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=2024)
    labels = kmeans.fit_predict(features)
    
    save_labels(user_name, labels, model_type, n_clusters)
    save_kmeans_model(user_name, kmeans, model_type, n_clusters)
    
    return labels, kmeans

def show_cluster_images(dataset, labels, cluster_num, num_samples=5):
    cluster_indices = np.where(labels == cluster_num)[0]
    
    sample_size = min(num_samples, len(cluster_indices))
    sample_indices = random.sample(list(cluster_indices), sample_size)
    
    print(f"군집 {cluster_num}에 속한 이미지들:")
    for idx in sample_indices:
        img_name = dataset.dataframe.iloc[idx]['Name']
        print(f"Image name: {img_name}")
    
    dataset.show_images(sample_indices)

def get_cluster_categories(dataset, kmeans, features, labels):
    cluster_categories = {}
    
    unique_clusters = np.unique(labels)
    
    for cluster_num in unique_clusters:
        centroid = kmeans.cluster_centers_[cluster_num]
        
        cluster_indices = np.where(labels == cluster_num)[0]
        
        if len(cluster_indices) > 0:
            cluster_features = features[cluster_indices]
            distances = euclidean_distances([centroid], cluster_features)[0]
            
            closest_idx = cluster_indices[np.argmin(distances)]
            
            img_name = dataset.dataframe.iloc[closest_idx]['Name']
            category = img_name[:2]
            
            cluster_categories[cluster_num] = category
    
    return cluster_categories
