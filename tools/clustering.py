from sklearn.cluster import KMeans
from tools.saving import save_labels, load_labels, save_kmeans_model, load_kmeans_model
import numpy as np
import random


def perform_clustering_with_cache(user_name, features, model_type, n_clusters=13, force_cluster=False):
    if not force_cluster:
        labels = load_labels(user_name, model_type, n_clusters)
        kmeans = load_kmeans_model(user_name, model_type, n_clusters)
        
        if labels is not None and kmeans is not None:
            return labels, kmeans
    
    print("K-means Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=2024)
    labels = kmeans.fit_predict(features)
    
    save_labels(user_name, labels, model_type, n_clusters)
    save_kmeans_model(user_name, kmeans, model_type, n_clusters)
    
    return labels, kmeans

def show_cluster_images(dataset, labels, cluster_num, num_samples=5):
    cluster_indices = np.where(labels == cluster_num)[0]
    
    sample_size = min(num_samples, len(cluster_indices))
    sample_indices = random.sample(list(cluster_indices), sample_size)
    
    print(f"Representative images from Cluster {cluster_num}")
    for idx in sample_indices:
        img_name = dataset.dataframe.iloc[idx]['Name']
        print(f"Image name: {img_name}")
    
    dataset.show_images(sample_indices)

def get_cluster_categories(dataset, kmeans, features, labels, threshold=0.3):
    cluster_categories = {}
    
    unique_clusters = np.unique(labels)
    
    for cluster_num in unique_clusters:
        cluster_indices = np.where(labels == cluster_num)[0]
        
        if len(cluster_indices) > 0:
            categories = [dataset.dataframe.iloc[idx]['Name'][:2] for idx in cluster_indices]
            category_counts = {}
            total = len(cluster_indices)
            
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            max_category, max_count = max(category_counts.items(), key=lambda x: x[1])
            if max_count/total >= threshold:
                cluster_categories[cluster_num] = max_category
            else:
                sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_cats) >= 2:
                    cluster_categories[cluster_num] = f"{sorted_cats[0][0]}/{sorted_cats[1][0]}"
                else:
                    cluster_categories[cluster_num] = max_category
    
    return cluster_categories

def compare_cluster_categories(cluster_dict, category_mapping):
   analysis = {
       'present': {}, 
       'missing': [], 
       'duplicates': {} 
   }
   
   for _, category in cluster_dict.items():
       analysis['present'][category] = analysis['present'].get(category, 0) + 1
   
   for expected in category_mapping.values():
       if expected not in analysis['present']:
           analysis['missing'].append(expected)
   
   analysis['duplicates'] = {k: v for k, v in analysis['present'].items() if v > 1}
   
   print("=== 클러스터 분석 ===")
   print(f"\n전체 카테고리 수: {len(category_mapping)}")
   print(f"현재 포함된 카테고리 수: {len(analysis['present'])}")
   print(f"누락된 카테고리 수: {len(analysis['missing'])}")
   
   print("\n현재 카테고리 분포:")
   for cat, count in analysis['present'].items():
       print(f"- {cat}: {count}번")
   
   if analysis['missing']:
       print("\n누락된 카테고리:")
       for cat in analysis['missing']:
           print(f"- {cat}")
   
   if analysis['duplicates']:
       print("\n중복된 카테고리:")
       for cat, count in analysis['duplicates'].items():
           print(f"- {cat}: {count}번 중복")
           
   return analysis