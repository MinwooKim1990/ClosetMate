from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from numba import jit, prange
import numpy as np

# visualise PCA and scree plot
def visualize_in_3d_with_scree_plot(features, labels):
    pca = PCA(n_components=3) # filter 3 components
    features_3d = pca.fit_transform(features)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], 
                         c=labels, cmap='tab20c', s=15)
    
    ax.set_title('Clustered Feature 3D Visualization')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    
    plt.show()

    pca_full = PCA().fit(features) # PCA perform
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
    n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
    n_components_70 = np.argmax(cumulative_variance >= 0.7) + 1
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid()
    
    plt.axvline(x=n_components_90, color='red', linestyle='--', label=f'90% Variance at PC {n_components_90}')
    plt.axvline(x=n_components_80, color='green', linestyle='--', label=f'80% Variance at PC {n_components_80}')
    plt.axvline(x=n_components_70, color='blue', linestyle='--', label=f'70% Variance at PC {n_components_70}')
    plt.legend()
    plt.show()

# check elbow point
def find_optimal_clusters(features, max_clusters=30, plot=True):
    inertias = []
    K = range(10, max_clusters + 1)
    
    for k in K:
        print(f"Testing k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.show()
    
    kn = KneeLocator(
        K, 
        inertias,
        curve='convex',
        direction='decreasing'
    )
    optimal_k = kn.elbow
    
    print(f"최적의 클러스터 수: {optimal_k}")
    return optimal_k

# get image paths for brand identity
def get_image_paths(folder_path, recursive=False):
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    
    image_paths = []
    
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Cannot find a folder from {folder_path}")
    
    pattern = "**/*" if recursive else "*"
    
    for file_path in folder.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(file_path))
    
    return sorted(image_paths) 

def enhance_features_with_category_weights_optimized(features, dataset, weight=0.4):
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    categories = dataset.dataframe['Name'].str[:2].values
    unique_categories = np.unique(categories)
    
    enhanced = features.copy()
    for cat in unique_categories:
        cat_mask = (categories == cat)
        if np.sum(cat_mask) > 1:
            cat_mean = np.mean(features[cat_mask], axis=0)
            enhanced[cat_mask] += weight * cat_mean
    
    enhanced = enhanced / np.linalg.norm(enhanced, axis=1, keepdims=True)
    
    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
    
    return enhanced

@jit(nopython=True, parallel=True)
def _process_categories_numba(features, cat_ids, cat_counts, weight):
    n_samples, n_features = features.shape
    enhanced = features.copy()
    
    for cat_idx in prange(len(cat_counts)):
        if cat_counts[cat_idx] > 1: 
            cat_mean = np.zeros(n_features)
            count = 0
            for i in range(n_samples):
                if cat_ids[i] == cat_idx:
                    cat_mean += features[i]
                    count += 1
            cat_mean /= count
            
            for i in range(n_samples):
                if cat_ids[i] == cat_idx:
                    enhanced[i] += weight * cat_mean
                
    return enhanced

def enhance_features_with_category_weights_numba(original_features, dataset, weight=0.1):
    features = original_features.astype(np.float64)
    
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    features = features / norms
    
    categories = dataset.dataframe['Name'].str[:2].values
    unique_cats = np.unique(categories)
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    cat_ids = np.array([cat_to_idx[cat] for cat in categories])
    cat_counts = np.array([np.sum(categories == cat) for cat in unique_cats])
    
    enhanced = _process_categories_numba(features, cat_ids, cat_counts, weight)
    
    final_norms = np.linalg.norm(enhanced, axis=1, keepdims=True)
    final_norms = np.where(final_norms == 0, 1e-10, final_norms)
    enhanced = enhanced / final_norms

    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
    
    return enhanced