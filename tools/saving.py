import numpy as np
import os
import faiss
import pickle

def save_features(user_name, features, model_type):
    save_path = f'saved/{user_name}_features_{model_type}.npy'
    np.save(save_path, features)
    print(f"{model_type} extracted feature saved to {save_path}.")

def load_features(user_name, model_type):
    load_path = f'saved/{user_name}_features_{model_type}.npy'
    if os.path.exists(load_path):
        features = np.load(load_path)
        print(f"{model_type} extracted feature load from {load_path}.")
        return features
    return None

def save_labels(user_name, labels, model_type, n_clusters):
    save_dir = 'saved'
    os.makedirs(save_dir, exist_ok=True) 
    save_path = os.path.join(save_dir, f"{user_name}_labels_{model_type}_{n_clusters}clusters.npy")
   
    np.save(save_path, labels)
    print(f"label saved to {save_path}.")

def load_labels(user_name, model_type, n_clusters):
    load_path = f'saved/{user_name}_labels_{model_type}_{n_clusters}clusters.npy'
    if os.path.exists(load_path):
        labels = np.load(load_path)
        print(f"{user_name} clustered labels were load from {load_path}.")
        return labels
    return None

def save_index(user_name, feature_vectors, index, model_type):
    """Save index and feature vectors"""
    if not user_name:
        return
        
    os.makedirs('saved', exist_ok=True)
    
    # Save feature vectors
    vectors_path = f'saved/{user_name}_{model_type}_similarity_vectors.npy'
    np.save(vectors_path, feature_vectors)
    
    # Save Faiss index
    index_path = f'saved/{user_name}_{model_type}__similarity_index.faiss'
    faiss.write_index(index, index_path)
    print(f"Saved index and vectors for user {user_name}")

def load_index(user_name, model_type):
    vectors_path = f'saved/{user_name}_{model_type}__similarity_vectors.npy'
    index_path = f'saved/{user_name}_{model_type}__similarity_index.faiss'
    
    if os.path.exists(vectors_path) and os.path.exists(index_path):
        try:
            feature_vectors = np.load(vectors_path)
            index = faiss.read_index(index_path)
            return feature_vectors, index
        except Exception as e:
            print(f"Error loading saved index: {e}")
            return None
    return None

def save_kmeans_model(user_name, model, model_type, n_clusters):
    save_dir = f"saved"
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = f"{save_dir}/{user_name}_{model_type}_{n_clusters}clusters_model.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_kmeans_model(user_name, model_type, n_clusters):
    model_path = f"saved/{user_name}_{model_type}_{n_clusters}clusters_model.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model load from {model_path}")
        return model
    else:
        print(f"Cannot find a model from {model_path}")
        return None