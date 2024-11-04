import numpy as np
import os

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
    save_path = f'saved/{user_name}_labels_{model_type}_{n_clusters}clusters.npy'
    np.save(save_path, labels)
    print(f"{user_name} clustered labels were saved to {save_path}.")

def load_labels(user_name, model_type, n_clusters):
    load_path = f'saved/{user_name}_labels_{model_type}_{n_clusters}clusters.npy'
    if os.path.exists(load_path):
        labels = np.load(load_path)
        print(f"{user_name} clustered labels were load from {load_path}.")
        return labels
    return None