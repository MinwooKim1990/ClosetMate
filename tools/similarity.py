import numpy as np
import faiss

def get_sim(feature_vectors, user_vector, dataset, k=10):
    feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    
    index = faiss.IndexFlatIP(feature_vectors.shape[1])
    index.add(feature_vectors)
    
    user_vector = user_vector / np.linalg.norm(user_vector, axis=1, keepdims=True)
    
    distances, indices = index.search(user_vector, k)
    
    recommended_indices = indices[0]
    recommended_items = [dataset[i] for i in recommended_indices]
    
    return list(recommended_indices)
