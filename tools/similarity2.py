import numpy as np
import faiss
from tools.saving import save_index, load_index

class Similarity:
    def __init__(self, feature_vectors=None, dataset=None, user_name=None, user_vector=None, model_type=None, force_new=False):
        """
        Initialize Similarity class
        
        Args:
            feature_vectors (np.ndarray, optional): Feature vectors for all items
            dataset (list, optional): Original dataset
            user_name (str, optional): User identifier for saving/loading
            user_vector (np.ndarray, optional): User preference vector
            force_new (bool): Force new index creation even if saved exists
        """
        self.user_name = user_name
        self.dataset = dataset
        self.model_type = model_type
        
        if not force_new and user_name:
            loaded_features = load_index(user_name, self.model_type)
            if loaded_features is not None:
                self.feature_vectors = loaded_features
                self.feature_vectors = self.feature_vectors / np.linalg.norm(self.feature_vectors, axis=1, keepdims=True)
                self.index = faiss.IndexFlatIP(self.feature_vectors.shape[1])
                self.index.add(self.feature_vectors)
                print(f"Loaded saved features for {user_name}")
            else:
                if feature_vectors is None:
                    raise ValueError("No feature vectors provided and no saved features found")
                self._initialize_new(feature_vectors)
        else:
            if feature_vectors is None:
                raise ValueError("Feature vectors must be provided for new initialization")
            self._initialize_new(feature_vectors)
        
        self.original_feature_vectors = feature_vectors.copy()

        # Initialize user vector if provided
        self.user_vector = None
        if user_vector is not None:
            self.user_vector = user_vector / np.linalg.norm(user_vector, axis=1, keepdims=True)
    
    def _initialize_new(self, feature_vectors):
        self.original_feature_vectors = feature_vectors.copy()
        self.feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(self.feature_vectors.shape[1])
        self.index.add(self.feature_vectors)
        
        if self.user_name:
            save_index(self.user_name, self.feature_vectors, self.index, self.model_type)
            print(f"Saved new features for {self.user_name}")

    def get_sim(self, user_vector=None, k=10):
        if user_vector is not None:
            self.user_vector = user_vector / np.linalg.norm(user_vector, axis=1, keepdims=True)
        elif self.user_vector is None:
            raise ValueError("No user vector provided or stored")
        
        _, indices = self.index.search(self.user_vector, k)
        recommended_indices = indices[0]
        original_recommended_vectors = [self.original_feature_vectors[idx] for idx in recommended_indices]
        return list(recommended_indices), original_recommended_vectors
    
    def calculate_similarity_percentage(self, similarity_score):
        percentage = max(0, similarity_score * 100)
        return round(percentage, 2)
    
    def threshold(self, custom_threshold=None):

        if custom_threshold is not None:
            return custom_threshold
            
        model_thresholds = {
            'dino': 88,
            'deit': 67,
            'vit': 72,
            'resnet': 88
        }
    
        return model_thresholds.get(self.model_type, 65)
    
    def get_recommendation_message(self, percentages, custom_threshold=None):
        similarity_threshold = self.threshold(custom_threshold)
        similar_items_count = sum(percentages[0] >= similarity_threshold)
        
        if similar_items_count == 0:
            return "유사한 제품이 없어 구매를 추천 드립니다."
        else:
            return f"{similar_items_count}개의 유사도가 높은 아이템이 있습니다. 구매에 고려해주세요."
    
    def attractiveness(self, query=None, k=None, return_percentage=False, custom_threshold=None):
        if query is not None:
            query_vector = query / np.linalg.norm(query, axis=1, keepdims=True)
        elif self.user_vector is not None:
            query_vector = self.user_vector
        else:
            raise ValueError("No query vector provided or stored")
        
        if k is None:
            k = self.feature_vectors.shape[0]
            
        D_all, _ = self.index.search(query_vector, k)
        
        if return_percentage:
            P_all = np.array([[self.calculate_similarity_percentage(score) 
                            for score in row] for row in D_all])
            recommendation = self.get_recommendation_message(P_all, custom_threshold)
            return P_all, recommendation
        
        return D_all