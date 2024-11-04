import numpy as np
import faiss

class Similarity:
    def __init__(self, feature_vectors, dataset, user_vector=None):
        """
        Initialize Similarity class
        
        Args:
            feature_vectors (np.ndarray): Feature vectors for all items
            dataset (list): Original dataset
            user_vector (np.ndarray, optional): User preference vector
        """
        self.dataset = dataset
        self.feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
        
        # Initialize and store user vector if provided
        self.user_vector = None
        if user_vector is not None:
            self.user_vector = user_vector / np.linalg.norm(user_vector, axis=1, keepdims=True)
        
        # Initialize Faiss index
        self.index = faiss.IndexFlatIP(self.feature_vectors.shape[1])
        self.index.add(self.feature_vectors)

    def get_sim(self, user_vector=None, k=10):
        """
        Get similar items based on user vector
        
        Args:
            user_vector (np.ndarray, optional): User preference vector. If None, uses the one from initialization
            k (int): Number of similar items to return
            
        Returns:
            list: Indices of recommended items
        """
        if user_vector is not None:
            self.user_vector = user_vector / np.linalg.norm(user_vector, axis=1, keepdims=True)
        elif self.user_vector is None:
            raise ValueError("No user vector provided or stored")
        
        # Search similar vectors
        _, indices = self.index.search(self.user_vector, k)
        
        recommended_indices = indices[0]
        
        return list(recommended_indices)
    
    def calculate_similarity_percentage(self, similarity_score):
        """
        Convert cosine similarity score to percentage (0-100)
        
        Args:
            similarity_score (float): Cosine similarity score (-1 to 1)
            
        Returns:
            float: Similarity percentage (0-100)
        """
        percentage = max(0, similarity_score * 100)
        return round(percentage, 2)
    
    def get_recommendation_message(self, percentages, similarity_threshold=70):
        """
        Get recommendation message based on number of similar items
        
        Args:
            percentages (np.ndarray): Array of similarity percentages
            similarity_threshold (float): Threshold for considering items as similar (default: 70%)
            
        Returns:
            str: Recommendation message
        """
        # 유사도가 threshold 이상인 아이템 개수 계산
        similar_items_count = sum(percentages[0] >= similarity_threshold)
        
        if similar_items_count == 0:
            return "유사한 제품이 없어 구매를 추천 드립니다."
        else:
            return f"{similar_items_count}개의 유사한 아이템이 있습니다."
    
    def attractiveness(self, query=None, k=None, return_percentage=False, similarity_threshold=70):
        """
        Find similar vectors and get recommendation
        
        Args:
            query (np.ndarray, optional): Query vector
            k (int, optional): Number of vectors to return. If None, checks all vectors
            return_percentage (bool): If True, returns percentages and recommendation
            similarity_threshold (float): Threshold for considering items as similar
            
        Returns:
            tuple: If return_percentage=True, returns (percentages, recommendation message)
                  else returns (distances)
        """
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
            
            recommendation = self.get_recommendation_message(P_all, similarity_threshold)
            
            return P_all, recommendation
        
        return D_all