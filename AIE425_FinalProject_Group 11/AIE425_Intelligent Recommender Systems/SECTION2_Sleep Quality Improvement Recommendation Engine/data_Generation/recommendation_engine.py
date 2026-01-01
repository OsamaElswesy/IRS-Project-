"""
Sleep Quality Recommendation Engine - Implementation Examples
Content-based recommendation approach for sleep improvements

This module demonstrates how to build a working recommendation system
using the generated synthetic dataset.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Load and preprocess the sleep dataset"""
    
    def __init__(self, data_dir='./'):
        self.data_dir = data_dir
        self.users = None
        self.items = None
        self.ratings = None
        self.reviews = None
    
    def load_all(self):
        """Load all dataset files"""
        self.users = pd.read_csv(f'{self.data_dir}sleep_users.csv', na_filter=False)
        self.items = pd.read_csv(f'{self.data_dir}sleep_items.csv', na_filter=False)
        self.ratings = pd.read_csv(f'{self.data_dir}sleep_ratings.csv', na_filter=False)
        self.reviews = pd.read_csv(f'{self.data_dir}sleep_reviews.csv', na_filter=False)
        
        print(f"[OK] Loaded {len(self.users)} users")
        print(f"[OK] Loaded {len(self.items)} items")
        print(f"[OK] Loaded {len(self.ratings)} ratings")
        print(f"[OK] Loaded {len(self.reviews)} reviews")
        
        return self.users, self.items, self.ratings, self.reviews

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureExtractor:
    """Extract and encode features for content-based recommendation"""
    
    def __init__(self, users_df, items_df):
        self.users = users_df.copy()
        self.items = items_df.copy()
        self.user_feature_matrix = None
        self.item_feature_matrix = None
    
    def extract_user_features(self) -> np.ndarray:
        """
        Convert user profile to feature vector
        
        Features: sleep_issue (one-hot), habits (numeric), age (normalized)
        Returns: (n_users, n_features) matrix
        """
        
        # Sleep issues (one-hot)
        all_issues = set(self.users['primary_sleep_issue'].unique()) | \
                     set(self.users['secondary_sleep_issue'].unique())
        
        issue_mapping = {issue: idx for idx, issue in enumerate(sorted(all_issues))}
        
        # Habit encodings
        habit_encodings = {
            'exercise_frequency': {'Never': 0, 'Occasionally': 1, '2-3x/week': 2, 'Daily': 3},
            'caffeine_intake': {'None': 0, 'Moderate': 1, 'High': 2, 'Very High': 3},
            'screen_time_before_bed': {'<30 min': 3, '30-60 min': 2, '1-2 hours': 1, '>2 hours': 0},
            'stress_level': {'Low': 3, 'Moderate': 2, 'High': 1, 'Very High': 0},
            'sleep_schedule': {'Irregular': 0, 'Somewhat Regular': 1, 'Regular': 2, 'Very Regular': 3},
            'alcohol_consumption': {'Never': 3, 'Occasionally': 2, 'Regular': 1, 'Heavy': 0},
            'napping_habit': {'No naps': 3, 'Occasional': 2, 'Regular short naps': 1, 'Long naps': 0},
        }
        
        # Build feature matrix
        features = []
        for idx, row in self.users.iterrows():
            feature_vec = []
            
            # Age (normalized 0-1)
            feature_vec.append(row['age'] / 75.0)
            
            # Primary sleep issue (one-hot)
            primary_issue_vec = [1.0 if issue == row['primary_sleep_issue'] else 0.0 
                                 for issue in sorted(all_issues)]
            feature_vec.extend(primary_issue_vec)
            
            # Habits (numeric 0-3 or 0-1)
            for habit_col, encoding in habit_encodings.items():
                feature_vec.append(encoding[row[habit_col]] / 3.0)
            
            # Years with issue (normalized)
            feature_vec.append(row['num_years_with_issue'] / 30.0)
            
            # Currently treated (binary)
            feature_vec.append(1.0 if row['currently_treated'] else 0.0)
            
            features.append(feature_vec)
        
        self.user_feature_matrix = np.array(features)
        return self.user_feature_matrix
    
    def extract_item_features(self) -> np.ndarray:
        """
        Convert item content to feature vector
        
        Features: category (one-hot), product features, rating
        Returns: (n_items, n_features) matrix
        """
        
        categories = sorted(self.items['category'].unique())
        
        features = []
        for idx, row in self.items.iterrows():
            feature_vec = []
            
            # Category (one-hot)
            cat_vec = [1.0 if cat == row['category'] else 0.0 for cat in categories]
            feature_vec.extend(cat_vec)
            
            # Average rating (normalized 0-1)
            feature_vec.append((row['average_rating'] - 1.0) / 4.0)
            
            # Parsed features from JSON
            try:
                features_dict = json.loads(row['features_json'])
                
                # Count boolean features (presence of key features)
                bool_features = sum(1 for v in features_dict.values() if isinstance(v, bool) and v)
                max_bool_features = sum(1 for v in features_dict.values() if isinstance(v, bool))
                feature_vec.append(bool_features / max(max_bool_features, 1))
                
                # Extract numeric features if present
                numeric_features = [v for v in features_dict.values() 
                                   if isinstance(v, (int, float))]
                if numeric_features:
                    feature_vec.append(np.mean(numeric_features) / 100.0)  # Normalized
                else:
                    feature_vec.append(0.0)
                
            except:
                feature_vec.extend([0.0, 0.0])
            
            # Number of reviews (popularity)
            feature_vec.append(min(row['num_reviews'] / 500.0, 1.0))
            
            # Pad with zeros to match user feature dimension (20)
            target_dim = 20
            current_dim = len(feature_vec)
            if current_dim < target_dim:
                feature_vec.extend([0.0] * (target_dim - current_dim))
            
            features.append(feature_vec)
        
        self.item_feature_matrix = np.array(features)
        return self.item_feature_matrix

# ============================================================================
# CONTENT-BASED RECOMMENDER
# ============================================================================

class ContentBasedRecommender:
    """Content-based recommendation engine using user and item features"""
    
    def __init__(self, user_features, item_features, items_df):
        self.user_features = user_features
        self.item_features = item_features
        self.items_df = items_df
        self.similarity_matrix = None
        
        # Compute similarity matrix (items × users)
        self._compute_similarities()
    
    def _compute_similarities(self):
        """Compute cosine similarity between all users and items"""
        self.similarity_matrix = cosine_similarity(self.user_features, self.item_features)
        print(f"[OK] Computed similarity matrix: {self.similarity_matrix.shape}")
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True, ratings_df: pd.DataFrame = None) -> List[Dict]:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID to recommend for
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude items user already rated
            ratings_df: Rating dataframe to exclude rated items
        
        Returns:
            List of dicts with item_id, name, category, similarity_score
        """
        
        user_idx = user_id - 1  # Convert to 0-indexed
        
        if user_idx < 0 or user_idx >= len(self.user_features):
            raise ValueError(f"User ID {user_id} not found")
        
        # Get similarity scores for this user
        scores = self.similarity_matrix[user_idx]
        
        # Exclude already-rated items
        if exclude_rated and ratings_df is not None:
            user_rated = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'].values)
            scores = np.array([
                score if (idx + 1) not in user_rated else -np.inf 
                for idx, score in enumerate(scores)
            ])
        
        # Get top-K items
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] == -np.inf:
                continue
            
            item = self.items_df.iloc[idx]
            recommendations.append({
                'rank': rank,
                'item_id': item['item_id'],
                'name': item['name'],
                'category': item['category'],
                'similarity_score': float(scores[idx]),
                'average_rating': item['average_rating'],
                'num_reviews': item['num_reviews'],
            })
        
        return recommendations

# ============================================================================
# EVALUATION
# ============================================================================

class RecommendationEvaluator:
    """Evaluate recommendation quality"""
    
    def __init__(self, recommendations_list: List[List[Dict]], 
                 ratings_df: pd.DataFrame, ratings_threshold: float = 4.0):
        """
        Args:
            recommendations_list: List of recommendation lists (one per user)
            ratings_df: Actual user-item ratings
            ratings_threshold: Rating threshold for "relevant" items
        """
        self.recommendations = recommendations_list
        self.ratings = ratings_df
        self.threshold = ratings_threshold
    
    def precision_at_k(self, k: int = 10) -> float:
        """Fraction of top-K recommendations that user liked"""
        
        precisions = []
        for user_id, recs in enumerate(self.recommendations, 1):
            user_ratings = self.ratings[self.ratings['user_id'] == user_id]
            relevant_items = set(user_ratings[user_ratings['rating'] >= self.threshold]['item_id'])
            
            recommended_items = {rec['item_id'] for rec in recs[:k]}
            
            if len(recommended_items) > 0:
                precision = len(relevant_items & recommended_items) / len(recommended_items)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(self, k: int = 10) -> float:
        """Fraction of user's liked items in top-K"""
        
        recalls = []
        for user_id, recs in enumerate(self.recommendations, 1):
            user_ratings = self.ratings[self.ratings['user_id'] == user_id]
            relevant_items = set(user_ratings[user_ratings['rating'] >= self.threshold]['item_id'])
            
            if len(relevant_items) == 0:
                continue
            
            recommended_items = {rec['item_id'] for rec in recs[:k]}
            recall = len(relevant_items & recommended_items) / len(relevant_items)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def catalog_coverage(self) -> float:
        """Fraction of catalog that gets recommended"""
        
        all_recommended = set()
        for recs in self.recommendations:
            all_recommended.update(rec['item_id'] for rec in recs)
        
        total_items = self.ratings['item_id'].max()
        return len(all_recommended) / total_items

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    print("=" * 70)
    print("Sleep Quality Recommendation Engine - Implementation Example")
    print("=" * 70)
    
    # 1. Load data
    print("\n1. Loading data...")
    loader = DataLoader(data_dir='../data/')
    users, items, ratings, reviews = loader.load_all()
    
    # 2. Extract features
    print("\n2. Extracting features...")
    extractor = FeatureExtractor(users, items)
    user_features = extractor.extract_user_features()
    item_features = extractor.extract_item_features()
    
    print(f"   User feature matrix: {user_features.shape}")
    print(f"   Item feature matrix: {item_features.shape}")
    
    # 3. Build recommender
    print("\n3. Building recommender...")
    recommender = ContentBasedRecommender(user_features, item_features, items)
    
    # 4. Generate recommendations for sample users
    print("\n4. Generating recommendations for sample users...")
    sample_users = [1, 42, 123, 999]
    
    for user_id in sample_users:
        print(f"\n   User {user_id}:")
        user = users[users['user_id'] == user_id].iloc[0]
        print(f"   Issue: {user['primary_sleep_issue']}, Age: {user['age']}, Stress: {user['stress_level']}")
        
        recommendations = recommender.recommend(user_id, n_recommendations=5, 
                                               exclude_rated=True, ratings_df=ratings)
        
        for rec in recommendations:
            print(f"     [{rec['rank']}] {rec['name']} ({rec['category']}) - {rec['similarity_score']:.3f}")
    
    print("\n" + "=" * 70)
    print("[OK] Recommendation engine working successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
