"""
Sleep Quality Improvement Recommendation Engine - Dataset Generator
Generates synthetic data meeting the project requirements:
- 5,000+ users
- 500+ items (sleep interventions/products)
- 50,000+ interactions/ratings with reviews
- Rich content features for content-based recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# 1. GENERATE USERS (5,000+ users)
# ============================================================================

def generate_users(n_users=5000):
    """Generate user profiles with sleep issues and lifestyle habits"""
    
    sleep_issues = [
        'Insomnia', 'Sleep Apnea', 'Restless Legs', 'Night Sweats',
        'Difficulty Falling Asleep', 'Frequent Awakenings', 'Early Morning Awakening',
        'Non-Restorative Sleep', 'Shift Work Sleep', 'Jet Lag'
    ]
    
    habits = {
        'Exercise Frequency': ['Never', 'Occasionally', '2-3x/week', 'Daily'],
        'Caffeine Intake': ['None', 'Moderate', 'High', 'Very High'],
        'Screen Time Before Bed': ['<30 min', '30-60 min', '1-2 hours', '>2 hours'],
        'Stress Level': ['Low', 'Moderate', 'High', 'Very High'],
        'Sleep Schedule': ['Irregular', 'Somewhat Regular', 'Regular', 'Very Regular'],
        'Alcohol Consumption': ['Never', 'Occasionally', 'Regular', 'Heavy'],
        'Napping Habit': ['No naps', 'Occasional', 'Regular short naps', 'Long naps'],
    }
    
    users = []
    for user_id in range(1, n_users + 1):
        user = {
            'user_id': user_id,
            'age': np.random.randint(18, 75),
            'gender': np.random.choice(['Male', 'Female']),
            'primary_sleep_issue': np.random.choice(sleep_issues),
            'secondary_sleep_issue': np.random.choice(sleep_issues),
            'exercise_frequency': np.random.choice(habits['Exercise Frequency']),
            'caffeine_intake': np.random.choice(habits['Caffeine Intake']),
            'screen_time_before_bed': np.random.choice(habits['Screen Time Before Bed']),
            'stress_level': np.random.choice(habits['Stress Level']),
            'sleep_schedule': np.random.choice(habits['Sleep Schedule']),
            'alcohol_consumption': np.random.choice(habits['Alcohol Consumption']),
            'napping_habit': np.random.choice(habits['Napping Habit']),
            'num_years_with_issue': np.random.randint(1, 30),
            'currently_treated': np.random.choice([True, False], p=[0.6, 0.4]),
        }
        users.append(user)
    
    return pd.DataFrame(users)

# ============================================================================
# 2. GENERATE ITEMS (500+ sleep interventions/products)
# ============================================================================

def generate_items(n_items=520):
    """Generate sleep interventions and products with rich content features"""
    
    categories = [
        'Mobile Apps', 'Wearable Devices', 'Supplements', 'Sleep Aids',
        'Bedding & Accessories', 'Lifestyle Practices', 'Medical Devices',
        'Sound & Environment', 'Therapy & Counseling'
    ]
    
    app_names = [
        'Calm Sleep', 'Dream Track', 'SlumberAI', 'Restful', 'Sleep Mastery',
        'Zensleep', 'NightGuard', 'PeakSleep', 'SleepWell', 'CircadianSync',
        'MindRest', 'SleepCycle Pro', 'Insomnia Coach', 'Breathe Easy', 'Relax & Restore'
    ]
    
    device_names = [
        'SmartBand Pro', 'SleepTracker X', 'BiometricWatch', 'OuraRing Plus',
        'FitSleep Monitor', 'NightSense Band', 'WellnessPod', 'SleepAnalyzer',
        'HeartRate Monitor Elite', 'WearableSleep'
    ]
    
    supplement_names = [
        'Melatonin 5mg', 'Magnesium Glycinate', 'Valerian Root Extract',
        'Passionflower Supplement', 'Ashwagandha Sleep Blend', 'L-Theanine Pure',
        'GABA Supplement', 'Phosphatidylserine', 'Chamomile Extract', 'Lavender Oil'
    ]
    
    device_brands = ['Apple', 'Oura', 'Fitbit', 'Garmin', 'Samsung', 'Whoop', 'Eight Sleep']
    
    accessory_names = [
        'Cooling Pillow Pro', 'Blackout Curtains', 'White Noise Machine',
        'Sleep Mask Premium', 'Weighted Blanket', 'Memory Foam Mattress',
        'Air Purifier', 'Humidifier', 'Aromatherapy Diffuser', 'Blue Light Glasses'
    ]
    
    therapy_names = [
        'CBT-I Program', 'Meditation Course', 'Sleep Hypnotherapy',
        'Yoga for Sleep', 'Sleep Coaching Session', 'Mindfulness Training',
        'Breathing Technique Workshop', 'Sleep Psychology Course'
    ]
    
    def get_item_details(item_id, category):
        """Generate detailed content features for each item"""
        
        if category == 'Mobile Apps':
            name = f"{app_names[item_id % len(app_names)]} v{(item_id // 10) + 1}"
            features = {
                'has_sleep_tracking': np.random.choice([True, False]),
                'has_meditation': np.random.choice([True, False]),
                'has_smart_alarm': np.random.choice([True, False]),
                'has_music_library': np.random.choice([True, False]),
                'has_analytics': np.random.choice([True, False]),
                'has_community': np.random.choice([True, False]),
                'subscription_required': np.random.choice([True, False]),
                'price_tier': np.random.choice(['Free', '$4.99/mo', '$9.99/mo', '$14.99/mo']),
                'supported_devices': np.random.choice(['iOS only', 'Android only', 'Both', 'Web only']),
            }
            description = f"{name} is a sleep app with meditation, tracking, and personalized recommendations. Features AI-powered sleep analysis and adaptive sleep programs."
            
        elif category == 'Wearable Devices':
            name = f"{device_brands[item_id % len(device_brands)]} {device_names[item_id % len(device_names)]}"
            features = {
                'has_heart_rate': np.random.choice([True, False]),
                'has_temperature': np.random.choice([True, False]),
                'has_SpO2': np.random.choice([True, False]),
                'has_movement': np.random.choice([True, False]),
                'battery_life_days': np.random.randint(1, 14),
                'water_resistant': np.random.choice([True, False]),
                'price_range': np.random.choice(['$99-150', '$150-250', '$250-400', '>$400']),
                'sleep_stages_detected': np.random.choice([True, False]),
                'has_haptic_feedback': np.random.choice([True, False]),
            }
            description = f"{name} is a wearable sleep tracker with advanced biometric sensors. Provides detailed sleep stage analysis, heart rate variability tracking, and actionable insights."
            
        elif category == 'Supplements':
            name = f"{supplement_names[item_id % len(supplement_names)]}"
            features = {
                'dosage_mg': np.random.choice([5, 10, 20, 30, 50, 100, 200, 500]),
                'natural': np.random.choice([True, False]),
                'vegan': np.random.choice([True, False]),
                'gluten_free': np.random.choice([True, False]),
                'third_party_tested': np.random.choice([True, False]),
                'contains_melatonin': np.random.choice([True, False]),
                'price_per_bottle': np.random.choice(['$8', '$12', '$15', '$20', '$25']),
                'capsules_per_bottle': np.random.choice([30, 60, 90, 120]),
                'requires_prescription': False,
            }
            description = f"{name} is a premium sleep supplement designed to promote natural sleep onset. Formulated with clinically studied ingredients for safe and effective use."
            
        elif category == 'Sleep Aids':
            name = f"{accessory_names[item_id % len(accessory_names)]}"
            features = {
                'adjustable': np.random.choice([True, False]),
                'temperature_control': np.random.choice([True, False]),
                'wireless': np.random.choice([True, False]),
                'app_controlled': np.random.choice([True, False]),
                'size_options': np.random.choice(['One Size', 'S/M/L', 'Full Range']),
                'price_range': np.random.choice(['$30-50', '$50-100', '$100-200', '>$200']),
                'washable': np.random.choice([True, False]),
                'eco_friendly': np.random.choice([True, False]),
            }
            description = f"{name} is a sleep accessory designed to optimize sleep comfort and environment. Premium materials ensure durability and long-lasting performance."
            
        elif category == 'Lifestyle Practices':
            name = f"{therapy_names[item_id % len(therapy_names)]}"
            features = {
                'duration_weeks': np.random.choice([2, 4, 6, 8, 12]),
                'sessions_per_week': np.random.choice([1, 2, 3, 5, 7]),
                'requires_trainer': np.random.choice([True, False]),
                'online_available': np.random.choice([True, False]),
                'self_paced': np.random.choice([True, False]),
                'evidence_based': np.random.choice([True, False]),
                'price': np.random.choice(['Free', '$29', '$49', '$99', '$199', 'Variable']),
            }
            description = f"{name} is a scientifically-backed sleep improvement program combining behavioral techniques, mindfulness, and lifestyle modifications for lasting results."
            
        else:  # Medical Devices, Sound & Environment, Therapy & Counseling
            name = f"Premium Sleep Solution {item_id}"
            features = {
                'clinical_evidence': np.random.choice([True, False]),
                'fda_approved': np.random.choice([True, False]),
                'requires_prescription': np.random.choice([True, False]),
                'price_range': np.random.choice(['$50-100', '$100-300', '>$300']),
            }
            description = f"Professional-grade sleep improvement solution. Backed by clinical research and trusted by sleep specialists worldwide."
        
        return name, features, description
    
    items = []
    category_idx = 0
    for item_id in range(1, n_items + 1):
        # Distribute items across categories
        category = categories[category_idx % len(categories)]
        if item_id % (n_items // len(categories)) == 0:
            category_idx += 1
        
        name, features, description = get_item_details(item_id, category)
        
        item = {
            'item_id': item_id,
            'name': name,
            'category': category,
            'features_json': json.dumps(features, cls=NumpyEncoder),
            'description': description,
            'average_rating': np.round(np.random.uniform(2.5, 5.0), 2),
            'num_reviews': np.random.randint(10, 500),
            'launch_date': (datetime.now() - timedelta(days=np.random.randint(30, 1000))).strftime('%Y-%m-%d'),
        }
        items.append(item)
    
    return pd.DataFrame(items)

# ============================================================================
# 3. GENERATE RATINGS & REVIEWS (50,000+ interactions)
# ============================================================================

def generate_ratings_and_reviews(users_df, items_df, n_interactions=52000):
    """Generate user-item ratings and review text"""
    
    review_templates = {
        5: [
            "This {item_category} completely changed my sleep life! I'm finally getting quality rest.",
            "Absolutely amazing! Fell asleep faster and slept deeper than ever before.",
            "Worth every penny. I've tried everything and this actually works.",
            "Life-changing product. Highly recommend to anyone with sleep issues.",
            "Best investment in my health I've made. Sleep quality improved dramatically.",
            "Finally found something that works for my {sleep_issue}!",
            "Exceeded all my expectations. Fast results and great customer support.",
            "This is the real deal. Scientific approach to better sleep.",
        ],
        4: [
            "Great product overall. Minor issues but definitely improved my sleep.",
            "Very good quality. Sleep improved noticeably after 2 weeks.",
            "Would recommend. Takes some time to adjust but works well.",
            "Solid choice for sleep improvement. Good value for money.",
            "Effective solution for my sleep problems. Very satisfied.",
            "Quality product with good results. Almost perfect.",
            "Definitely helping with my {sleep_issue}. Happy with purchase.",
            "Great features and reliable. Only minor drawbacks.",
        ],
        3: [
            "It's okay. Helped a bit but not as much as expected.",
            "Average product. Some benefits but had higher hopes.",
            "Works sometimes. Results are inconsistent for me.",
            "Decent option if you want to try something different.",
            "Moderate improvement in sleep. Worth trying.",
            "Neither great nor terrible. Fair value for money.",
            "Helped with {sleep_issue} but not as dramatically as hoped.",
            "Good foundation but might need additional help.",
        ],
        2: [
            "Didn't work for me. Wasted money on this.",
            "Minimal benefits. Disappointed with results.",
            "Not effective for my sleep issues. Returning it.",
            "Expected more based on reviews. Underwhelming.",
            "Doesn't live up to the hype. Poor value.",
            "Only slight improvement if any. Likely won't use again.",
            "Didn't help much with my {sleep_issue}.",
            "Frustrating experience. Didn't match expectations.",
        ],
        1: [
            "Terrible product. Complete waste of money.",
            "Doesn't work at all. Made my sleep worse!",
            "Awful experience. Returning immediately.",
            "This is the worst sleep product I've ever tried.",
            "Absolute scam. Avoid at all costs.",
            "Broke after one week. Terrible quality.",
            "Zero improvement. Biggest regret of my life.",
            "Completely useless for sleep improvement.",
        ]
    }
    
    ratings = []
    reviews = []
    
    user_ids = users_df['user_id'].values
    item_ids = items_df['item_id'].values
    
    # Create interactions
    rated_pairs = set()
    for _ in range(n_interactions):
        user_id = np.random.choice(user_ids)
        item_id = np.random.choice(item_ids)
        
        # Avoid duplicate ratings
        if (user_id, item_id) not in rated_pairs:
            rated_pairs.add((user_id, item_id))
            
            # Generate rating (biased towards higher ratings)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.3, 0.35])
            
            rating_date = (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d')
            
            ratings.append({
                'rating_id': len(ratings) + 1,
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'date': rating_date,
            })
            
            # Generate review text for 80% of ratings
            if np.random.random() < 0.8:
                user_issue = users_df[users_df['user_id'] == user_id]['primary_sleep_issue'].values[0]
                item_category = items_df[items_df['item_id'] == item_id]['category'].values[0]
                
                template = np.random.choice(review_templates[rating])
                review_text = template.format(item_category=item_category, sleep_issue=user_issue)
                
                # Add some variation
                if np.random.random() < 0.3:
                    additional = np.random.choice([
                        f" Used for {np.random.randint(1, 12)} weeks.",
                        f" Compared to other solutions, this is better.",
                        f" Would rate {rating} out of 5 stars.",
                        f" Combination with other methods works best.",
                        f" Customer service was responsive and helpful.",
                    ])
                    review_text += additional
                
                reviews.append({
                    'review_id': len(reviews) + 1,
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating,
                    'review_text': review_text,
                    'date': rating_date,
                    'helpful_votes': np.random.randint(0, 50),
                    'verified_purchase': np.random.choice([True, False], p=[0.85, 0.15]),
                })
    
    return pd.DataFrame(ratings), pd.DataFrame(reviews)

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    print("Generating Sleep Quality Improvement Recommendation Engine Dataset...")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating 5,000 user profiles...")
    users_df = generate_users(n_users=5000)
    print(f"   [OK] Created {len(users_df)} user records")
    
    print("\n2. Generating 520 sleep intervention items...")
    items_df = generate_items(n_items=520)
    print(f"   [OK] Created {len(items_df)} item records")
    
    print("\n3. Generating 52,000+ user-item interactions and reviews...")
    ratings_df, reviews_df = generate_ratings_and_reviews(users_df, items_df, n_interactions=52000)
    print(f"   [OK] Created {len(ratings_df)} ratings")
    print(f"   [OK] Created {len(reviews_df)} reviews with text")
    
    # Define data directory
    import os
    DATA_DIR = '../data/'
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save to CSV
    print("\n4. Saving datasets to CSV files...")
    users_df.to_csv(f'{DATA_DIR}sleep_users.csv', index=False)
    print(f"   [OK] Saved: {DATA_DIR}sleep_users.csv")
    
    items_df.to_csv(f'{DATA_DIR}sleep_items.csv', index=False)
    print(f"   [OK] Saved: {DATA_DIR}sleep_items.csv")
    
    ratings_df.to_csv(f'{DATA_DIR}sleep_ratings.csv', index=False)
    print(f"   [OK] Saved: {DATA_DIR}sleep_ratings.csv")
    
    reviews_df.to_csv(f'{DATA_DIR}sleep_reviews.csv', index=False)
    print(f"   [OK] Saved: {DATA_DIR}sleep_reviews.csv")
    
    # Generate summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\nUsers:\n  Total: {len(users_df)}")
    print(f"  Age range: {users_df['age'].min()}-{users_df['age'].max()}")
    print(f"  Gender distribution:\n{users_df['gender'].value_counts()}")
    
    print(f"\nItems:\n  Total: {len(items_df)}")
    print(f"  Categories:\n{items_df['category'].value_counts()}")
    
    print(f"\nInteractions:\n  Total ratings: {len(ratings_df)}")
    print(f"  Total reviews with text: {len(reviews_df)}")
    print(f"  Rating distribution:\n{ratings_df['rating'].value_counts().sort_index()}")
    
    print(f"\nSparsity:\n  Density: {len(ratings_df) / (len(users_df) * len(items_df)) * 100:.2f}%")
    
    # Sample data
    print("\n" + "=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)
    
    print("\nSample User:")
    print(users_df.iloc[0].to_string())
    
    print("\n\nSample Item:")
    print(items_df.iloc[0].to_string())
    
    print("\n\nSample Rating:")
    print(ratings_df.iloc[0].to_string())
    
    print("\n\nSample Review:")
    print(reviews_df.iloc[0].to_string())
    
    print("\n" + "=" * 70)
    print("[OK] Dataset generation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
