# Sleep Quality Improvement Recommendation Engine

## 1. Project Overview
This project focuses on building an **Intelligent Recommender System** to suggest sleep quality improvement products and applications. By analyzing user profiles, sleep habits, and product features, the system recommends personalized solutions such as sleep tracking apps, meditation guides, and smart sleep devices.

The goal is to leverage machine learning techniques including **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Approach** to enhance user well-being through better sleep.

## 2. Dataset Description
The project uses a synthetic dataset comprising three main components:

- **Users (`users.csv`)**: Contains profiles of **5,000 users**.
    - Attributes include: `user_id`, `age`, `gender`, `primary_sleep_issue`, `stress_level`, `sleep_schedule`, `caffeine_intake`, and more.
- **Items (`items.csv`)**: Contains details of **520 sleep-related products/apps**.
    - Attributes include: `item_id`, `name` (e.g., "Dream Track v1"), `category` (e.g., Mobile Apps), `description`, `features_json`, and `average_rating`.
- **Ratings (`ratings.csv`)**: Contains **51,488 interactions** between users and items.
    - Attributes include: `rating_id`, `user_id`, `item_id`, `rating` (1-5 scale), and `date`.

## 3. Methodology

### 3.1 Data Preprocessing & EDA
Located in `code/Data_preprocessing.ipynb`.
- **Loading**: Data is loaded from CSV files.
- **Cleaning**: Handling missing values, parsing dates, and checking constraints.
- **Exploratory Data Analysis (EDA)**: Visualizing distributions of user demographics, sleep issues, and item ratings to understand the data landscape.

### 3.2 Content-Based Filtering
Located in `code/Content_based.ipynb`.
- **Concept**: Recommends items similar to those a user has liked in the past, based on item attributes.
- **Technique**:
    - **TF-IDF Vectorization**: Applied to item `description` and `features` to create feature vectors.
    - **Cosine Similarity**: Calculates the similarity between items.
    - **User Profiles**: Built by aggregating vectors of items rated highly by the user.

### 3.3 Collaborative Filtering
Located in `code/Collaborative Filtering & Hybrid Approach .ipynb`.
- **Concept**: Recommends items based on the preferences of similar users.
- **Technique**:
    - **Matrix Factorization (SVD)**: Decomposes the user-item interaction matrix to predict missing ratings.
    - **User-Based / Item-Based Filtering**: Using cosine similarity on interaction vectors.

### 3.4 Hybrid Approach
Located in `code/Collaborative Filtering & Hybrid Approach .ipynb`.
- **Concept**: Combines Content-Based and Collaborative Filtering to overcome limitations of individual models (e.g., cold start).
- **Technique**: Weighted combination of scores from both content-based and collaborative models to produce the final recommendation rank.

## 4. How to Run
1.  **Prerequisites**: Ensure Python 3.x and necessary libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) are installed.
2.  **Order of Execution**:
    1.  Run `Data_preprocessing.ipynb` to clean data and generate insights.
    2.  Run `Content_based.ipynb` to train and test the content-based recommender.
    3.  Run `Collaborative Filtering & Hybrid Approach .ipynb` to execute collaborative filtering and the hybrid model.

## 5. Results
- The system successfully identifies relevant sleep aids for users based on their specific sleep issues (e.g., Insomnia, Anxiety) and past preferences.
- The Hybrid model generally offers more robust recommendations by leveraging both content features and community signals.