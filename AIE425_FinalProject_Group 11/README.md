# Intelligent Recommender Systems Project

data link: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV.csv
## Repository Structure

```
AIE425_FinalProject_Group 11/
├── AIE425_Intelligent Recommender Systems/
│   ├── SECTION1_DimensionalityReduction/
│   │   ├── code/
│   │   │   ├── SVD_Analysis .ipynb
│   │   │   ├── pca_mean_filling.ipynb
│   │   │   └── pca_mle .ipynb
│   │   ├── data/
│   │   ├── results/
│   │   │   ├── plots/
│   │   │   │   ├── svd_scree_plot.png
│   │   │   │   ├── truncated_svd_elbow.png
│   │   │   │   ├── latent_space_visualization.png
│   │   │   │   ├── robustness_sparsity.png
│   │   │   │   ├── rating_distribution.png
│   │   │   │   ├── filling_comparison.png
│   │   │   │   ├── pca_analysis_combined.png
│   │   │   │   └── prediction_comparison.png
│   │   │   └── tables/
│   │   │       ├── pca_eigenvalues_top5.csv
│   │   │       ├── pca_eigenvalues_top10.csv
│   │   │       ├── pca_peers_I1_top5.csv
│   │   │       ├── pca_peers_I1_top10.csv
│   │   │       ├── pca_peers_I2_top5.csv
│   │   │       ├── pca_peers_I2_top10.csv
│   │   │       └── pca_target_means.csv
│   │   └── README_SECTION1.md
│   └── SECTION2_Sleep Quality Improvement Recommendation Engine/
│       ├── code/
│       │   ├── Data_preprocessing.ipynb
│       │   ├── Content_based.ipynb
│       │   └── Collaborative Filtering & Hybrid Approach .ipynb
│       ├── data/
│       ├── plots/
│       └── results/
│           └── README_SECTION2.md
├── Statistical_Analysis/
│   ├── Statistical_Analysis_Refactored.ipynb
│   └── Results/
├── requirements.txt
└── README.md
```

## Section 1: Dimensionality Reduction for Recommender Systems

### 1. Overview
This section focuses on applying **Dimensionality Reduction** techniques—specifically **Singular Value Decomposition (SVD)** and **Principal Component Analysis (PCA)**—to the Amazon Movies & TV dataset. The primary goal is to address the **Sparsity Problem** inherent in large-scale recommender systems and to extract meaningful **Latent Factors** that capture user preferences and item characteristics.

### 2. Objectives
1.  **Reduce Dimensionality:** Compress the sparse user-item matrix into a dense, low-rank approximation.
2.  **Handle Sparsity:** Evaluate imputation strategies (Mean-Filling vs. MLE) to manage missing data.
3.  **Predict Ratings:** Use the decomposed matrices ($U, \Sigma, V$) to predict missing ratings for target users.
4.  **Evaluate Performance:** Compare SVD against PCA and baseline Collaborative Filtering (KNN) methods.

### 3. Methodology

#### 3.1 Data Preparation
-   **Dataset:** Amazon Movies & TV ratings (~8.7M ratings).
-   **Filtering:** Retained the **Top 10,000** most active users and **Top 1,000** most popular items to ensure a dense core for analysis.
-   **Target Items:** `B00PCSVODW` and `B005GISDXW`
-   **Preprocessing:**
    -   Converted raw ratings to a User-Item Matrix.
    -   **Imputation:** Applied **Item-Mean Filling** to handle missing values.

#### 3.2 SVD Implementation
We implemented **SVD (Singular Value Decomposition)** from scratch using Eigenvalue Decomposition of the correlation matrix ($R^T R$):
-   **Decomposition:** $R = U \Sigma V^T$
-   **Orthogonality Check:** Verified $U^T U = I$ and $V^T V = I$ to ensure mathematical correctness.
-   **Truncation:** Constructed low-rank approximations by keeping only the top $k$ singular values.

#### 3.3 PCA Mean-Filling
-   Computed item-wise covariance matrix on mean-centered data.
-   Identified **Top-5** and **Top-10** peer items by covariance for target items.
-   Predicted missing ratings using item-based collaborative filtering weighted by covariance.

#### 3.4 PCA MLE (Maximum Likelihood Estimation)
-   Computed pairwise covariance using MLE on overlapping user ratings only.
-   Constructed covariance matrix for 1,002 items.
-   Used Gaussian conditional distribution for prediction.

### 4. Key Findings

#### 4.1 Optimal Latent Factors ($k$)
-   **Selected $k = 50$**
-   **Justification:** Analysis of the **Elbow Curve** (RMSE vs. $k$) and **Variance Retained** plot showed that at $k=50$, the model captures approximately **90%** of the total variance.

**Scree Plot & Elbow Method:**

![Scree Plot](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/results/plots/svd_scree_plot.png)
![Elbow Curve](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/results/plots/truncated_svd_elbow.png)

#### 4.2 Latent Space Visualization
Visualizing Users (Blue) and Items (Red) in the first 2 latent dimensions.

![Latent Space](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/results/plots/latent_space_visualization.png)

#### 4.3 Prediction Performance
-   **Targets:** Predicted ratings for items `B00PCSVODW` and `B005GISDXW`.
-   **Accuracy:** The Truncated SVD model provided robust predictions (Scale 1-5).
-   **Comparison:** SVD predictions were comparable to **User-Based KNN** baseline.

| Metric | SVD ($k=50$) | KNN (Baseline) |
| :--- | :--- | :--- |
| **MAE** | **0.281** | 0.315 |
| **RMSE** | **0.342** | 0.402 |
| **Runtime** | ~6.0s | >60s |

#### 4.4 Robustness to Sparsity
SVD performance degradation as data sparsity increases (Mean-Filling strategy).

![Robustness](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/results/plots/robustness_sparsity.png)

#### 4.5 Cold-Start Analysis
-   **Scenario:** Simulated "Cold-Start" users by hiding 80% of their ratings.
-   **Method:** Used **Projection (Folding-in)** technique: $u_{new} = r_{new} V \Sigma^{-1}$.
-   **Result:** A **Hybrid Strategy** (50% SVD Projection + 50% Global Item Means) significantly reduced RMSE.

#### 4.6 PCA Mean-Filling Analysis
Analysis of the dataset distribution and the mean-filling technique used for PCA.

**Data Distribution & Sparsity:**
The rating distribution (left) and user rating frequency (right) highlights the long-tail nature of the dataset.

![Rating Distribution](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/results/plots/rating_distribution.png)

**Mean-Filling Visualization:**
Comparison of a subset of the ratings matrix before (raw) and after mean-filling.

![Filling Comparison](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/results/plots/filling_comparison.png)

**PCA Analysis:**
Includes the distribution of item means, the covariance matrix of top items, and the scree plot showing eigenvalue decay.

![PCA Analysis](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/results/plots/pca_analysis_combined.png)

#### 4.7 Prediction Comparison
Comparison of prediction methods across Top-5 and Top-10 peers.

![Prediction Comparison](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/results/plots/prediction_comparison.png)

### 5. Comparative Analysis

| Feature | SVD (Truncated) | PCA (Mean-Filled) | PCA (MLE) | KNN (User-Based) |
| :--- | :--- | :--- | :--- | :--- |
| **Approach** | Matrix Factorization | Linear Transformation | Gaussian Conditional | Memory-Based Iteration |
| **Sparsity Handling** | Requires Imputation | Requires Imputation | Overlap-only computation | Native Handling |
| **Scalability** | Good (Fast Inference) | Good (Fast Inference) | Moderate | Poor (Slow Inference) |
| **Space Complexity** | Moderate ($O(k \cdot (m+n))$) | Low (Covariance Matrix) | Low (Covariance Matrix) | High (Full Matrix) |
| **Cold-Start** | Projection Strategy | Projection Strategy | Conditional Mean | Re-computation |

**Conclusion:** SVD with Mean-Filling is the preferred method for this dataset due to its balance of accuracy, scalability, and ability to extract interpretable latent features.

### 6. Project Structure

-   `code/SVD_Analysis .ipynb`: Full SVD implementation with data loading, decomposition, visualization, and cold-start analysis.
-   `code/pca_mean_filling.ipynb`: PCA with mean-filling imputation, covariance-based peer selection, and rating prediction.
-   `code/pca_mle .ipynb`: PCA using MLE covariance estimation on overlapping users with Gaussian conditional prediction.
-   `results/`: Directory containing generated prediction tables and plots.

## Section 2: Sleep Quality Improvement Recommendation Engine

### 1. Project Overview
This project focuses on building an **Intelligent Recommender System** to suggest sleep quality improvement products and applications. By analyzing user profiles, sleep habits, and product features, the system recommends personalized solutions such as sleep tracking apps, meditation guides, and smart sleep devices.

The goal is to leverage machine learning techniques including **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Approach** to enhance user well-being through better sleep.

### 2. Dataset Description
The project uses a synthetic dataset comprising three main components:

- **Users (`users.csv`)**: Contains profiles of **5,000 users**.
    - Attributes include: `user_id`, `age`, `gender`, `primary_sleep_issue`, `stress_level`, `sleep_schedule`, `caffeine_intake`, and more.
- **Items (`items.csv`)**: Contains details of **520 sleep-related products/apps**.
    - Attributes include: `item_id`, `name` (e.g., "Dream Track v1"), `category` (e.g., Mobile Apps), `description`, `features_json`, and `average_rating`.
- **Ratings (`ratings.csv`)**: Contains **51,488 interactions** between users and items.
    - Attributes include: `rating_id`, `user_id`, `item_id`, `rating` (1-5 scale), and `date`.

### 3. Methodology

#### 3.1 Data Preprocessing & EDA
Located in `AIE425_Intelligent Recommender Systems/SECTION2_Sleep Quality Improvement Recommendation Engine/code/Data_preprocessing.ipynb`.
- **Loading**: Data is loaded from CSV files.
- **Cleaning**: Handling missing values, parsing dates, and checking constraints.
- **Exploratory Data Analysis (EDA)**: Visualizing distributions of user demographics, sleep issues, and item ratings to understand the data landscape.

#### 3.2 Content-Based Filtering
Located in `AIE425_Intelligent Recommender Systems/SECTION2_Sleep Quality Improvement Recommendation Engine/code/Content_based.ipynb`.
- **Concept**: Recommends items similar to those a user has liked in the past, based on item attributes.
- **Technique**:
    - **TF-IDF Vectorization**: Applied to item `description` and `features` to create feature vectors.
    - **Cosine Similarity**: Calculates the similarity between items.
    - **User Profiles**: Built by aggregating vectors of items rated highly by the user.

#### 3.3 Collaborative Filtering
Located in `AIE425_Intelligent Recommender Systems/SECTION2_Sleep Quality Improvement Recommendation Engine/code/Collaborative Filtering & Hybrid Approach .ipynb`.
- **Concept**: Recommends items based on the preferences of similar users.
- **Technique**:
    - **Matrix Factorization (SVD)**: Decomposes the user-item interaction matrix to predict missing ratings.
    - **User-Based / Item-Based Filtering**: Using cosine similarity on interaction vectors.

#### 3.4 Hybrid Approach
Located in `AIE425_Intelligent Recommender Systems/SECTION2_Sleep Quality Improvement Recommendation Engine/code/Collaborative Filtering & Hybrid Approach .ipynb`.
- **Concept**: Combines Content-Based and Collaborative Filtering to overcome limitations of individual models (e.g., cold start).
- **Technique**: Weighted combination of scores from both content-based and collaborative models to produce the final recommendation rank.

### 4. How to Run
1.  **Prerequisites**: Ensure Python 3.x and necessary libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) are installed.
2.  **Order of Execution**:
    1.  Run `Data_preprocessing.ipynb` to clean data and generate insights.
    2.  Run `Content_based.ipynb` to train and test the content-based recommender.
    3.  Run `Collaborative Filtering & Hybrid Approach .ipynb` to execute collaborative filtering and the hybrid model.

### 5. Results
- The system successfully identifies relevant sleep aids for users based on their specific sleep issues (e.g., Insomnia, Anxiety) and past preferences.
- The Hybrid model generally offers more robust recommendations by leveraging both content features and community signals.
