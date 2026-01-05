# Section 1: Dimensionality Reduction for Recommender Systems

## 1. Overview
This section focuses on applying **Dimensionality Reduction** techniques—specifically **Singular Value Decomposition (SVD)** and **Principal Component Analysis (PCA)**—to the Amazon Movies & TV dataset. The primary goal is to address the **Sparsity Problem** inherent in large-scale recommender systems and to extract meaningful **Latent Factors** that capture user preferences and item characteristics.

## 2. Objectives
1.  **Reduce Dimensionality:** Compress the sparse user-item matrix into a dense, low-rank approximation.
2.  **Handle Sparsity:** Evaluate imputation strategies (Mean-Filling vs. MLE) to manage missing data.
3.  **Predict Ratings:** Use the decomposed matrices ($U, \Sigma, V$) to predict missing ratings for target users.
4.  **Evaluate Performance:** Compare SVD against PCA and baseline Collaborative Filtering (KNN) methods.

## 3. Methodology

### 3.1 Data Preparation
-   **Dataset:** Amazon Movies & TV ratings (~8.7M ratings).
-   **Filtering:** Retained the **Top 10,000** most active users and **Top 1,000** most popular items to ensure a dense core for analysis.
-   **Target Items:** `B00PCSVODW` and `B005GISDXW`
-   **Preprocessing:**
    -   Converted raw ratings to a User-Item Matrix.
    -   **Imputation:** Applied **Item-Mean Filling** to handle missing values, effectively converting the sparse matrix into a dense format suitable for standard SVD.

### 3.2 SVD Implementation
We implemented **SVD (Singular Value Decomposition)** from scratch using Eigenvalue Decomposition of the correlation matrix ($R^T R$):
-   **Decomposition:** $R = U \Sigma V^T$
-   **Orthogonality Check:** Verified $U^T U = I$ and $V^T V = I$ to ensure mathematical correctness.
-   **Truncation:** Constructed low-rank approximations by keeping only the top $k$ singular values.

### 3.3 PCA Mean-Filling
-   Computed item-wise covariance matrix on mean-centered data.
-   Identified **Top-5** and **Top-10** peer items by covariance for target items.
-   Predicted missing ratings using item-based collaborative filtering weighted by covariance.

### 3.4 PCA MLE (Maximum Likelihood Estimation)
-   Computed pairwise covariance using MLE on overlapping user ratings only.
-   Constructed covariance matrix for 1,002 items.
-   Used Gaussian conditional distribution for prediction: $\hat{r}(u, i) = \mu_i + \Sigma_{iP} \Sigma_{PP}^{-1} (r_P - \mu_P)$

## 4. Key Findings

### 4.1 Optimal Latent Factors ($k$)
-   **Selected $k = 50$**
-   **Justification:** Analysis of the **Elbow Curve** (RMSE vs. $k$) and **Variance Retained** plot showed that at $k=50$, the model captures approximately **90%** of the total variance. Increasing $k$ further yields diminishing returns in accuracy while increasing computational cost.

### 4.2 Prediction Performance
-   **Targets:** Predicted ratings for items `B00PCSVODW` and `B005GISDXW`.
-   **Accuracy:** The Truncated SVD model provided robust predictions (Scale 1-5).
-   **Comparison:** SVD predictions were comparable to **User-Based KNN** (Assignment 1 baseline), demonstrating that global latent factors can effectively approximate local neighborhood-based predictions.

### 4.3 Cold-Start Analysis
-   **Scenario:** Simulated "Cold-Start" users by hiding 80% of their ratings.
-   **Method:** Used **Projection (Folding-in)** technique: $u_{new} = r_{new} V \Sigma^{-1}$.
-   **Result:** A **Hybrid Strategy** (50% SVD Projection + 50% Global Item Means) significantly reduced RMSE compared to pure SVD projection for users with very sparse history.

### 4.4 PCA Peer Analysis
-   **Top-5 Peers for B00PCSVODW:** B00TKIJGDA, B000006B4Y, B00HSJ2CVQ, B00NCDVVLY, B0090JBOC0
-   **Top-5 Peers for B005GISDXW:** 6305480869, B00ZL4Q7NE, B00HW3EI3I, B00BTFK07I, B00FL31UF0
-   Cosine similarity computed alongside covariance for validation.

## 5. Comparative Analysis

| Feature | SVD (Truncated) | PCA (Mean-Filled) | PCA (MLE) | KNN (User-Based) |
| :--- | :--- | :--- | :--- | :--- |
| **Approach** | Matrix Factorization | Linear Transformation | Gaussian Conditional | Memory-Based Iteration |
| **Sparsity Handling** | Requires Imputation | Requires Imputation | Overlap-only computation | Native Handling |
| **Scalability** | Good (Fast Inference) | Good (Fast Inference) | Moderate | Poor (Slow Inference) |
| **Space Complexity** | Moderate ($O(k \cdot (m+n))$) | Low (Covariance Matrix) | Low (Covariance Matrix) | High (Full Matrix) |
| **Cold-Start** | Projection Strategy | Projection Strategy | Conditional Mean | Re-computation |

**Conclusion:** SVD with Mean-Filling is the preferred method for this dataset due to its balance of accuracy, scalability, and ability to extract interpretable latent features.

## 6. Project Structure

```
SECTION1_DimensionalityReduction/
├── code/
│   ├── SVD_Analysis .ipynb      # Full SVD implementation & analysis
│   ├── pca_mean_filling.ipynb   # PCA with mean-filling imputation
│   └── pca_mle .ipynb           # PCA with MLE covariance estimation
├── data/
│   └── Movies_and_TV.csv        # Amazon Movies & TV dataset
├── results/
│   ├── plots/
│   │   ├── svd_scree_plot.png
│   │   ├── truncated_svd_elbow.png
│   │   ├── latent_space_visualization.png
│   │   ├── robustness_sparsity.png
│   │   ├── rating_distribution.png
│   │   ├── filling_comparison.png
│   │   ├── pca_analysis_combined.png
│   │   └── prediction_comparison.png
│   └── tables/
│       ├── pca_eigenvalues_top5.csv
│       ├── pca_eigenvalues_top10.csv
│       ├── pca_peers_I1_top5.csv
│       ├── pca_peers_I1_top10.csv
│       ├── pca_peers_I2_top5.csv
│       ├── pca_peers_I2_top10.csv
│       └── pca_target_means.csv
└── README_SECTION1.md
```

## 7. Notebooks Description

### 7.1 SVD_Analysis .ipynb
Complete implementation of Singular Value Decomposition:
- Data Loading & Cleaning (8.7M ratings)
- Sparse Matrix Construction (10,000 users × 1,000 items)
- Full & Truncated SVD Implementation
- Orthogonality Verification
- Visualization (Scree Plots, Latent Space)
- Rating Prediction & Evaluation
- Sensitivity & Cold-Start Analysis

### 7.2 pca_mean_filling.ipynb
PCA implementation with mean-filling strategy:
- Target Items: B00PCSVODW, B005GISDXW
- Item-mean computation for all 182,032 items
- Covariance-based peer selection
- Top-5 and Top-10 peer identification
- Weighted prediction using covariance
- Cosine similarity validation

### 7.3 pca_mle .ipynb
PCA implementation using Maximum Likelihood Estimation:
- Pairwise MLE covariance computation on overlapping users only
- Covariance matrix (1002 × 1002 items)
- Gaussian conditional distribution for prediction
- Top-5 vs Top-10 peer comparison
- Reduced space projection