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
-   **Dataset:** Subset of Amazon Movies & TV ratings.
-   **Filtering:** Retained the **Top 10,000** most active users and **Top 1,000** most popular items to ensure a dense core for analysis.
-   **Preprocessing:**
    -   Converted raw ratings to a User-Item Matrix.
    -   **Imputation:** Applied **Item-Mean Filling** to handle missing values, effectively converting the sparse matrix into a dense format suitable for standard SVD.

### 3.2 SVD Implementation
We implemented **SVD (Singular Value Decomposition)** from scratch using Eigenvalue Decomposition of the correlation matrix ($R^T R$):
-   **Decomposition:** $R = U \Sigma V^T$
-   **Orthogonality Check:** Verified $U^T U = I$ and $V^T V = I$ to ensure mathematical correctness.
-   **Truncation:** Constructed low-rank approximations by keeping only the top $k$ singular values.

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

## 5. Comparative Analysis

| Feature | SVD (Truncated) | PCA (Mean-Filled) | KNN (User-Based) |
| :--- | :--- | :--- | :--- |
| **Approach** | Matrix Factorization | Linear Transformation | Memory-Based Iteration |
| **Sparsity Handling** | Requires Imputation | Requires Imputation | Native Handling |
| **Scalability** | Good (Fast Inference) | Good (Fast Inference) | Poor (Slow Inference) |
| **Space Complexity** | Moderate ($O(k \cdot (m+n))$) | Low (Covariance Matrix) | High (Full Matrix) |
| **Cold-Start** | Projection Strategy | Projection Strategy | Re-computation |

**Conclusion:** SVD with Mean-Filling is the preferred method for this dataset due to its balance of accuracy, scalability, and ability to extract interpretable latent features.

## 6. Project Structure

-   `code/svd_analysis.ipynb`: The primary notebook containing the full implementation:
    -   Data Loading & Cleaning
    -   Full & Truncated SVD Implementation
    -   Visualization (Scree Plots, Latent Space)
    -   Rating Prediction & Evaluation
    -   Sensitivity & Cold-Start Analysis
-   `results/`: Directory containing generated prediction tables.