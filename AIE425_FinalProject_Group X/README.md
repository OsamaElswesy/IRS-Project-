# Intelligent Recommender Systems Project

## Repository Structure

```
AIE425_FinalProject_Group X/
├── AIE425_Intelligent Recommender Systems/
│   ├── SECTION1_DimensionalityReduction/
│   │   ├── code/
│   │   ├── plots/
│   │   ├── tables/
│   │   └── README_SECTION1.md
│   └── SECTION2_DomainRecommender/
├── Statistical_Analysis/
├── requirements.txt
└── README.md
```

## Section 1: Dimensionality Reduction Analysis Results

### 1. Visualization of Latent Factors
**Scree Plot & Elbow Method:** used to determine the optimal $k=50$.

![Scree Plot](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/plots/svd_scree_plot.png)
![Elbow Curve](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/plots/truncated_svd_elbow.png)

### 2. Latent Space
Visualizing Users (Blue) and Items (Red) in the first 2 latent dimensions.

![Latent Space](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/plots/latent_space_visualization.png)

### 3. Key Results Table

| Metric | SVD ($k=50$) | KNN (Baseline) |
| :--- | :--- | :--- |
| **MAE** | **0.281** | 0.315 |
| **RMSE** | **0.342** | 0.402 |
| **Runtime** | ~6.0s | >60s |

### 4. Robustness to Sparsity
SVD performance degradation as data sparsity increases (Mean-Filling strategy).

![Robustness](AIE425_Intelligent%20Recommender%20Systems/SECTION1_DimensionalityReduction/plots/robustness_sparsity.png)