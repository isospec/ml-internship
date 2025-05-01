# ML Internship Projects

This repository contains code, notebooks and helper scripts for the machine-learning assignment. The focus is on two main projects:

- **Task 1**: **Biomarker Discover**
- **Task 2**: **Biomarker Embedding**

---

## 📁 Repository Structure

```
ml-internship/
├── data/  
│   ├── input/
│   │   ├── exogenous_standards.csv           # Exogenous Standards
│   │   ├── internship_data_matrix.csv        # Main Data Matrix with peak area values of features
│   │   ├── internship_feature_metadata.csv   # Feature Metadata
│   │   ├── intership_acquisition_list.csv    # Sample metadata including batch and run order
│   │   ├── clean_data_matrix.csv             # Pre-processed data matrix obtained from Data Processing
│   ├── glycan_embedding
│   │   ├── glycan_list.csv      # List of discovered glycans
│   │   ├── df_glycan.pkl        # Glycan dataset
│   │   ├── glycan_binding.pkl   # Protein-glycan binding interactions
│   │   ├── N_glycans_df.pkl     # N-glycans sequences to use for control
│   ├── biomarkers
│   │   ├── biomarker_panel.csv  # Final Biomarker Panel obtained from Discriminatory Analysis
│   └──                       
├── notebooks/  
│   ├── TASK1_EDA.ipynb
│   ├── TASK1_data_processing.ipynb  
│   ├── TASK1_discriminatory_analysis.ipynb 
│   └── TASK2_biomarker_embedding.ipynb  
├── src/  
│   └── helpers_TASK1.py          # Utility functions for Task 1
│   └── helpers_TASK2.py          # Utility functions for Task 2  
├── requirements.txt              # Python dependencies  
└── README.md                     # You are here  
```

## Content

1. **TASK 1**: **Biomarker Discovery**
    - *TASK1_EDA.ipynb*: Exploratory Data Analysis
        - Loads the data matrix, feature metadata and sample metadata, then inspects shapes and missing values.
        - Computes key quality metrics—Coefficient of Variation (CV) across pooled QCs and Dispersion Ratio (D-Ratio)—to gauge technical vs. biological variability.
        - Searches for potential isomers (same m/z, different RT) and isotopes/adducts (same RT, different m/z), looks at their pairwise correlations, and examines detection rates across m/z, RT and sample classes.
        - Assesses contamination levels (blanks vs. real samples), checks consistency of exogenous standards, and explores intensity distributions by class and run order to flag drift or batch effects.
    - *TASK1_data_processing.ipynb*: Data Processing and Cleaning
        - Applies filtering rules to keep only features with low within-QC variability (< 30% CV), high detection support (≥ 70% of samples), and mass > 500 m/z.
        - Calculates D-Ratio per feature to identify those with excessive technical noise.
        - Tests normalization methods (median normalization, then global scaling) to normalize sample intensities and visualizes pre- and post-normalization distributions.
        - Performs a “final check” on kept features & samples, resulting in the cleaned data matrix to continue biomarker discovery.
    - *TASK1_discriminatory_analysis.ipynb*: 
        - Statistical Significance: Runs ANOVA across the three mapped classes (cancer, benign, healthy), applies FDR correction and post-hoc comparisons to features.
        - Predictive Modeling: Trains Random Forest and L1-regularized logistic regression to rank features by importance/coefficients, then finds common top predictors.
        - Decision Boundaries and Visualisation: Explores LDA and PCA projections to illustrate class separation, and evaluates model performance.
        - Concludes with a panel of glycan features that contains discovered biomarkers.
2. **TASK 2**: **Biomarker Embedding**
    - *TASK2_biomarker_embedding.ipynb*: Glycan Embedding and Enrichment
        - Plan: Defines a two-part strategy: first learn an embedding of the full glycan library based on sequence, composition, tissue and species origin, then project the discovered biomarkers into that space.
        - Part 1: Preprocesses `df_glycan.pkl` and `N_glycans_df.pkl`, engineers features from glycan sequences/compositions and trains an embedding model, e.g. via dimensionality reduction.
        - Part 2: Applies the learned embedding to the discovered glycans in `glycan_list.csv`, then for each biomarker infers disease associations and konwn protein-glycan binding partners from `glycan_binding.pkl` using propagation of labels after clustering.
        - Validates that known N-glycans cluster appropriately in embedding space.


## ▶️ Usage

There’s no need to rerun the notebooks, all cells have already been executed. Please open them in the following order:

```bash
jupyter lab notebooks/TASK1_EDA.ipynb
```

```bash
jupyter lab notebooks/TASK1_data_processing.ipynb
```

```bash
jupyter lab notebooks/TASK1_discriminatory_analysis.ipynb
```

```bash
jupyter lab notebooks/TASK2_biomarker_embedding.ipynb
```

Steps are clearly documented with code cells and inline commentary. 
 
Note: If you do rerun the notebooks, `TASK2_biomarker_embedding.ipynb` takes a long time to run. 

---

## ⚙️ Helper Functions

All reusable code for Task 1 and 2 lives in **`src/helpers_TASK1.py`** and **`src/helpers_TASK2.py`**:

- Embedding methods  
- Evaluation metrics  
- Data-wrangling utilities  

---

> **Note**: There’s still room for improving the embedding—especially local coherence—but this pipeline provides a reproducible framework for integrating multi-modal glycan data into a unified biomarker space.