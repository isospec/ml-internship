# ML Internship Projects

This repository contains code, notebooks and helper scripts for the machine-learning assignment. The focus is on two main projects:

- **Task 1**: **Biomarker Discover**
- **Task 2**: **Biomarker Embedding**

---

## 📁 Repository Structure

```
ml-internship/
├── data/  
│   ├── df_glycan.csv             # Glycan feature table  
│   ├── glycan_binding.csv        # Protein–glycan binding data  
│   └── …                         
├── notebooks/  
│   ├── TASK1_data_processing.ipynb  
│   ├── TASK1_discriminatory_analysis.ipynb 
│   └── TASK2_biomarker_embedding.ipynb  
├── src/  
│   └── helpers_TASK1.py          # Utility functions for Task 1
│   └── helpers_TASK2.py          # Utility functions for Task 2  
├── requirements.txt              # Python dependencies  
└── README.md                     # You are here  
```

## ▶️ Usage

Open and run the notebooks in this order:

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

---

## ⚙️ Helper Functions

All reusable code for Task 1 and 2 lives in **`src/helpers_TASK1.py`** and **`src/helpers_TASK2.py`**:

- Embedding methods  
- Evaluation metrics  
- Data-wrangling utilities  

---

> **Note**: There’s still room for improving the embedding—especially local coherence—but this pipeline provides a reproducible framework for integrating multi-modal glycan data into a unified biomarker space.