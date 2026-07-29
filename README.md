# Distribution Shift Feature Selection - Covariate Shift and Concept Drift

This repository implements a comprehensive benchmark for feature selection under distribution shift, covering both covariate shift(P(X) shifts) and concept drift(P(Y|X) shifts) scenarios -
- ***For concept drift, the meta learner*** is leveraged to capture the discrepancy of the conditional mean E[Y|X, T=1](new batch of data) - E[Y|X, T=0], achieved SOTA performance on multiple modalities of datasets.
- ***For covaraite shift, the RF OOB variable importance along with the LOCO-MMD*** is leveraged with the highly competitive performance in comparison with the current SOTA(fsl-net and datafix), yielding higher computationally effciency.

## Benchmark Methods

The benchmark includes the following methods, categorized by their underlying methodology:

### Concept Drift Methods

- **ARD**: Automatic Relevance Divergence (Primary for Concept Drift)
- **Delta Shapley Value** (Primary for Concept Drift)
- **L1-Penalized Regression - specified** (For Concept Drift Only) — from [arXiv:2505.20634](https://arxiv.org/pdf/2505.20634)
- **L1-Penalized Regression - misspecified** (For Concept Drift Only) — from [arXiv:2505.20634](https://arxiv.org/pdf/2505.20634)
- **Knockoff based procedure** (Primarily for Concept Drift) — from [arXiv:2505.20634](https://arxiv.org/pdf/2505.20634)
- **GRF** (Generalized Random Forest) — from Causal Forest Variable Importance Procedure (Primarily for Concept Drift)
- **PermuCATE** (Permutation Variable Importance in the PO-pseudo outcome learner) — from Meta-Learner Variable Importance Procedure
- **LOCO (R-risk)** (Primarily for Concept Drift, indeed for concept drift only) — from Meta-Learner Variable Importance Procedure

### Covariate Shift Methods

- **MMD-LOCO** (Maximum Mean Discrepancy with LOCO — Leave-One-Covariate-Out, Covariate Shift) — from statistical-based measure procedure
- **HSIC-LOCO** (Hilbert-Schmidt Independence Criterion, Covariate Shift) — from statistical-based measure procedure
- **Boruta** (Covariate Shift Only) — built on top of binary classifier to distinguish two batches of data (P(X_exist) and P(X_new))
- **ShadowVIMP** (Covariate Shift Only) — built on top of binary classifier to distinguish two batches of data (P(X_exist) and P(X_new))

### Joint Distribution (Covariate + Concept) Methods

- **FSL-Net** (Originally for Covariate Shift, can be adapted to Joint Distribution) — from [AI-sandbox/FSL-Net](https://github.com/AI-sandbox/FSL-Net), current SOTA
- **Data-Fix** (Covariate Shift, can be adapted to Joint Distribution) — from [AI-sandbox/DataFix](https://github.com/AI-sandbox/DataFix), current SOTA
- **DRPerm-LOCO** (Doubly Robust Pseudo-Outcome Learner + Leave-One-Covariate-Out, both for covariate shift and concept drift)
- **Vita PIMP** (Permutation Variable Importance) — built on top of binary classifier to distinguish two batches of data (P(X_exist) and P(X_new))

### Proposed Method

- **RF-domain: OOB-VIMP** for Random Forest classifier (XGBoost, MLP, and Kernel Ridge Regression are also leveraged)

### Additional Domain Classifier Methods
- **Boruta** (Covariate Shift Only)
- **ShadowVIMP** (Covariate Shift Only)
- **Vita PIMP** (Permutation Variable Importance)

## Repository Structure

### Core Scripts

- **`benchmark_methods.py`**: Main function for comparing all benchmark methods across different scenarios
- **`synthetic_DGP.py`**: Data-generating processes for synthetic experiments
- **`real_data_utils.py`**: Utility functions for real-data applications
- **`fsds_nonuniqueness.py`**: Non-unique decomposition for covariate shift (MMD, HSIC) and concept drift (R-risk), with the visualization illustration displayed in https://colab.research.google.com/drive/1w5fKyTqnWoKEfixGnpaZCMSzQg0IOr2-
- **`domain_classifier_VIMP_whole.py`**: RF VIMP adapter along with the other domain classifier methods
- **`fetch_tabular_openml_datasets.py`**: Data preprocessing for 8 OpenML sklearn datasets; results stored in `real_data/tabular/`

### Non-unique Decomposition of distribution(Covariate Shift, Concept Drift and the difference of Model Performance)

- **Nonunique decomposition of Covariate Shift & Concept Drift & Causal Testing for Distribution Shift** — Colab notebook: [Meta Learner as Feature Selection](https://colab.research.google.com/drive/1w5fKyTqnWoKEfixGnpaZCMSzQg0IOr2)
- **Illustration for the impossibility for decomposing covariate shift & concept drift into individual features** 
- ***HSIC - Hilbert Schmidt Independence Criterion(Covariate Shift Only)***
<img width="2873" height="2225" alt="linearCovariateShift_HSIC_Gamma02" src="https://github.com/user-attachments/assets/5be0201c-7157-453c-95f8-971d54d2bf3d" />
- ***MMD - Maximal Mean Discrepancy(Covariate Shift Only)***
<img width="2873" height="2225" alt="linearCovariateShift_MMD_Gamma02" src="https://github.com/user-attachments/assets/eaeed113-d43c-4d95-b353-3a30883690a1" />
- ***R-Risk(Concept Drift Only)***
<img width="2873" height="2225" alt="linearConceptDrift_RRisk_deltabeta03" src="https://github.com/user-attachments/assets/9ec4f09f-d7b6-4a44-8fe8-fbff4fc2bef0" />

### Results

- The `results/` folder contains all data file results. Files uploaded to Overleaf are formatted without underscores (`_`) in the naming convention.
- **Selected Results**
<img width="1190" height="1448" alt="Screenshot 2026-07-24 at 11 08 22" src="https://github.com/user-attachments/assets/2a5b82d4-e98c-4a40-a32a-48663e23fe2a" />
<img width="1468" height="1200" alt="Screenshot 2026-07-24 at 11 07 37" src="https://github.com/user-attachments/assets/8286cf69-48d2-4556-9c63-7b0e92fb50f1" />
<img width="1182" height="920" alt="Screenshot 2026-07-24 at 11 03 10" src="https://github.com/user-attachments/assets/b7939189-ede2-4b4b-bac3-254e3a58c909" />

### Real-Data Applications


The following real-world datasets are included with corresponding benchmark scripts:

- **FMoW** (Functional Map of the World, https://github.com/fMoW/dataset)
- **WhyShift** (https://github.com/namkoong-lab/whyshift)
- **Tabular** (8 OpenML datasets, fetch_openml)
- **ChronoBerg** (gradual sentiment shift data, https://huggingface.co/datasets/spaul25/Chronoberg)

### Sensitivity Analyses

- **Weak overlap sensitivity analysis**: `weakly_overlapped_benchmark.py`
<img width="1311" height="1234" alt="feature_selection_overlap_sensitivity_analysis" src="https://github.com/user-attachments/assets/8e816930-d64e-4d0b-a463-39f69f9ea9be" />
- **Outcome Model Sensitivity Analysis**: `outcome_model_sensitivity.py`
- **Adversarial perturbations**: `adversarial_conceptdrift.py` — includes 5 types of adversarial perturbations for concept drift robustness testing.
