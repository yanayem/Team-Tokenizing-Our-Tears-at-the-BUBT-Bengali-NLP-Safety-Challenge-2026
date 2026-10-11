# Team Tokenizing_Our_Tears at BUBT Bengali NLP Safety Challenge 2026

[![Kaggle](https://img.shields.io/badge/Kaggle-Bengali%20NLP%20Safety%20Challenge%202026-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/competitions/bengali-nlp-safety-challenge-2026/)
[![Task](https://img.shields.io/badge/Task-Fine--Grained%20Bengali%20Toxicity%20Understanding-blue.svg)](https://www.kaggle.com/competitions/bengali-nlp-safety-challenge-2026/)
[![Model](https://img.shields.io/badge/Model-BanglaBERT%20%2B%20BiLSTM%20%2B%20TF--IDF-green.svg)](https://huggingface.co/csebuetnlp/banglabert)
[![Private%20LB](https://img.shields.io/badge/Private%20LB-0.589-orange.svg)]()
[![Public%20LB](https://img.shields.io/badge/Public%20LB-0.593-yellow.svg)]()
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-red.svg)](LICENSE)

This repository contains the official implementation, dataset, and experiments for **Team Tokenizing_Our_Tears** (Paper ID: `BNSC-026`) at the [**BUBT Bengali NLP Safety Challenge 2026**](https://www.kaggle.com/competitions/bengali-nlp-safety-challenge-2026/) competition on **Fine-Grained Bengali Toxicity Understanding**.

Our solution proposes a hybrid system combining contextual representations from a pretrained **BanglaBERT--BiLSTM** neural network with surface-level lexical features from a **Word and Character TF--IDF** logistic regression classifier.

---

## 📌 Executive Summary

Online toxicity in Bengali encompasses fine-grained categories such as threats, identity attacks, sexual content, profanity, and indirect hostility like sarcasm or implicit toxicity. Furthermore, social media Bengali comments frequently feature informal spellings, Banglish (Latin-script Bengali), and character-level variations.

To address these challenges, our approach uses:
1. **Transformer Branch**: Pretrained `csebuetnlp/banglabert` coupled with a 2-layer Bidirectional LSTM (BiLSTM) sequence module, trained with Focal Loss ($\alpha=0.25, \gamma=2.0$) and positive class weights to mitigate severe class imbalance.
2. **Lexical Branch**: High-dimensional word (unigrams/bigrams) and character (2–6 n-grams) TF--IDF features paired with Logistic Regression to capture informal slang, Banglish, and explicit profanity patterns.
3. **Hybrid Ensemble & Post-Processing**: A weighted probability blend (70% BanglaBERT-base + 30% TF--IDF) combined with label-specific classification thresholds and an overall toxicity gating mechanism.

---

## 🏆 Key Results & Performance

| Evaluation Level | Model / Configuration | Combined Score / Metric | Toxic Macro-F1 | Mean Attribute F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Private Leaderboard** | **Hybrid Ensemble (Submitted v12)** | **0.589** | -- | -- |
| **Public Leaderboard** | **Hybrid Ensemble (Submitted v12)** | **0.593** | -- | -- |
| **Out-of-Fold (OOF)** | **Hybrid Blend (70:30 + Gate)** | **0.56458** | -- | -- |
| Out-of-Fold (OOF) | Word/Character TF--IDF | 0.52252 | 0.69339 | 0.35166 |
| Out-of-Fold (OOF) | BanglaBERT-base | 0.48130 | 0.66016 | 0.30243 |
| Out-of-Fold (OOF) | BanglaBERT-large | 0.44923 | 0.61376 | 0.28470 |

*Note: Evaluation metric $S = \frac{1}{2}\left(F_{\mathrm{tox}} + \frac{1}{6}\sum_{j=1}^{6}F_j\right)$ averages macro-F1 for overall toxicity and mean positive-class F1 across the six attributes.*

---

## 📊 Task & Dataset Overview

The system predicts **7 binary target labels** for each input comment:

| Label Name | Description | Training Positive Rate (%) |
| :--- | :--- | :---: |
| `is_toxic` | Overall toxicity flag | 66.92% |
| `implicit_toxicity` | Indirect or implied toxicity | 5.89% |
| `sarcasm` | Sarcastic or ironic harmful comment | 6.47% |
| `contains_profanity` | Explicit profanity or vulgar language | 14.00% |
| `threatening` | Intimidation, violence, or threats | 4.10% |
| `identity_attack` | Attack based on religion, gender, ethnicity, etc. | 12.88% |
| `sexual_content` | Explicit or suggestive sexual commentary | 3.43% |

### Dataset Splits
- **Train split**: 3,292 labeled comments
- **Validation split**: 500 labeled comments
- **Combined Training Set**: 3,792 comments (used for final model retraining)
- **Test split**: 1,000 comments (for competition submission and evaluation)

---

## 🏗️ System Architecture & Methodology

```
+-----------------------------------------------------------------------------------+
|                                   Input Comment                                   |
+------------------------------------------+----------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
       [ Cleaned Text Stream ]                      [ Raw + Cleaned Text Stream ]
                    |                                             |
                    v                                             v
     BanglaBERT-base Transformer                       Word & Character TF-IDF
                    |                                  (70k Word + 120k Char)
                    v                                             |
         2-Layer BiLSTM (256)                                     v
                    |                                    Logistic Regression
                    v                                     (Balanced Weights)
           Masked Mean Pooling                                    |
                    |                                             |
                    v                                             v
       [ BanglaBERT Probability ]                    [ Lexical Probability ]
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                              Probability Blend (70% / 30%)
                                           |
                                           v
                          Label-Specific Thresholds & Toxicity Gate
                                           |
                                           v
                              Final 7 Binary Target Labels
```

### 1. Transformer Branch
- **Base Model**: `csebuetnlp/banglabert`
- **Head**: 2-layer Bidirectional LSTM (256 units per direction) with masked mean pooling and linear classification layer.
- **Loss Function**: Focal-style loss ($\alpha=0.25, \gamma=2.0$) with clipped positive weights ($[1, 20]$).
- **Optimization**: AdamW (lr=$2\times10^{-5}$, weight decay=0.01) with linear warm-up (10%) over 4 epochs.

### 2. Lexical Branch
- **Features**: Word unigrams/bigrams (max 70,000 features) + Character 2–6 n-grams (max 120,000 features).
- **Classifier**: Multi-output Logistic Regression ($C=1.5$, `liblinear` solver, balanced class weights).

### 3. Post-Processing & Ensemble
- **Blending**: 0.70 $\times$ BanglaBERT + 0.30 $\times$ TF--IDF.
- **Thresholds**: Per-label decision thresholds $[0.54, 0.47, 0.46, 0.57, 0.53, 0.49, 0.48]$.
- **Toxicity Gate**: 0.40 overall-toxicity probability threshold to suppress false attributes on non-toxic comments.

---

## 📁 Repository Structure

```
.
├── BUBT Bengali NLP Safety Challenge 2026 dataset/
│   ├── train.csv                            # Training dataset (3,292 rows)
│   ├── validation.csv                       # Validation dataset (500 rows)
│   ├── test.csv                             # Public test dataset (1,000 rows)
│   └── test_gold_PRIVATE.csv                # Gold standard private test labels
├── NLP Safety Challenge on 70% test data final version.ipynb   # Main pipeline & experiments notebook
├── NLP Safety Challenge on golden test data.ipynb               # Evaluation notebook on golden test set
└── README.md                                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ and PyTorch installed with CUDA support. You can install all dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Notebooks
1. **Model Training & Submission Generation**:
   Open and execute `NLP Safety Challenge on 70% test data final version.ipynb` to run 5-fold cross-validation, feature extraction, model fitting, and export `submission_v12_targeted_upgrade.csv`.
2. **Golden Test Set Evaluation**:
   Open and execute `NLP Safety Challenge on golden test data.ipynb` to evaluate predictions against private test set ground truth labels.

---

## 👥 Team Members & Acknowledgments

**Team Tokenizing_Our_Tears** (Team Phoenix)
- **Yacin Arafat**
  - ID: `20245103160` | Intake: 53, Section: 2 | Dept. of CSE, BUBT
  - Email: [yanayem01@gmail.com](mailto:yanayem01@gmail.com)
- **Zikra**
  - ID: `20244103132` | Intake: 54, Section: 5 | Dept. of CSE, BUBT
  - Email: [zikraak04@gmail.com](mailto:zikraak04@gmail.com)
- **Shudipto Ghosh**
  - ID: `20245103151` | Intake: 53, Section: 8 | Dept. of CSE, BUBT
  - Email: [shudiptoghosh2001@gmail.com](mailto:shudiptoghosh2001@gmail.com)

**Supervisor / Advisor**
- **Md. Mithun Hossain** ([mithun@bubt.edu.bd](mailto:mithun@bubt.edu.bd))
- Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT), Dhaka, Bangladesh.

*Organized as part of the BUBT Bengali NLP Safety Challenge 2026.*

---

## 🔒 License & Usage Terms

### 1. Code & Pipeline (Non-Commercial Open Source)
This repository and its codebase are open-source for **academic, research, and non-commercial educational purposes only** under the **CC BY-NC 4.0** license. Any commercial or business use of the code, pipeline, or model weights is **strictly prohibited**.

### 2. Dataset Ownership & Restrictions
- All rights, titles, and intellectual property of the **BUBT Bengali NLP Safety Challenge 2026** dataset belong exclusively to the **BUBT Authors & Competition Organizers**.
- **Usage Restrictions**: No individual, company, or third party is permitted to use, redistribute, republish, or modify the dataset for personal or commercial purposes without explicit prior written authorization from the BUBT competition authors.

---

## 📄 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{yacin2026tokenizing_our_tears,
  title     = {Team Tokenizing\_Our\_Tears at the BUBT Bengali NLP Safety Challenge 2026: A Hybrid BanglaBERT--BiLSTM and TF--IDF System for Multi-Label Bengali Toxicity Classification},
  author    = {Yacin Arafat and Zikra and Shudipto Ghosh and Md. Mithun Hossain},
  booktitle = {Proceedings of the BUBT Bengali NLP Safety Challenge 2026},
  year      = {2026},
  note      = {Paper ID: BNSC-026}
}
```
