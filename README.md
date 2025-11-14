# 🧠 South Asian Name Classification Model

A modern, research-driven **Natural Language Processing (NLP)** model built to identify individuals of **South Asian origin** from their names alone.  

This project enables more inclusive, culturally-aware tools—like a [searchable database of South Asian mental-health providers](https://findsouthasianmhc.org/)—to be built on real, data-backed foundations.

---

<details>
<summary><b>🔍 Overview</b></summary>

### What the model does
This repository contains a **hybrid ensemble model** that classifies whether a person’s name indicates South Asian origin (India, Pakistan, Bangladesh, Sri Lanka, Nepal, or Bhutan).  
The system was trained on a large, verified dataset of U.S. individuals with known race and ancestry labels.

### Why it’s innovative
- **Research-grounded:** built on character-level linguistic patterns and TF-IDF + deep contextual embeddings (NameBERT-SA).  
- **Hybrid ensemble:** combines a calibrated logistic regression (TF-IDF/SGD) and a fine-tuned Transformer model for near-state-of-the-art accuracy.  
- **Ethically designed:** aims to help users find culturally-aligned care without collecting sensitive personal data beyond the name field.  
- **Explainable outputs:** each run returns model probabilities, ensemble fusion scores, and abstention logic for uncertain cases.

### Performance snapshot
| Model Component | PR-AUC | ROC-AUC | Best F1 | Precision @ 0.90 | Recall @ 0.90 |
|-----------------|:------:|:------:|:-------:|:----------------:|:--------------:|
| TF-IDF + SGD | 0.85 | 0.96 | 0.78 | 0.90 | 0.63 |
| NameBERT-SA | 0.73 | 0.94 | 0.63 | 0.88 | 0.58 |
| **Fusion Ensemble** | **0.89** | **0.97** | **0.77** | **0.90+** | **0.76** |

</details>

---

<details>
<summary><b>⚙️ How to Run the Model on Your Own Dataset</b></summary>

### 📁 Folder setup
Your project should look like this:
```
/SouthAsianNameClassifier
│
├── artifacts_tuned_sgd/
├── artifacts_namebert_sa/
├── artifacts_ensemble/
├── batch_score.py
├── src/south_asian_nlp_mhisa.ipynb
└── test_providers.csv
```

> The three `artifacts_*` folders contain the pre-trained model weights and calibrators.  
> Do **not** rename these directories.

---

### ▶️ Example: Running the model
You can run the classifier on any CSV file containing a column of names (the names must be a single column, if names are seperated into first_name and last_name, add a clolumn and use =concatenate(A1, " ", B1), delete column after run).  
Using the provided `test_providers.csv`:

```bash
python batch_score.py   --in test_providers.csv   --out test_providers_scored.csv   --name-col "Provider Name"   --summary-out summary.json
```

#### Output files
- **`test_providers_scored.csv`** — your original dataset plus:
  - `prob_ensemble`: calibrated final probability of South Asian origin  
  - `prob_sgd`, `prob_namebert`: component model probabilities  
  - `decision_abstain_band`: `"sa"`, `"non_sa"`, or `"abstain"` (gray-zone confidence)  
  - `hard_label`: final binary classification (`"sa"` / `"non_sa"`)
- **`summary.json`** — a short metrics report printed to console and saved to file, e.g.:

```
=== South Asian Name Classification — Run Summary ===
Input rows (total):    1500
Non-blank scored:      1487
Skipped (blank names): 13
Operating params:      threshold=0.5942, abstain_low=0.49, abstain_high=0.69
Decision breakdown (non-blank only):
  SA:         135  (9.08%)
  non-SA:    1260  (84.73%)
  abstain:     92  (6.19%)
hard_label == 'sa': 142 / 1487 (9.55%)
Output CSV: /.../test_providers_scored.csv
```

---

### 🧾 Interpreting results
- **`prob_ensemble`** is the most accurate probability estimate.  
- **`hard_label`** is the recommended categorical prediction.  
- **`abstain`** means the model wasn’t confident enough to decide definitively.  
- Blank names are skipped automatically and excluded from totals.

---

### 💡 Optional flags
| Flag | Description |
|------|--------------|
| `--threshold`, `--abstain-low`, `--abstain-high` | Override default decision parameters |
| `--include-params-in-output` | Include thresholds in every output row |
| `--summary-out <file>` | Save a JSON summary of the run |

</details>

---

<details>
<summary><b>🧩 Working with the Source Notebook (`src/south_asian_nlp_mhisa.ipynb`)</b></summary>

The notebook walks through the **entire model development pipeline** — from data ingestion to feature engineering, TF-IDF + SGD training, Transformer fine-tuning, and calibration.

### 🧱 Environment setup
Install dependencies (Python ≥ 3.10 recommended):

```bash
pip install -r requirements.txt
```

Required core libraries:
```
pandas numpy scikit-learn torch transformers tokenizers joblib
```

If training locally on macOS (Apple Silicon), use:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### 🧭 Running the notebook
1. Open `src/south_asian_nlp_mhisa.ipynb` in VSCode, JupyterLab, or Google Colab.  
2. Run cells **sequentially** — they’re numbered by step.  
   - Step 1: Data ingestion & preprocessing  
   - Step 2: Feature engineering (TF-IDF n-grams)  
   - Step 4–5: Baseline + calibration  
   - Step 7–8: Transformer training (NameBERT-SA)  
   - Step 9: Ensemble fusion & export  
3. When complete, the notebook will generate:
   - `artifacts_tuned_sgd/`
   - `artifacts_namebert_sa/`
   - `artifacts_ensemble/`

Those folders are all that `batch_score.py` needs to perform inference on any new dataset.

---

### 🧠 Updating or retraining the model
If you retrain using a newer dataset:
- Ensure column names and data schema match the original (names + labels).  
- After Step 9, **replace** your artifacts directories in the project root.  
- Re-run `batch_score.py` — it will automatically use the new model files.

</details>

---

<details>
<summary><b>📜 Citation and Acknowledgments</b></summary>

This work builds on contemporary approaches in **onomastic machine learning** and **cross-linguistic text classification**, incorporating:

- Character n-grams & TF-IDF representations for name morphology  
- Transformer-based contextual embeddings for phonetic/semantic cues  
- Model calibration for real-world reliability  
- Ethical consideration in demographic inference

Developed by **MHISA Research Initiative** and collaborators in applied machine learning for cultural accessibility.

</details>
