# Machine Learning Midterm

## Breast Cancer Wisconsin and Wine Classification

**Author:** Mohammad Nour AL ELAWI  
**University:** İstanbul Topkapı Üniversitesi

This project evaluates supervised classification and unsupervised clustering on
the Breast Cancer Wisconsin and Wine Classification datasets.

The experimental pipeline contains exactly **144 model fits**:

- **132 classifier fits**
  - Decision Tree
  - K-Nearest Neighbours
  - Gaussian Naive Bayes
- **12 K-Means fits**
- **3 random seeds:** `42`, `52`, and `62`
- **2 representations:** unscaled and training-only StandardScaler-scaled

The pipeline uses a fixed stratified test set and selects classifier
configurations using validation results only.

## Project Structure

```text
.
├── breast_cancer.ipynb
├── wine.ipynb
├── midterm_pipeline.py
├── requirements.txt
├── report.docx
├── AI_DISCLOSURE.md
├── figures/
├── classifier_runs.csv
├── classifier_summary.csv
├── classifier_test.csv
├── kmeans_runs.csv
├── kmeans_summary.csv
└── kmeans_test.csv
```

## Requirements

- Python **3.10 or newer**
- pandas
- scikit-learn
- matplotlib
- NumPy
- seaborn
- Jupyter Notebook or JupyterLab

The tested environment used Python 3.14.3. Exact package versions are listed
in [requirements.txt](requirements.txt).

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.10 is not installed, use another Python version newer than 3.10:

```powershell
py -m venv .venv
```

## Running the Project

Start Jupyter:

```bash
python -m pip install jupyter
jupyter notebook
```

Run both notebooks from a fresh kernel:

1. Open and run `breast_cancer.ipynb`.
2. Open and run `wine.ipynb`.

The notebooks load the datasets, perform the required exploratory analysis,
create the fixed and repeated splits, run the classification and K-Means
experiments, evaluate the selected seed-42 models, and generate the result
files and figures.

The shared experiment implementation is in
[midterm_pipeline.py](midterm_pipeline.py).

## Dataset Sources

Both datasets are loaded directly from scikit-learn. No external data files
are required.

```python
from sklearn.datasets import load_breast_cancer, load_wine

breast_cancer = load_breast_cancer()
wine = load_wine()
```

- `load_breast_cancer()` loads the Breast Cancer Wisconsin diagnostic dataset.
- `load_wine()` loads the Wine recognition dataset.

## Results

The experiment produces six CSV files:

| File | Required rows | Description |
|---|---:|---|
| `classifier_runs.csv` | 132 | Individual classifier fits |
| `classifier_summary.csv` | 44 | Mean and standard-deviation summaries |
| `classifier_test.csv` | 2 | Final seed-42 classifier evaluations |
| `kmeans_runs.csv` | 12 | Individual K-Means fits |
| `kmeans_summary.csv` | 4 | K-Means representation summaries |
| `kmeans_test.csv` | 4 | Final seed-42 clustering evaluations |

Figures are saved in the [figures/](figures/) directory.

## Required Links

> **GitHub repository:** [https://github.com/Mnour3593/ML-Midterm](https://github.com/Mnour3593/ML-Midterm)

> **Unlisted YouTube walkthrough:** [https://youtu.be/mHdRzwNX7nw](https://youtu.be/mHdRzwNX7nw)

## Academic Integrity and AI Disclosure

AI/tool use is documented in [AI_DISCLOSURE.md](AI_DISCLOSURE.md).
