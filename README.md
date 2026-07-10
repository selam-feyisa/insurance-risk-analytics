# Insurance Risk Analytics & Predictive Modeling

End-to-end analysis of 18 months (Feb 2014 – Aug 2015) of ACIS car-insurance
data to identify low-risk segments, statistically validate risk drivers, and
build predictive models for risk-based pricing.

## Project Structure

```
insurance-risk-analytics/
├── .github/workflows/ci.yml   # CI: lint + tests on every push
├── data/                      # tracked by DVC, not Git
├── notebooks/
│   └── 01_eda.ipynb           # Task 1: exploratory data analysis
├── src/
│   ├── data_loader.py         # loading, dtype handling, missing-value reporting
│   └── eda_utils.py           # reusable plotting / aggregation functions
├── tests/                     # pytest unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

## Reproducing the Data Pipeline (DVC)

This project uses [DVC](https://dvc.org) to version the raw and cleaned
datasets without storing large files directly in Git.

1. Install DVC (already in `requirements.txt`).
2. Pull the tracked data after cloning:
   ```bash
   dvc pull
   ```
   This retrieves the dataset from the configured local remote and places
   it at `data/cleaned_data.csv` (and any other tracked versions).
3. If you need to point to a different remote (e.g. on a new machine),
   reconfigure it:
   ```bash
   dvc remote add -d localstorage /path/to/local/storage
   ```
4. To add a new version of the data after modifying it:
   ```bash
   dvc add data/cleaned_data.csv
   git add data/cleaned_data.csv.dvc
   git commit -m "Update cleaned dataset version"
   dvc push
   ```

See `.dvc/config` for the currently configured remote.

## Branches

- `task-1` — EDA and CI setup
- `task-2` — DVC pipeline setup
- `task-3` — A/B hypothesis testing
- `task-4` — predictive modeling

Each task branch is merged into `main` via Pull Request once complete.
