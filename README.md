#  Search Queries Anomaly Detection

A compact notebook project to **detect anomalies** in search‑query datasets (SEO/paid search/site). It performs light data cleaning, quick exploratory analysis (EDA), and flags suspicious rows with **Isolation Forest**.

---

##  What the project does

* Loads a CSV with query metrics (`Queries.csv`).
* Converts `CTR` from strings like `"5%"` to numeric `0.05`.
* Runs EDA:

  * top terms in queries (frequencies),
  * top queries by `Clicks`/`Impressions`,
  * top/bottom by `CTR`,
  * correlations among `Clicks`, `Impressions`, `CTR`, `Position`.
* Trains **Isolation Forest** on `Clicks`, `Impressions`, `CTR`, `Position` and marks anomalies.
* Outputs an anomaly table: `Top queries`, `Clicks`, `Impressions`, `CTR`, `Position`.

> Libraries: `pandas`, `plotly.express`, `scikit-learn` (IsolationForest), `collections.Counter`, `re`.

---

##  Repo layout

```
.
├── Search_Queries_Anomaly_Detection.ipynb   # main notebook
└── Queries.csv                               # input data (format below)
```

### Expected `Queries.csv` format

| Column      | Type / example            | Description                               |
| ----------- | ------------------------- | ----------------------------------------- |
| Top queries | `"how to …"` (str)        | Query text                                |
| Clicks      | `123` (int)               | Number of clicks                          |
| Impressions | `4567` (int)              | Number of impressions                     |
| CTR         | `"2.3%"` (str) or `0.023` | Click‑through rate; `%` strings converted |
| Position    | `3.4` (float)             | Average position                          |

> If `CTR` is already numeric, the conversion logic skips the `%` handling safely.

---

##  Installation & run

### Option A — Jupyter/Colab

1. Install deps in a cell:

```python
%pip install -q pandas plotly scikit-learn
```

2. Place `Queries.csv` next to the notebook.
3. Open `Search_Queries_Anomaly_Detection.ipynb` and run cells top‑to‑bottom.

### Option B — Local (venv)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install pandas plotly scikit-learn jupyter
jupyter notebook
```

Open the notebook and execute.

---

##  Quick script (no notebook)

Minimal script that reproduces anomaly detection and prints the top anomalies:

```python
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1) Load
df = pd.read_csv("Queries.csv")

# 2) Clean CTR: '5%' -> 0.05 (if already numeric, convert safely)
if df['CTR'].dtype == object:
    df['CTR'] = (df['CTR'].astype(str)
                           .str.rstrip('%')
                           .str.replace(',', '.')
                           .astype(float) / 100)

# 3) Features
features = df[['Clicks', 'Impressions', 'CTR', 'Position']]

# 4) Model
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso.fit(features)

# 5) Prediction & output
df['anomaly'] = iso.predict(features)  # -1 = anomaly, 1 = normal
anom = df[df['anomaly'] == -1][['Top queries', 'Clicks', 'Impressions', 'CTR', 'Position']]
print(anom.head(20) if not anom.empty else "No anomalies found at current settings.")
```

> `contamination=0.01` defines expected anomaly share (1%). Increase/decrease for sensitivity.

---

##  How to read the results

* **Top terms**: unexpected spikes may indicate spam or topic bursts.
* **Top by clicks/impressions**: locomotive queries; drops/spikes are audit candidates.
* **Top/bottom CTR**: extremely low/high CTR often signals mis‑alignment (intent vs. snippet/content).
* **Correlations**: typical patterns (Clicks↔Impressions +, CTR↔Position −). Strong deviations merit deeper investigation.
* **Anomaly table**: Isolation Forest results — validate each row in context (landing page, region, device, date).

---

##  Tuning

* `contamination` — prior fraction of anomalies. Start at `0.01` and tune to business reality.
* IsolationForest usually works without feature scaling, but for heavy right‑skew consider log‑transforming `Clicks/Impressions`.
* Train separate models by segment (brand vs. non‑brand, device, country) for better precision.

---

##  Limitations & improvements

* **No time awareness**: the model does not distinguish seasonality/viral spikes from true anomalies.

  * Improvement: add **time‑series features** (moving averages, seasonal indices) and/or use `prophet`/`STL` for decomposition.
* **Single algorithm** (IsolationForest):

  * Consider alternatives: **One‑Class SVM**, **Local Outlier Factor**, **Elliptic Envelope**; or ensemble the votes.
* **No text features** in the model:

  * Add query length, TF‑IDF/embeddings, and categorical features (brand vs. non‑brand).

---

##  Troubleshooting

* `ModuleNotFoundError`: install the missing package(s) (see Installation).
* `UnicodeDecodeError` when reading CSV: try `encoding='utf-8-sig'` or `encoding='cp1251'` in `pd.read_csv`.
* `ValueError` while converting CTR: ensure values look like `"5%"` (no stray spaces/symbols); consider pre‑cleaning with `str.strip()`.

---

##  License

Copyright (c) 2025 Nikita Marshchonok. All rights reserved.


##  Acknowledgments

* `scikit-learn` — Isolation Forest
* `plotly` — interactive charts
* `pandas` — tabular data wrangling



## Author:

telegram: @nikitamarshchonok

LinkedIn: https://www.linkedin.com/in/nikita-marshchonok

email: n.marshchonok@gmail.com
