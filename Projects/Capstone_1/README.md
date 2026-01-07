# Loudspeaker Price Prediction (Capstone)

End-to-end ML project to predict loudspeaker price from Thiele–Small (T/S) parameters and selected product features.

## Problem statement

Given loudspeaker driver specifications, estimate the market price. The model focuses on physical descriptors (T/S parameters) plus a few manufacturer attributes (brand, magnet type, power handling) that plausibly affect price.

## Physical context (short)

At low frequency, a direct-radiator driver can be modeled by a second-order band-pass system. A minimal, physically grounded parameter set is:

$$
R_e,\; Q_{es},\; Q_{ms},\; f_s,\; S_d,\; V_{as}
$$

From these, key mechanical/electrical quantities (e.g., $Q_{ts}$, $M_{ms}$, $C_{ms}$, $Bl$) can be derived. See `notebooks/Lot0_minimal_ts_explanation.md` for the short derivation and physical interpretation.

## Dataset

Source: historical supplier database (~3,000 entries, partially filled).  
Filtering: keep drivers with complete minimal T/S set, then add a small set of extra features (power, efficiency, magnet type, etc.).

Final dataset: `Datas/speaker_db_selected_refined.csv`

## Project structure

```
Capstone_1/
├── README.md
├── Datas/
│   └── speaker_db_selected_refined.csv
├── notebooks/
│   ├── Lot0_minimal_ts_explanation.md
│   ├── Lot1_Data_analysis_and_cleaning.ipynb
│   ├── Lot2_EDA_price.ipynb
│   ├── Lot3_Modelling_te.ipynb
│   ├── Lot3_Modelling_te.py
│   └── Lot3_Modelling_2.py
├── Lot1_database_scrap/
│   └── tout_le_hp/
│       ├── A6_clean_selected_db.py
│       └── A7_refine_selected_db.py
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

## Notebooks guide

- `notebooks/Lot0_minimal_ts_explanation.md` — physics background (minimal T/S set).
- `notebooks/Lot1_Data_analysis_and_cleaning.ipynb` — quick overview of features, missing values, and column roles.
- `notebooks/Lot2_EDA_price.ipynb` — EDA: price distribution, correlation, and feature relationships.
- `notebooks/Lot3_Modelling_te.ipynb` — model tuning + evaluation (Ridge, RF, XGBoost).
- `notebooks/Lot3_Modelling_te.py` — baseline training script.
- `notebooks/Lot3_Modelling_2.py` — full pipeline with grouped split + target encoding.

## Quick start (local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run notebooks:

```bash
jupyter notebook notebooks/
```

Train via script:

```bash
python notebooks/Lot3_Modelling_te.py
```

## Notes

- Price is right-skewed; models use `log1p(price)` for stability.
- Some columns are intentionally dropped due to high missingness or low signal (see `Lot2_EDA_price.ipynb`).

