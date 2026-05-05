# ASHRAE – Great Energy Predictor III

Hourly building energy consumption forecasting across electricity, chilled water, steam, and hot water meters for the [ASHRAE GEPIII Kaggle competition](https://www.kaggle.com/competitions/ashrae-energy-prediction). Evaluated with RMSLE.

## Approach

- Per-meter LightGBM regressors trained on `log1p(meter_reading)`.
- Time, weather, and building features; categorical columns passed natively to LightGBM.
- Known bad rows filtered (site 0 electricity before 2016-05-21 is in kBTU and unreliable).
- Time-based holdout for validation (last month of training data).

## Layout

```
src/features.py        feature engineering
notebooks/01_eda.ipynb data exploration
notebooks/02_modeling.ipynb training + submission
```

## Setup

```bash
pip install -r requirements.txt
```

Place the Kaggle competition CSVs under `data/` (gitignored):

```
data/train.csv  data/test.csv  data/building_metadata.csv
data/weather_train.csv  data/weather_test.csv
```

Run notebooks in order. `02_modeling.ipynb` writes `submission.csv` for upload.
