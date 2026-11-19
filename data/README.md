# Data

The notebooks expect `data/fraud_data.pkl` (≈174 MB, not committed because GitHub limits files to 100 MB).

It is a preprocessed version of the Kaggle **Credit Card Transactions Fraud Detection Dataset**
(`fraudTrain.csv` + `fraudTest.csv`, 1,852,394 simulated transactions, 2019-01-01 → 2020-12-31, generated with Sparkov):
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Expected schema — a `pandas.DataFrame` indexed by `trans_date_trans_time` (datetime) with 23 columns:

| Column | Description |
|---|---|
| `amt` | transaction amount |
| `city_pop` | cardholder city population |
| `is_fraud` | target (1 = fraud) |
| `count` | transaction count feature |
| `amt_z_score` | amount z-score |
| `cat_z_score` | amount z-score within category (219 missing values) |
| `distance`, `distance_z_score` | cardholder–merchant distance and its z-score |
| `age` | cardholder age |
| `category_*` | one-hot merchant category (13 columns, baseline category dropped) |
| `gender_M` | one-hot gender |

Set `FRAUD_DATA_PATH` to use a different location. On Google Colab the notebooks fall back to
`/content/drive/MyDrive/datasets/fraud_data.pkl`.
