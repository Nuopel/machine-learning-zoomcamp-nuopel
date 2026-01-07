# smoothed_target_encoder.py
import numpy as np
import pandas as pd


class SmoothedTargetEncoder:
    def __init__(self, cat_cols, global_mean, k, mapping):
        self.cat_cols = cat_cols
        self.global_mean = float(global_mean)
        self.k = float(k)
        self.mapping = mapping  # dict[col] -> df with mean,count

    def transform(self, X):
        X = X.copy()
        for col in self.cat_cols:
            if col not in X.columns:
                continue

            s = X[col].astype(str).fillna("NaN")
            stats = self.mapping[col]

            mean = s.map(stats["mean"]).fillna(self.global_mean)
            count = s.map(stats["count"]).fillna(0)

            X[f"{col}__te"] = (count * mean + self.k * self.global_mean) / (count + self.k)
            X.drop(columns=[col], inplace=True)

        return X


def fit_target_encoder(X, y_eur, cat_cols, k):
    mapping = {}
    y_eur = np.asarray(y_eur, dtype=float)

    for col in cat_cols:
        if col not in X.columns:
            continue
        tmp = pd.DataFrame({"cat": X[col].astype(str).fillna("NaN"), "y": y_eur})
        mapping[col] = tmp.groupby("cat")["y"].agg(["mean", "count"])

    return SmoothedTargetEncoder(cat_cols, global_mean=np.mean(y_eur), k=k, mapping=mapping)
