import numpy as np

class TargetEncoder:
    def __init__(self, region_stats, global_mean, global_median, global_std, global_count, k):
        self.region_stats = region_stats
        self.global_mean = global_mean
        self.global_median = global_median
        self.global_std = global_std
        self.global_count = global_count
        self.k = k

    def transform(self, X_set, numerical):
        X_encoded = X_set[numerical].copy()
        X_temp = X_set[['region']].merge(self.region_stats, on='region', how='left')

        X_temp['mean'] = X_temp['mean'].fillna(self.global_mean)
        X_temp['median'] = X_temp['median'].fillna(self.global_median)
        X_temp['std'] = X_temp['std'].fillna(self.global_std)
        X_temp['count'] = X_temp['count'].fillna(self.global_count)

        n = X_temp['count']
        X_encoded['region_mean_smoothed'] = (n * X_temp['mean'] + self.k * self.global_mean) / (n + self.k)
        X_encoded['region_median'] = X_temp['median']
        X_encoded['region_count_log'] = np.log1p(X_temp['count'])
        X_encoded['region_std'] = X_temp['std']

        return X_encoded