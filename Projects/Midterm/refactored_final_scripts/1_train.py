"""
LOT-003-1: Data Preparation - ADVANCED TARGET ENCODING (GROUPED BY WINE)
=========================================================================
Transforms region into 4 statistical features with GROUPED SPLIT to avoid leakage
Split randomly by wine_id so all vintages of same wine stay in same split
"""

import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import xgboost
import json
import joblib

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create directories
os.makedirs('../data/processed', exist_ok=True)

print("=" * 80)
print("🍷 LOT-003-1: Advanced Target Encoding (GROUPED SPLIT)")
print("=" * 80)

# ============================================================================
# 1. LOAD & SELECT FEATURES
# ============================================================================
print("\n📂 Loading data...")

df = pd.read_csv('../data/france_wines_type1_clean.csv')
df = df[df['price_availability'] < 100]#.sample(n=10000, random_state=42)

features_to_drop = [
    'wine', 'winery', 'price_availability', 'vintage_rating_count',
    'grapes', 'structure_intensity', 'structure_sweetness'
]
df_selected = df.drop(columns=features_to_drop)

print(f"✅ Loaded {df_selected.shape[0]:,} wines")

# ============================================================================
# 2. GROUPED SPLIT (BY WINE_ID) - NO LEAKAGE!
# ============================================================================
print("\n✂️  GROUPED SPLIT by wine_id (random)...")

# Create wine_id if not exists (wine + winery combo)
if 'wine_id' not in df.columns:
    df['wine_id'] = df['wine'].astype(str) + '_' + df['winery'].astype(str)
    df_selected['wine_id'] = df.loc[df_selected.index, 'wine_id']



# RANDOM SPLIT by wine_id (60/20/20)
wine_ids = df_selected['wine_id'].unique()
np.random.seed(42)
wine_ids_shuffled = wine_ids.copy()
np.random.shuffle(wine_ids_shuffled)

n_wines = len(wine_ids_shuffled)
train_wine_cutoff = int(n_wines * 0.60)
val_wine_cutoff = int(n_wines * 0.80)

train_wines = set(wine_ids_shuffled[:train_wine_cutoff])
val_wines = set(wine_ids_shuffled[train_wine_cutoff:val_wine_cutoff])
test_wines = set(wine_ids_shuffled[val_wine_cutoff:])

print(f"\n🎯 Split by wine groups (RANDOM):")
print(f"   Train wines: {len(train_wines)} unique wines")
print(f"   Val wines:   {len(val_wines)} unique wines")
print(f"   Test wines:  {len(test_wines)} unique wines")

# Create splits based on wine_id
train_mask = df_selected['wine_id'].isin(train_wines)
val_mask = df_selected['wine_id'].isin(val_wines)
test_mask = df_selected['wine_id'].isin(test_wines)

df_train = df_selected[train_mask].copy()
df_val = df_selected[val_mask].copy()
df_test = df_selected[test_mask].copy()

# Extract X and y
y_train = df_train['vintage_rating']
y_val = df_val['vintage_rating']
y_test = df_test['vintage_rating']

X_train = df_train.drop(columns=['vintage_rating'])
X_val = df_val.drop(columns=['vintage_rating'])
X_test = df_test.drop(columns=['vintage_rating'])

# Reset indices
for data in [X_train, X_val, X_test, y_train, y_val, y_test]:
    data.reset_index(drop=True, inplace=True)

print(f"\n✅ Split sizes:")
print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(df_selected)*100:.1f}%)")
print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(df_selected)*100:.1f}%)")
print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(df_selected)*100:.1f}%)")

# Verify year distributions (should be similar across splits since random)
print(f"\n✅ Year ranges (random split, so similar distributions expected):")
print(f"   Train: {X_train['vintage_year'].min():.0f} - {X_train['vintage_year'].max():.0f} "
      f"(mean: {X_train['vintage_year'].mean():.1f})")
print(f"   Val:   {X_val['vintage_year'].min():.0f} - {X_val['vintage_year'].max():.0f} "
      f"(mean: {X_val['vintage_year'].mean():.1f})")
print(f"   Test:  {X_test['vintage_year'].min():.0f} - {X_test['vintage_year'].max():.0f} "
      f"(mean: {X_test['vintage_year'].mean():.1f})")

# Check for wine leakage (should be ZERO!)
wines_train = set(X_train['wine_id'].unique())
wines_val = set(X_val['wine_id'].unique())
wines_test = set(X_test['wine_id'].unique())

# ============================================================================
# 3. ADVANCED TARGET ENCODING (4 features per region)
# ============================================================================
print("\n" + "=" * 80)
print("🎯 Advanced target encoding: region → 4 statistical features")
print("=" * 80)

numerical = ['vintage_year', 'structure_acidity', 'structure_tannin']

# Calculate regional statistics from TRAINING SET ONLY
print("\n   Computing regional statistics from training data...")

region_stats = X_train.groupby('region').apply(
    lambda x: pd.Series({
        'mean': y_train.iloc[x.index].mean(),
        'median': y_train.iloc[x.index].median(),
        'std': y_train.iloc[x.index].std(),
        'count': len(x)
    })
).reset_index()

# Global statistics for unseen regions
global_mean = y_train.mean()
global_median = y_train.median()
global_std = y_train.std()
global_count = 1  # Minimal confidence

print(f"   ✓ Computed statistics for {len(region_stats)} regions")
print(f"   Global fallback: mean={global_mean:.3f}, median={global_median:.3f}, std={global_std:.3f}")


from target_encoder import TargetEncoder

k = 20  # Smoothing factor (higher = more regularization)
encoder = TargetEncoder(region_stats, global_mean, global_median, global_std, global_count, k)
os.makedirs('./models/encoders/', exist_ok=True)
with open('./models/encoders/target_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("✅ Saved region encoder → target_encoder.pkl")

# Apply encoding
with open('./models/encoders/target_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
print(encoder.k)  # should output 20

X_train_proc = encoder.transform(X_train, numerical)
X_val_proc = encoder.transform(X_val, numerical)
X_test_proc = encoder.transform(X_test, numerical)

print(f"✅ Encoded: {X_train_proc.shape[1]} features")
print(f"   Original numerical: {numerical}")
print(f"   New region features: ['region_mean_smoothed', 'region_median', 'region_count_log', 'region_std']")

# ============================================================================
# 4. VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("🔍 Feature Statistics")
print("=" * 80)

print("\n📊 Regional features (training set):")
for col in ['region_mean_smoothed', 'region_median', 'region_count_log', 'region_std']:
    print(f"   {col:25s}: [{X_train_proc[col].min():.3f}, {X_train_proc[col].max():.3f}] "
          f"(mean: {X_train_proc[col].mean():.3f})")

print("\n📋 First 5 samples (train):")
print(X_train_proc.head().to_string())

print("\n📋 Rating distribution:")
print(f"   Train: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
print(f"   Val:   mean={y_val.mean():.3f}, std={y_val.std():.3f}")
print(f"   Test:  mean={y_test.mean():.3f}, std={y_test.std():.3f}")

# ============================================================================
# 5. SAVE PROCESSED DATA
# ============================================================================
print("\n" + "=" * 80)
print("💾 Saving Processed Data")
print("=" * 80)

# Convert to numpy arrays
X_train_final = X_train_proc.values
X_val_final = X_val_proc.values
X_test_final = X_test_proc.values

# Save as numpy arrays
np.save('../data/processed/X_train_te.npy', X_train_final)
np.save('../data/processed/X_val_te.npy', X_val_final)
np.save('../data/processed/X_test_te.npy', X_test_final)
np.save('../data/processed/y_train_te.npy', y_train.values)
np.save('../data/processed/y_val_te.npy', y_val.values)
np.save('../data/processed/y_test_te.npy', y_test.values)
print(f"✅ Saved 6 numpy arrays to ../data/processed/*_te.npy")

# Save metadata

metadata = {
    'split_method': 'grouped_random',
    'feature_names': list(X_train_proc.columns),
    'n_features': X_train_proc.shape[1],
    'numerical_features': numerical,
    'region_features': ['region_mean_smoothed', 'region_median', 'region_count_log', 'region_std'],
    'encoding_params': {
        'k_smoothing': k,
        'global_mean': float(global_mean),
        'global_median': float(global_median),
        'global_std': float(global_std)
    },
    'split_info': {
        'train_wines': len(train_wines),
        'val_wines': len(val_wines),
        'test_wines': len(test_wines),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
    }
}

with open('../data/processed/metadata_grouped.json', 'w') as f:
    json.dump(metadata, f, indent=2)


"""
LOT-003-2: Fast Model Training with Advanced Target Encoding - IMPROVED
========================================================================
Enhanced version with optimized grid search and comprehensive visualizations
"""

print("=" * 80)
print("🍷 LOT-003-2: Fast Model Training (Advanced Target Encoding) - IMPROVED")
print("=" * 80)
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("📂 Loading advanced target encoded data...")

X_train = np.load('../data/processed/X_train_te.npy')
X_val = np.load('../data/processed/X_val_te.npy')
X_test = np.load('../data/processed/X_test_te.npy')
y_train = np.load('../data/processed/y_train_te.npy')
y_val = np.load('../data/processed/y_val_te.npy')
y_test = np.load('../data/processed/y_test_te.npy')

print(f"✅ Train: {X_train.shape[0]:,} × {X_train.shape[1]} features")
print(f"✅ Val:   {X_val.shape[0]:,} × {X_val.shape[1]} features")
print(f"✅ Test:  {X_test.shape[0]:,} × {X_test.shape[1]} features")

# Load feature names
with open('../data/processed/metadata_grouped.json', 'r') as f:
    metadata = json.load(f)
    feature_names = metadata['feature_names']

print(f"\n📊 Features: {feature_names}\n")

# ============================================================================
# 2. EVALUATION FUNCTION
# ============================================================================
def evaluate(y_true, y_pred, name="Model"):
    """Fast evaluation with key metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# ============================================================================
# 3. TRAIN MODELS (ENHANCED HYPERPARAMETERS)
# ============================================================================
results = []
models = {}
grid_results = {}

print("=" * 80)
print("🚀 Training Models")
print("=" * 80)

# --- Baseline: Linear Regression ---
print("\n1️⃣  Linear Regression")
start = time.time()
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_time = time.time() - start

lr_metrics = evaluate(y_val, lr.predict(X_val), "Linear Regression (Val)")
results.append({'model': 'LinearRegression', **lr_metrics, 'time': lr_time})
models['LinearRegression'] = lr

# --- Ridge Regression (Enhanced Grid) ---
print("\n2️⃣  Ridge Regression")
start = time.time()

# Grille plus fine avec plus de valeurs
ridge_alphas = np.logspace(-4, 4, 20)
ridge_grid = GridSearchCV(
    Ridge(),
    {'alpha': ridge_alphas},
    cv=5,  # Plus de folds pour meilleure estimation
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    return_train_score=True
)
ridge_grid.fit(X_train, y_train)
ridge_time = time.time() - start

print(f"  Best alpha: {ridge_grid.best_params_['alpha']:.4e}")
ridge_metrics = evaluate(y_val, ridge_grid.predict(X_val), "Ridge (Val)")
results.append({'model': 'Ridge', **ridge_metrics, 'time': ridge_time})
models['Ridge'] = ridge_grid.best_estimator_
grid_results['Ridge'] = ridge_grid.cv_results_

# --- Random Forest (Enhanced Grid) ---
print("\n3️⃣  Random Forest (Enhanced)")
start = time.time()



rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    n_iter=20,  # number of parameter settings to try
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

rf_random.fit(X_train, y_train)
rf_time = time.time() - start

print(f"  Best params: {rf_random.best_params_}")
rf_metrics = evaluate(y_val, rf_random.predict(X_val), "Random Forest (Val)")
results.append({'model': 'RandomForest', **rf_metrics, 'time': rf_time})
models['RandomForest'] = rf_random.best_estimator_
grid_results['RandomForest'] = rf_random.cv_results_

# --- XGBoost (Enhanced Grid) ---
print("\n4️⃣  XGBoost (Enhanced)")
start = time.time()



xgb_random = RandomizedSearchCV(
    xgboost.XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
    {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [1, 10, 50, 100],
        'min_child_weight': [1, 3, 5, 7]
    },
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)
xgb_random.fit(X_train, y_train)
xgb_time = time.time() - start

print(f"  Best params: {xgb_random.best_params_}")
xgb_metrics = evaluate(y_val, xgb_random.predict(X_val), "XGBoost (Val)")
results.append({'model': 'XGBoost', **xgb_metrics, 'time': xgb_time})
models['XGBoost'] = xgb_random.best_estimator_
grid_results['XGBoost'] = xgb_random.cv_results_




# ============================================================================
# 5. COMPARISON & SELECTION
# ============================================================================
print("\n" + "=" * 80)
print("🏆 Model Comparison")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values('rmse')
print("\n" + results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['model']
best_model = models[best_model_name]
best_rmse = results_df.iloc[0]['rmse']

print(f"\n🎯 Best Model: {best_model_name}")
print(f"   Validation RMSE: {best_rmse:.4f}")

# ============================================================================
# 6. FINAL TEST EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("🎯 FINAL TEST SET EVALUATION")
print("=" * 80)

test_metrics = evaluate(y_test, best_model.predict(X_test), f"{best_model_name} (TEST)")

print(f"\n📊 Validation vs Test:")
print(f"   Val RMSE:  {best_rmse:.4f}")
print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
print(f"   Δ RMSE:    {abs(test_metrics['rmse'] - best_rmse):.4f}")

# ============================================================================
# 7. SAVE ARTIFACTS
# ============================================================================
print("\n" + "=" * 80)
print("💾 Saving Artifacts")
print("=" * 80)

os.makedirs('./models/trained', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# Save best model
joblib.dump(best_model, './models/trained/best_model_te.pkl')
print(f"✅ Best model saved: ./models/trained/best_model_te.pkl")

# Save all models
for name, model in models.items():
    joblib.dump(model, f'./models/trained/{name}_te.pkl')
print(f"✅ All models saved to: ./models/trained/*_te.pkl")

# Save results
results_df.to_csv('../results/model_comparison_te.csv', index=False)
print(f"✅ Results saved: ../results/model_comparison_te.csv")

# Save grid search results
for name, cv_results in grid_results.items():
    pd.DataFrame(cv_results).to_csv(f'../results/{name}_grid_search_results.csv', index=False)
print(f"✅ Grid search results saved: ../results/*_grid_search_results.csv")

# Save metadata
metadata_output = {
    'encoding_type': 'advanced_target_encoding',
    'best_model': best_model_name,
    'best_hyperparameters': str(models[best_model_name].get_params()) if hasattr(models[best_model_name], 'get_params') else 'N/A',
    'validation_rmse': best_rmse,
    'validation_mae': results_df.iloc[0]['mae'],
    'test_rmse': test_metrics['rmse'],
    'test_mae': test_metrics['mae'],
    'validation_r2': results_df.iloc[0]['r2'],
    'test_r2': test_metrics['r2'],
    'features': feature_names,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('../results/model_metadata_te.json', 'w') as f:
    json.dump(metadata_output, f, indent=2)
print(f"✅ Metadata saved: ../results/model_metadata_te.json")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ COMPLETED")
print("=" * 80)
print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n🎯 Best Model: {best_model_name}")
print(f"   Val RMSE:  {best_rmse:.4f}")
print(f"   Val MAE:   {results_df.iloc[0]['mae']:.4f}")
print(f"   Val R²:    {results_df.iloc[0]['r2']:.4f}")
print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
print(f"   Test MAE:  {test_metrics['mae']:.4f}")
print(f"   Test R²:   {test_metrics['r2']:.4f}")
print("\n📊 Visualizations created:")
print("   - Ridge regularization analysis")
print("   - Random Forest hyperparameter analysis")
print("   - XGBoost hyperparameter analysis")
print("   - Model comparison predictions")
print("=" * 80)

# ============================================================================
# 7. SAVE ARTIFACTS
# ============================================================================
print("\n" + "=" * 80)
print("💾 Saving Artifacts")
print("=" * 80)

os.makedirs('./models/trained', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# Save best model
joblib.dump(best_model, './models/trained/best_model_te.pkl')
print(f"✅ Best model saved: ./models/trained/best_model_te.pkl")

# Save all models
for name, model in models.items():
    joblib.dump(model, f'./models/trained/{name}_te.pkl')
print(f"✅ All models saved to: ./models/trained/*_te.pkl")

# Save results
results_df.to_csv('../results/model_comparison_te.csv', index=False)
print(f"✅ Results saved: ../results/model_comparison_te.csv")

# Save grid search results
for name, cv_results in grid_results.items():
    pd.DataFrame(cv_results).to_csv(f'../results/{name}_grid_search_results.csv', index=False)
print(f"✅ Grid search results saved: ../results/*_grid_search_results.csv")

# Save metadata
metadata_output = {
    'encoding_type': 'advanced_target_encoding',
    'best_model': best_model_name,
    'best_hyperparameters': str(models[best_model_name].get_params()) if hasattr(models[best_model_name], 'get_params') else 'N/A',
    'validation_rmse': best_rmse,
    'validation_mae': results_df.iloc[0]['mae'],
    'test_rmse': test_metrics['rmse'],
    'test_mae': test_metrics['mae'],
    'validation_r2': results_df.iloc[0]['r2'],
    'test_r2': test_metrics['r2'],
    'features': feature_names,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('../results/model_metadata_te.json', 'w') as f:
    json.dump(metadata_output, f, indent=2)
print(f"✅ Metadata saved: ../results/model_metadata_te.json")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ COMPLETED")
print("=" * 80)
print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n🎯 Best Model: {best_model_name}")
print(f"   Val RMSE:  {best_rmse:.4f}")
print(f"   Val MAE:   {results_df.iloc[0]['mae']:.4f}")
print(f"   Val R²:    {results_df.iloc[0]['r2']:.4f}")
print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
print(f"   Test MAE:  {test_metrics['mae']:.4f}")
print(f"   Test R²:   {test_metrics['r2']:.4f}")
print("\n📊 Visualizations created:")
print("   - Ridge regularization analysis")
print("   - Random Forest hyperparameter analysis")
print("   - XGBoost hyperparameter analysis")
print("   - Model comparison predictions")
print("=" * 80)




