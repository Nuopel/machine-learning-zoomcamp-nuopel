import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Ensure output dir
os.makedirs("./results_log_hybrid/plots", exist_ok=True)

print("\n📈 Plot: Model comparison (Predicted vs Actual) — VALIDATION")

# Which models output log1p(price)?
# In your hybrid script, ALL models are trained on y_train_log -> so True for all.
MODEL_OUTPUT_IS_LOG = {
    "Linear": True,
    "Ridge": True,
    "RF": True,
    "XGB": True,
}

def predict_model_eur(model, X, is_log=True):
    yhat = model.predict(X)
    if is_log:
        yhat = np.expm1(yhat)
    yhat = np.clip(yhat, 0.0, None)  # no negative prices
    return yhat

fig = plt.figure(figsize=(16, 4))

names = list(models.keys())
for idx, name in enumerate(names):
    model = models[name]
    ax = plt.subplot(1, len(names), idx + 1)

    is_log = MODEL_OUTPUT_IS_LOG.get(name, True)
    y_pred = predict_model_eur(model, X_val_np, is_log=is_log)

    ax.scatter(y_val_eur, y_pred, alpha=0.5, s=18)

    # Perfect line
    min_val = min(np.min(y_val_eur), np.min(y_pred))
    max_val = max(np.max(y_val_eur), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # Metrics in €
    r2 = r2_score(y_val_eur, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_eur, y_pred))
    mae = mean_absolute_error(y_val_eur, y_pred)

    ax.set_xlabel("True Price (€)", fontsize=10)
    ax.set_ylabel("Predicted Price (€)", fontsize=10)
    ax.set_title(f"{name}\nR²={r2:.3f}\nRMSE={rmse:.1f}€ | MAE={mae:.1f}€",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = "./results_log_hybrid/plots/model_comparison_predictions_val.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"✅ Saved: {outpath}")
