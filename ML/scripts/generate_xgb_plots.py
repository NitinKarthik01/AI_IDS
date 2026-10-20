import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay
)

import xgboost as xgb

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = "/home/dhanush2026/dhanush2026/AI_IDS/ML"

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "RESULTS",
    "XGBOOST_MULTICLASS_SMOTE",
    "threshold_0.9"
)

REPORT_PATH = os.path.join(
    RESULT_PATH,
    "REPORT"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "MODELS",
    "XGBOOST_MULTICLASS_SMOTE",
    "threshold_0.9"
)

os.makedirs(REPORT_PATH, exist_ok=True)

# ==========================================================
# LOAD RESULTS
# ==========================================================

results = pd.read_csv(
    os.path.join(
        RESULT_PATH,
        "10Fold_Results.csv"
    )
)

# ==========================================================
# FOLD ACCURACY
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    results["Fold"],
    results["Accuracy"],
    marker="o",
    linewidth=2
)

plt.title("Fold Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Fold_Accuracy.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# METRIC COMPARISON
# ==========================================================

metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Macro ROC AUC"
]

means = results[metrics].mean()

plt.figure(figsize=(10,5))

means.plot(kind="bar")

plt.ylabel("Score")
plt.ylim(0,1.05)
plt.title("Average Metric Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Metric_Comparison.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# METRIC BOXPLOT
# ==========================================================

plt.figure(figsize=(10,5))

results[metrics].boxplot()

plt.xticks(rotation=20)

plt.ylabel("Score")

plt.title("Metric Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Metric_Boxplot.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# AVERAGE CONFUSION MATRIX
# ==========================================================

cm_files = sorted(
    glob.glob(
        os.path.join(
            RESULT_PATH,
            "confusion_matrix_fold_*.csv"
        )
    )
)

first_cm = pd.read_csv(
    cm_files[0],
    index_col=0
)

num_classes = first_cm.shape[0]

cm_sum = np.zeros(
    (num_classes, num_classes)
)

for file in cm_files:

    cm = pd.read_csv(
        file,
        index_col=0
    ).to_numpy(dtype=float)

    cm_sum += cm

cm_avg = cm_sum / len(cm_files)

labels = first_cm.index.tolist()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_avg,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(10,10))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format=".1f",
    xticks_rotation=90,
    colorbar=False
)

plt.title("Average Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Average_Confusion_Matrix.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# FEATURE IMPORTANCE (GAIN)
# ==========================================================

model = xgb.Booster()

model.load_model(
    os.path.join(
        MODEL_PATH,
        "best_xgboost_model.json"
    )
)

plt.figure(figsize=(12,8))

xgb.plot_importance(
    model,
    importance_type="gain",
    max_num_features=20,
    height=0.6
)

plt.title("Top 20 Feature Importance (Gain)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Feature_Importance_Gain.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# FEATURE IMPORTANCE (WEIGHT)
# ==========================================================

plt.figure(figsize=(12,8))

xgb.plot_importance(
    model,
    importance_type="weight",
    max_num_features=20,
    height=0.6
)

plt.title("Top 20 Feature Importance (Weight)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Feature_Importance_Weight.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# FEATURE IMPORTANCE (COVER)
# ==========================================================

plt.figure(figsize=(12,8))

xgb.plot_importance(
    model,
    importance_type="cover",
    max_num_features=20,
    height=0.6
)

plt.title("Top 20 Feature Importance (Cover)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Feature_Importance_Cover.png"
    ),
    dpi=300
)

plt.close()

print("="*60)
print("XGBoost Multiclass plots generated successfully.")
print("="*60)