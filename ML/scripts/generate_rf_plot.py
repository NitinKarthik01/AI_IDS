import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = r"C:\Dhanush\D\MSIS\Mini Project\ML"

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "RESULTS",
    "RandomForest",          
    "threshold_0.10"
)

REPORT_PATH = os.path.join(
    RESULT_PATH,
    "REPORT"
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
    "Precision",
    "Recall",
    "F1 Score",
    "Macro F1",
    "AUC",
    "Specificity"
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

plt.xticks(rotation=25)

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
# AVERAGE ROC
# ==========================================================

roc_files = sorted(
    glob.glob(
        os.path.join(
            RESULT_PATH,
            "roc_fold_*.csv"
        )
    )
)

mean_fpr = np.linspace(0,1,100)

tprs = []

aucs = []

plt.figure(figsize=(6,6))

for file in roc_files:

    roc = pd.read_csv(file)

    interp = np.interp(
        mean_fpr,
        roc["FPR"],
        roc["TPR"]
    )

    interp[0]=0

    tprs.append(interp)

    aucs.append(
        auc(
            roc["FPR"],
            roc["TPR"]
        )
    )

mean_tpr = np.mean(
    tprs,
    axis=0
)

mean_tpr[-1]=1

mean_auc = auc(
    mean_fpr,
    mean_tpr
)

plt.plot(
    mean_fpr,
    mean_tpr,
    label=f"AUC = {mean_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Average ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Average_ROC.png"
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

cm_sum = np.zeros((2,2),dtype=float)

for file in cm_files:

    cm = pd.read_csv(file)

    cm = cm.iloc[:,1:].to_numpy(dtype=float)

    cm_sum += cm

cm_avg = cm_sum / len(cm_files)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_avg,
    display_labels=["Benign","Attack"]
)

fig, ax = plt.subplots(figsize=(6,6))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format=".1f",
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

print("="*60)
print("Random Forest plots generated successfully.")
print("="*60)