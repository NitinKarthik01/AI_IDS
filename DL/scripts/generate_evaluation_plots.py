import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================================

PROJECT_ROOT = r"C:\Dhanush\D\MSIS\Mini Project\DL"

MODEL_NAME = "ANN"
# PROJECT_ROOT = r"C:\Dhanush\D\MSIS\Mini Project\ML"

# MODEL_NAME = "RandomForest"

THRESHOLD = "0.10"

# ==========================================================

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "RESULTS",
    MODEL_NAME,
    f"threshold_{THRESHOLD}"
)

REPORT_PATH = os.path.join(
    RESULT_PATH,
    "REPORT"
)

os.makedirs(REPORT_PATH, exist_ok=True)

# ==========================================================
# ROC
# ==========================================================

roc_files = sorted(
    glob.glob(
        os.path.join(
            RESULT_PATH,
            "roc_fold_*.csv"
        )
    )
)

mean_fpr = np.linspace(0,1,200)

tprs = []

for file in roc_files:

    roc = pd.read_csv(file)

    interp = np.interp(
        mean_fpr,
        roc["FPR"],
        roc["TPR"]
    )

    interp[0] = 0

    tprs.append(interp)

mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1

plt.figure(figsize=(6,6))

plt.plot(
    mean_fpr,
    mean_tpr,
    linewidth=2,
    label="Average ROC"
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
# CONFUSION MATRIX
# ==========================================================

cm_files = sorted(
    glob.glob(
        os.path.join(
            RESULT_PATH,
            "confusion_matrix_fold_*.csv"
        )
    )
)

cm_sum = np.zeros((2,2))

for file in cm_files:

    cm = pd.read_csv(file)
    # Remove the first (text) column
    cm = cm.iloc[:, 1:].to_numpy(dtype=float)

cm_sum += cm

cm_avg = cm_sum / len(cm_files)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_avg,
    display_labels=["Normal","Attack"]
)

disp.plot(values_format=".1f")

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

print("Evaluation Plots Generated Successfully.")

