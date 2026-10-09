import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================================

PROJECT_ROOT = "/home/dhanush2026/dhanush2026/AI_IDS/DL"

MODEL_NAME = "CNN_MULTICLASS"
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

# Read first matrix to determine size
first_cm = pd.read_csv(cm_files[0], index_col=0)
num_classes = first_cm.shape[0]

cm_sum = np.zeros((num_classes, num_classes))

for file in cm_files:

    cm = pd.read_csv(
        file,
        index_col=0
    ).to_numpy(dtype=float)

    cm_sum += cm

cm_avg = cm_sum / len(cm_files)
# ==========================================================
# LOAD LABELS
# ==========================================================

label_map = pd.read_csv(
    os.path.join(
        RESULT_PATH,
        "label_mapping.csv"
    )
)

class_names = label_map["Original_Label"].tolist()

# ==========================================================
# PLOT CONFUSION MATRIX
# ==========================================================

fig, ax = plt.subplots(figsize=(18, 18))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_avg,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    values_format=".0f",
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

print("Evaluation Plots Generated Successfully.")
