import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CHANGE ONLY THESE
# ==========================================================

PROJECT_ROOT = r"C:\Dhanush\D\MSIS\Mini Project\DL"

MODEL_NAME = "ANN"

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
# LOAD HISTORY FILES
# ==========================================================

history_files = sorted(
    glob.glob(
        os.path.join(
            RESULT_PATH,
            "history_fold_*.csv"
        )
    )
)

acc_list = []
val_acc_list = []

loss_list = []
val_loss_list = []

for file in history_files:

    df = pd.read_csv(file)

    acc_list.append(df["accuracy"].values)

    val_acc_list.append(df["val_accuracy"].values)

    loss_list.append(df["loss"].values)

    val_loss_list.append(df["val_loss"].values)

# ==========================================================
# MAKE ALL FOLDS SAME LENGTH
# ==========================================================

lengths = [len(x) for x in acc_list]

print("Epochs per fold:", lengths)

min_epochs = min(lengths)

print("Using first", min_epochs, "epochs from every fold.")

acc_list = [x[:min_epochs] for x in acc_list]
val_acc_list = [x[:min_epochs] for x in val_acc_list]

loss_list = [x[:min_epochs] for x in loss_list]
val_loss_list = [x[:min_epochs] for x in val_loss_list]

acc_mean = np.mean(np.array(acc_list), axis=0)
val_acc_mean = np.mean(np.array(val_acc_list), axis=0)

loss_mean = np.mean(np.array(loss_list), axis=0)
val_loss_mean = np.mean(np.array(val_loss_list), axis=0)

epochs = np.arange(1, len(acc_mean)+1)

# ==========================================================
# TRAINING ACCURACY
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    acc_mean,
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_acc_mean,
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Average Training Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Average_Training_Accuracy.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# TRAINING LOSS
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    loss_mean,
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss_mean,
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Average Training Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Average_Training_Loss.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# FOLD ACCURACY
# ==========================================================

results = pd.read_csv(
    os.path.join(
        RESULT_PATH,
        "10Fold_Results.csv"
    )
)

plt.figure(figsize=(8,5))

plt.plot(
    results["Fold"],
    results["Accuracy"],
    marker="o",
    linewidth=2
)

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("10-Fold Cross Validation Accuracy")
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
# BOXPLOT
# ==========================================================

metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Macro F1",
    "AUC",
    "Specificity"
]

plt.figure(figsize=(11,6))

results[metrics].boxplot()

plt.xticks(rotation=30)

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
# METRIC COMPARISON
# ==========================================================

summary = pd.read_csv(
    os.path.join(
        RESULT_PATH,
        "experiment_summary.csv"
    )
)

plt.figure(figsize=(10,5))

plt.bar(
    summary["Metric"],
    summary["Mean"]
)

plt.ylabel("Score")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Metric_Comparison.png"
    ),
    dpi=300
)

plt.close()

print("Training plots generated successfully.")