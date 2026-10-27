import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ROOT = "C:/Dhanush/D/MSIS/Mini Project/DL"
# PROJECT_ROOT = "/home/dhanush2026/dhanush2026/AI_IDS/DL"

MODEL_NAME = "CNN_MULTICLASS_SMOTE"

THRESHOLD = "0.90"

# ==========================================================
# PATHS
# ==========================================================

RESULTS_DIR = os.path.join(PROJECT_ROOT, "RESULTS", MODEL_NAME)

# Try common threshold folder spellings (0.9 / 0.90)
candidates = [
    f"threshold_{THRESHOLD}",
    f"threshold_{float(THRESHOLD):.2f}",
]

RESULT_PATH = None
for c in candidates:
    p = os.path.join(RESULTS_DIR, c)
    if os.path.isdir(p):
        RESULT_PATH = p
        break

if RESULT_PATH is None:
    print("Results dir:", RESULTS_DIR)
    print("Exists:", os.path.isdir(RESULTS_DIR))
    if os.path.isdir(RESULTS_DIR):
        print("Contents:", os.listdir(RESULTS_DIR))
    raise FileNotFoundError(
        f"No threshold folder found. Tried: {candidates} in {RESULTS_DIR}"
    )

REPORT_PATH = os.path.join(RESULT_PATH, "REPORT")
os.makedirs(REPORT_PATH, exist_ok=True)

print("Looking in:", RESULT_PATH)

# ==========================================================
# LOAD + AVERAGE CONFUSION MATRICES
# ==========================================================

cm_files = sorted(
    glob.glob(os.path.join(RESULT_PATH, "confusion_matrix_fold_*.csv"))
)

if not cm_files:
    print("Contents:", os.listdir(RESULT_PATH))
    raise FileNotFoundError(
        f"No 'confusion_matrix_fold_*.csv' files found in: {RESULT_PATH}"
    )

print(f"Found {len(cm_files)} confusion matrix files.")

first_cm = pd.read_csv(cm_files[0], index_col=0)
cm_labels = first_cm.index.tolist()

cm_sum = np.zeros(first_cm.shape, dtype=float)

for file in cm_files:
    cm = pd.read_csv(file, index_col=0)
    cm = cm.reindex(index=cm_labels, columns=first_cm.columns, fill_value=0)
    cm_sum += cm.to_numpy(dtype=float)

cm_avg = cm_sum / len(cm_files)

# ==========================================================
# LOAD LABELS
# ==========================================================

label_file = os.path.join(RESULT_PATH, "label_mapping.csv")

if os.path.exists(label_file):
    label_map = pd.read_csv(label_file)
    class_names = label_map["Original_Label"].tolist()
    if len(class_names) != cm_avg.shape[0]:
        print("Warning: label_mapping size != matrix size, using CSV index labels.")
        class_names = [str(c) for c in cm_labels]
else:
    print("label_mapping.csv not found, using CSV index labels.")
    class_names = [str(c) for c in cm_labels]


def shorten(name, max_len=18):
    name = str(name)
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


short_names = [shorten(c) for c in class_names]

# ==========================================================
# ROW-NORMALISE (diagonal = per-class recall)
# ==========================================================

row_sums = cm_avg.sum(axis=1, keepdims=True)
cm_norm = np.divide(
    cm_avg, row_sums,
    out=np.zeros_like(cm_avg),
    where=row_sums != 0
)

# ==========================================================
# PLOT
# ==========================================================

def plot_confusion_matrix(matrix, labels, title, save_path,
                          value_fmt, annot_min, cmap="Blues", vmax=None):
    n = len(labels)

    fig_size = max(10, min(26, n * 0.55))
    tick_font = max(6, 14 - n * 0.2)
    annot_font = max(5, 11 - n * 0.2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_font)
    ax.set_yticklabels(labels, fontsize=tick_font)
    ax.set_xlabel("Predicted Label", fontsize=tick_font + 3)
    ax.set_ylabel("True Label", fontsize=tick_font + 3)
    ax.set_title(title, fontsize=tick_font + 6, pad=15)

    # Light grid between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.4)
    ax.tick_params(which="minor", length=0)

    # Annotate only meaningful cells
    color_thresh = (vmax if vmax is not None else matrix.max()) / 2
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if v >= annot_min:
                ax.text(
                    j, i, value_fmt(v),
                    ha="center", va="center",
                    fontsize=annot_font,
                    color="white" if v > color_thresh else "black"
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


save_file = os.path.join(REPORT_PATH, "Average_Confusion_Matrix_Normalized.png")

plot_confusion_matrix(
    cm_norm, short_names,
    title=f"Average Confusion Matrix - {MODEL_NAME} (Row-Normalised, %)",
    save_path=save_file,
    value_fmt=lambda v: f"{v * 100:.0f}",
    annot_min=0.01,      # hide cells below 1%
    vmax=1.0
)

print("=" * 60)
print(f"{MODEL_NAME} confusion matrix generated successfully.")
print("Saved to:", save_file)
print("=" * 60)