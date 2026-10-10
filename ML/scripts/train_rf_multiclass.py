
import os
import sys
import random
import warnings
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_DIR)

from utils.plots import (
    save_fold_accuracy,
    save_metrics_bar,
    save_feature_importance
)

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    balanced_accuracy_score
)

warnings.filterwarnings("ignore")

# ==========================================================
# TRAINING PARAMETERS
# ==========================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

THRESHOLD = "0.10"

CLASSIFICATION = "multiclass"

N_SPLITS = 10

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = "/home/dhanush2026/dhanush2026/AI_IDS"

DATASET_FILE = (
    f"{PROJECT_ROOT}/PCC_RESULTS/"
    f"threshold_{THRESHOLD}/reduced_dataset.csv"
)

MODEL_DIR = (
    f"{PROJECT_ROOT}/ML/MODELS/"
    f"RandomForest_MULTICLASS/threshold_{THRESHOLD}"
)

RESULT_DIR = (
    f"{PROJECT_ROOT}/ML/RESULTS/"
    f"RandomForest_MULTICLASS/threshold_{THRESHOLD}"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Reduced Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_FILE)

print(df.shape)
print(df.head())
print(df.columns.tolist())

# ==========================================================
# CLEAN DATASET
# ==========================================================

if "Label" not in df.columns:
    raise ValueError("Label column not found in dataset!")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

assert np.isfinite(
    df.select_dtypes(include=[np.number]).values
).all(), "Dataset still contains NaN or Inf!"

# ==========================================================
# MULTICLASS LABEL ENCODING
# ==========================================================

if CLASSIFICATION == "multiclass":

    df["Label"] = df["Label"].astype(str).str.strip()

    label_encoder = LabelEncoder()

    df["Label"] = label_encoder.fit_transform(
        df["Label"]
    )

    class_names = label_encoder.classes_

    print("\nClass Mapping:")

    for i, name in enumerate(class_names):
        print(i, ":", name)

print("\nClass Distribution:")
print(df["Label"].value_counts())

# ==========================================================
# FEATURES
# ==========================================================

X = df.drop(columns=["Label"])
y = df["Label"]

feature_names = X.columns

print("\nFeature Shape:", X.shape)
print("Label Shape:", y.shape)

# ==========================================================
# NUMPY CONVERSION
# ==========================================================

X = X.values.astype(np.float32)
y = y.values.astype(int)

n_classes = len(np.unique(y))

print("\nNumber of Classes:", n_classes)

if n_classes < 3:
    raise ValueError(
        "Multiclass classification requires at least 3 classes."
    )

class_counts = np.bincount(y)

print("\nClass Distribution:")
for name, count in zip(class_names, class_counts):
    print(f"{name}: {count}")

if np.max(class_counts) < N_SPLITS:
    raise ValueError(
        "Insufficient samples for 10-fold cross-validation."
    )

if np.min(class_counts) < N_SPLITS:
    print(
        "\nWARNING: Some classes contain fewer than "
        "10 samples. They may be absent from certain "
        "test folds."
    )
# ==========================================================
# CROSS VALIDATION
# ==========================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

# ==========================================================
# STORE RESULTS
# ==========================================================

results = []

fold = 1

# ==========================================================
# 10-FOLD CROSS VALIDATION
# ==========================================================

for train_idx, test_idx in skf.split(X, y):

    print("\n" + "=" * 70)
    print(f"Fold {fold}/{N_SPLITS}")
    print("=" * 70)

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Training Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)

    # ======================================================
    # FEATURE SCALING
    # ======================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ======================================================
    # RANDOM FOREST MODEL
    # ======================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1
    )

    # ======================================================
    # TRAIN MODEL
    # ======================================================

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    print("Training Completed.")

    # ======================================================
    # PREDICTION
    # ======================================================

    print("\nGenerating Predictions...")

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)

    print("Prediction Completed.")

    # ======================================================
    # EVALUATION METRICS
    # ======================================================

    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # ======================================================
    # MULTICLASS ROC AUC
    # ======================================================

    auc_scores = []
    auc_weights = []

    for i, class_label in enumerate(model.classes_):

        y_test_binary = (
            y_test == class_label
        ).astype(int)

        if len(np.unique(y_test_binary)) < 2:
            continue

        class_auc = roc_auc_score(
            y_test_binary,
            y_prob[:, i]
        )

        auc_scores.append(class_auc)

        auc_weights.append(
            np.sum(y_test_binary)
        )

    if len(auc_scores) > 0:

        auc = np.average(
            auc_scores,
            weights=auc_weights
        )

    else:

        auc = np.nan

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(n_classes)
    )

    fig, ax = plt.subplots(figsize=(16, 14))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        values_format="d",
        colorbar=False,
        include_values=False
    )

    plt.title(
        f"Confusion Matrix - Fold {fold}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"confusion_matrix_fold_{fold}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual " + str(x)
            for x in class_names
        ],
        columns=[
            "Predicted " + str(x)
            for x in class_names
        ]
    )

    cm_df.to_csv(
        os.path.join(
            RESULT_DIR,
            f"confusion_matrix_fold_{fold}.csv"
        )
    )

    # ======================================================
    # MULTICLASS SPECIFICITY, FPR, FNR
    # ======================================================

    specificity_values = []
    fpr_values = []
    fnr_values = []

    for i in range(n_classes):

        TP = cm[i, i]

        FN = cm[i, :].sum() - TP

        FP = cm[:, i].sum() - TP

        TN = cm.sum() - TP - FN - FP

        specificity_i = (
            TN / (TN + FP)
            if (TN + FP) > 0 else 0
        )

        fpr_i = (
            FP / (FP + TN)
            if (FP + TN) > 0 else 0
        )

        fnr_i = (
            FN / (FN + TP)
            if (FN + TP) > 0 else 0
        )

        specificity_values.append(specificity_i)
        fpr_values.append(fpr_i)
        fnr_values.append(fnr_i)

    specificity = np.mean(specificity_values)

    fpr = np.mean(fpr_values)

    fnr = np.mean(fnr_values)

    # ======================================================
    # PRINT EVALUATION RESULTS
    # ======================================================

    print("\nEvaluation Results")
    print("-" * 40)

    print("Accuracy :", accuracy)
    print("Balanced Accuracy :", balanced_accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("Macro F1 :", macro_f1)
    print("ROC AUC  :", auc)
    print("Specificity:", specificity)
    print("FPR:", fpr)
    print("FNR:", fnr)

    print("\nConfusion Matrix")
    print(cm)

    # ======================================================
    # SAVE ROC
    # ======================================================

    roc_rows = []

    for i, class_label in enumerate(model.classes_):

        binary_y_test = (
            y_test == class_label
        ).astype(int)

        if len(np.unique(binary_y_test)) < 2:
            continue

        fpr_curve, tpr_curve, _ = roc_curve(
            binary_y_test,
            y_prob[:, i]
        )

        for fp, tp in zip(
            fpr_curve,
            tpr_curve
        ):

            roc_rows.append({
                "Class": class_names[class_label],
                "FPR": fp,
                "TPR": tp
            })

    roc_df = pd.DataFrame(
        roc_rows,
        columns=["Class", "FPR", "TPR"]
    )

    roc_df.to_csv(
        os.path.join(
            RESULT_DIR,
            f"roc_fold_{fold}.csv"
        ),
        index=False
    )

    # ======================================================
    # SAVE MULTICLASS ROC PLOT
    # ======================================================

    plt.figure(figsize=(10, 8))

    for i, class_label in enumerate(model.classes_):

        binary_y_test = (
            y_test == class_label
        ).astype(int)

        if len(np.unique(binary_y_test)) < 2:
            continue

        fpr_curve, tpr_curve, _ = roc_curve(
            binary_y_test,
            y_prob[:, i]
        )

        plt.plot(
            fpr_curve,
            tpr_curve,
            label=str(class_names[class_label])
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"Multiclass ROC Curve - Fold {fold}"
    )

    plt.legend(
        fontsize=7,
        loc="lower right",
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"roc_fold_{fold}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================================
    # STORE FOLD RESULTS
    # ======================================================

    results.append({

        "Fold": fold,

        "Accuracy": accuracy,

        "Balanced Accuracy": balanced_accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "Macro F1": macro_f1,

        "AUC": auc,

        "Specificity": specificity,

        "FPR": fpr,

        "FNR": fnr
    })

    fold += 1

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

print("="*60)
print("MODEL INFORMATION")
print("="*60)

print("Number of Trees:", len(model.estimators_))

node_counts = [tree.tree_.node_count for tree in model.estimators_]

print("Average Nodes :", np.mean(node_counts))
print("Maximum Nodes :", np.max(node_counts))
print("Minimum Nodes :", np.min(node_counts))

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "random_forest.joblib"
    )
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "10Fold_Results.csv"
    ),
    index=False
)

# ==========================================================
# EXPERIMENT SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Macro F1",
        "AUC",
        "Specificity"
    ],

    "Mean": [
        results_df["Accuracy"].mean(),
        results_df["Balanced Accuracy"].mean(),
        results_df["Precision"].mean(),
        results_df["Recall"].mean(),
        results_df["F1 Score"].mean(),
        results_df["Macro F1"].mean(),
        results_df["AUC"].mean(),
        results_df["Specificity"].mean()
    ],

    "Std": [
        results_df["Accuracy"].std(),
        results_df["Balanced Accuracy"].std(),
        results_df["Precision"].std(),
        results_df["Recall"].std(),
        results_df["F1 Score"].std(),
        results_df["Macro F1"].std(),
        results_df["AUC"].std(),
        results_df["Specificity"].std()
    ]
})

summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "experiment_summary.csv"
    ),
    index=False
)

# ==========================================================
# SAVE FOLD ACCURACY PLOT
# ==========================================================

save_fold_accuracy(
    results_df,
    RESULT_DIR
)

# ==========================================================
# SAVE METRICS BAR PLOT
# ==========================================================

save_metrics_bar(
    summary,
    RESULT_DIR
)

# ==========================================================
# SAVE FEATURE IMPORTANCE
# ==========================================================

save_feature_importance(
    model,
    feature_names,
    RESULT_DIR
)

print("\nRESULTS SAVED")
print("MULTICLASS RANDOM FOREST COMPLETED")
