import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def save_confusion_matrix(cm, result_dir, fold):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Attack"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(
        ax=ax,
        colorbar=False
    )

    plt.title(f"Confusion Matrix - Fold {fold}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            result_dir,
            f"confusion_matrix_fold_{fold}.png"
        ),
        dpi=300
    )

    plt.close()
    
# ==========================================================
# ROC CURVE
# ==========================================================

def save_roc_curve(y_test, y_prob, result_dir, fold):

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        "--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"ROC Curve - Fold {fold}")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            result_dir,
            f"roc_curve_fold_{fold}.png"
        ),
        dpi=300
    )

    plt.close()

# ==========================================================
# FOLD ACCURACY
# ==========================================================

def save_fold_accuracy(results_df, result_dir):

    folds = results_df["Fold"].to_numpy()
    accuracy = results_df["Accuracy"].to_numpy()

    plt.figure(figsize=(8,5))

    plt.plot(
        folds,
        accuracy,
        marker="o",
        linewidth=2
    )

    plt.xticks(folds)

    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across 10 Folds")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            result_dir,
            "fold_accuracy.png"
        ),
        dpi=300
    )

    plt.close()

def save_metrics_bar(summary, result_dir):

    metrics = summary["Metric"].to_numpy()
    mean = summary["Mean"].to_numpy()

    plt.figure(figsize=(8,5))

    plt.bar(
        metrics,
        mean
    )

    plt.ylim(0,1)

    plt.ylabel("Score")
    plt.title("Average Performance Metrics")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            result_dir,
            "metrics_bar.png"
        ),
        dpi=300
    )

    plt.close()

def save_feature_importance(

    model,

    feature_names,

    result_dir

):

    importance = model.feature_importances_

    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10,6))

    plt.bar(

        range(len(indices)),

        importance[indices]

    )

    plt.xticks(

        range(len(indices)),

        np.array(feature_names)[indices],

        rotation=90

    )

    plt.ylabel("Importance")

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            result_dir,

            "feature_importance.png"

        ),

        dpi=300

    )

    plt.close()


