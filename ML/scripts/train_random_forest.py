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
    save_confusion_matrix,
    save_roc_curve,
    save_fold_accuracy,
    save_metrics_bar,
    save_feature_importance
)

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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

CLASSIFICATION = "binary"

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
    f"RandomForest/threshold_{THRESHOLD}"
)

RESULT_DIR = (
    f"{PROJECT_ROOT}/ML/RESULTS/"
    f"RandomForest/threshold_{THRESHOLD}"
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

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(0, inplace=True)

assert np.isfinite(
    df.select_dtypes(include=[np.number]).values
).all(), "Dataset still contains NaN or Inf!"

# ==========================================================
# LABEL ENCODING
# ==========================================================

if CLASSIFICATION == "binary":

    df["Label"] = df["Label"].apply(
        lambda x: 0 if x == "BENIGN" else 1
    )

print(df["Label"].value_counts())

#FEATURES
X = df.drop(columns=["Label"])

y = df["Label"]

print(X.shape)

print(y.shape)

#NUMPY CONVERSION

X = X.values.astype(np.float32)

y = y.values

#CROSS VALIDATION
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
    
    # ==========================================================
    # FEATURE SCALING
    # ==========================================================
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    
    X_test = scaler.transform(X_test)
    
    # ==========================================================
    # RANDOM FOREST MODEL
    # ==========================================================
    model = RandomForestClassifier(

    n_estimators=100,

    random_state=SEED,

    n_jobs=-1
    
    )
    
    # ==========================================================
    # TRAIN MODEL
    # ==========================================================
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    print("Training Completed.")
    
    # ==========================================================
    # PREDICTION
    # ==========================================================
    
    print("\nGenerating Predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Prediction Completed.")
    
    accuracy = accuracy_score(y_test, y_pred)
    
    balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
    )
    
    precision = precision_score(y_test, y_pred)
    
    recall = recall_score(y_test, y_pred)
    
    f1 = f1_score(y_test, y_pred)
    
    macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
    )
    
    auc = roc_auc_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    
    save_confusion_matrix(
    cm,
    RESULT_DIR,
    fold
    )
    
    
    
    TN, FP, FN, TP = cm.ravel()
    
    specificity = TN / (TN + FP)
    
    fpr = FP / (FP + TN)
    
    fnr = FN / (FN + TP)
    
    print("\nEvaluation Results")
    
    print("-" * 40)
    
    print("Accuracy :", accuracy)
    print("Balanced Accuracy :",balanced_accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("Macro F1 :",macro_f1)
    print("ROC AUC  :", auc)
    print("\nConfusion Matrix")
    print(cm)
    
    #SAVE ROC
    fpr_curve, tpr_curve, _ = roc_curve(
    y_test,
    y_prob
    )
    roc_df = pd.DataFrame({

    "FPR": fpr_curve,

    "TPR": tpr_curve
    })
    
    roc_df.to_csv(

    os.path.join(
        RESULT_DIR,
        f"roc_fold_{fold}.csv"
    ),

    index=False
    
    )
    
    save_roc_curve(
    y_test,
    y_prob,
    RESULT_DIR,
    fold
    )
    
    
    
    #STORE FOLD_RESULTS
    results.append({

    "Fold": fold,

    "Accuracy": accuracy,
    
    "Balanced Accuracy": balanced_accuracy,

    "Precision": precision,

    "Recall": recall,

    "F1 Score": f1,
    
    "Macro F1": macro_f1

    "AUC": auc,

    "Specificity": specificity,

    "FPR": fpr,

    "FNR": fnr
    })
    
    fold += 1
    
#SAVE FINAL MODEL
joblib.dump(

    model,

    os.path.join(

        MODEL_DIR,

        "random_forest.joblib"

    )

)

#SAVE RESULTS
results_df = pd.DataFrame(results)

results_df.to_csv(

    os.path.join(

        RESULT_DIR,

        "10Fold_Results.csv"

    ),

    index=False

)

#EXPERIMENT SUMMARY
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

# Save Fold Accuracy Plot
save_fold_accuracy(
    results_df,
    RESULT_DIR
)

# Save Metrics Bar Plot
save_metrics_bar(
    summary,
    RESULT_DIR
)

feature_names = df.drop(columns=["Label"]).columns

save_feature_importance(
    model,
    feature_names,
    RESULT_DIR
)



print("RESULTS SAVED")
