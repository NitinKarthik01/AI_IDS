import os
import random
import warnings

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score
)

from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42

CLASSIFICATION = "multiclass"

THRESHOLD = "0.9"

N_SPLITS = 10

DEBUG_ONE_FOLD = False

# ==========================================================
# XGBOOST PARAMETERS
# ==========================================================

N_ESTIMATORS = 1000

MAX_DEPTH = 12

LEARNING_RATE = 0.03

SUBSAMPLE = 0.8

COLSAMPLE_BYTREE = 0.8

MIN_CHILD_WEIGHT = 3

GAMMA = 0

REG_ALPHA = 0

REG_LAMBDA = 1

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = "/home/dhanush2026/dhanush2026/AI_IDS"

DATASET_FILE = (
    f"{PROJECT_ROOT}/DATASETS/"
    f"MERGED_CSV_PCC_Selected.csv"
)

MODEL_DIR = (
    f"{PROJECT_ROOT}/ML/MODELS/XGBOOST_MULTICLASS_SMOTE/"
    f"threshold_{THRESHOLD}"
)

RESULT_DIR = (
    f"{PROJECT_ROOT}/ML/RESULTS/XGBOOST_MULTICLASS_SMOTE/"
    f"threshold_{THRESHOLD}"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
# SAVE HYPERPARAMETERS
# ==========================================================

hyperparameters = pd.DataFrame({

    "Parameter":[
        "Trees",
        "Learning Rate",
        "Max Depth",
        "Subsample",
        "Colsample",
        "Min Child Weight",
        "Gamma",
        "Regularization Alpha",
        "Regularization Lambda",
        "Cross Validation"
    ],

    "Value":[
        N_ESTIMATORS,
        LEARNING_RATE,
        MAX_DEPTH,
        SUBSAMPLE,
        COLSAMPLE_BYTREE,
        MIN_CHILD_WEIGHT,
        GAMMA,
        REG_ALPHA,
        REG_LAMBDA,
        f"{N_SPLITS}-Fold Stratified"
    ]

})

hyperparameters.to_csv(
    os.path.join(
        RESULT_DIR,
        "hyperparameters.csv"
    ),
    index=False
)

print("Hyperparameters Saved.")

# ======================================================
# RANDOM SEED
# ======================================================

np.random.seed(SEED)
random.seed(SEED)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_FILE)

print("\nOriginal Class Distribution")
print(df["Label"].value_counts())

# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder = LabelEncoder()

df["Label"] = label_encoder.fit_transform(df["Label"])

num_classes = len(label_encoder.classes_)

mapping = pd.DataFrame({

    "Original_Label": label_encoder.classes_,
    "Encoded_Label": range(num_classes)

})

mapping.to_csv(
    os.path.join(
        RESULT_DIR,
        "label_mapping.csv"
    ),
    index=False
)

print(f"\nNumber of Classes : {num_classes}")

print("\nLabel Mapping")

for i, label in enumerate(label_encoder.classes_):
    print(f"{i:2d} --> {label}")

# ==========================================================
# CLEAN DATASET
# ==========================================================

print("\nCleaning Dataset...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(0, inplace=True)

assert np.isfinite(
    df.select_dtypes(include=[np.number]).values
).all()

print("Cleaning Completed.")

# ==========================================================
# FEATURES
# ==========================================================

X = df.drop(columns=["Label"]).values.astype(np.float32)

y = df["Label"].values

print("\nFeature Matrix :", X.shape)

print("Labels         :", y.shape)

# ==========================================================
# STRATIFIED K-FOLD
# ==========================================================

skf = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=SEED

)

print("\n10-Fold Stratified Cross Validation Initialized")

# ==========================================================
# 10-FOLD CROSS VALIDATION
# ==========================================================

all_results = []

best_accuracy = 0.0

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

    print("\n" + "=" * 70)
    print(f"Fold {fold}/{N_SPLITS}")
    print("=" * 70)

    # ======================================================
    # TRAIN TEST SPLIT
    # ======================================================

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

    if fold == 1:

        joblib.dump(
            scaler,
            os.path.join(
                MODEL_DIR,
                "scaler.pkl"
            )
        )

    print("\nFeature Scaling Completed")

    # ======================================================
    # BUILD SAMPLING STRATEGY
    # ======================================================

    sampling_strategy = {}

    class_counts = (
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    print("=" * 60)
    print("Building Sampling Strategy")
    print("=" * 60)

    for label, count in class_counts.items():

        if count < 100:
            sampling_strategy[label] = 5000

        elif count < 500:
            sampling_strategy[label] = 7000

        elif count < 2000:
            sampling_strategy[label] = 10000

        elif count < 10000:
            sampling_strategy[label] = 20000

        elif count < 30000:
            sampling_strategy[label] = 30000

        elif count < 50000:
            sampling_strategy[label] = 50000

    print("\nSampling Strategy")

    for k, v in sampling_strategy.items():

        print(f"Class {k:2d}: {class_counts[k]} -> {v}")

    # ======================================================
    # APPLY SMOTE
    # ======================================================

    smote = SMOTE(

        sampling_strategy=sampling_strategy,

        random_state=SEED,

        k_neighbors=2

    )

    X_train, y_train = smote.fit_resample(

        X_train,
        y_train

    )

    print("\nAfter SMOTE")

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    # ======================================================
    # BUILD XGBOOST MODEL
    # ======================================================

    print("\nBuilding XGBoost Model...")

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=num_classes,

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        min_child_weight=MIN_CHILD_WEIGHT,

        gamma=GAMMA,

        reg_alpha=REG_ALPHA,

        reg_lambda=REG_LAMBDA,

        tree_method="hist",

        random_state=SEED,

        eval_metric="mlogloss",

        n_jobs=-1

    )

    print("Training XGBoost...\n")

    model.fit(

        X_train,

        y_train

    )

    print("Training Completed.")

    # ======================================================
    # SAVE MODEL
    # ======================================================

    model_path = os.path.join(

        MODEL_DIR,

        f"xgboost_fold_{fold}.json"

    )

    model.save_model(model_path)

    print("Model Saved.")

    # ==========================================================
    # PREDICTIONS
    # ==========================================================

    print("\nGenerating Predictions...")

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)

    print("Prediction Shape :", y_pred.shape)

    print("\ny_train distribution:")
    print(pd.Series(y_train).value_counts())

    print("\ny_test distribution:")
    print(pd.Series(y_test).value_counts())

    print("Probability Shape :", y_prob.shape)

    # ==========================================================
    # EVALUATION
    # ==========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

    print("\nEvaluation Results")
    print("-"*40)

    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Balanced Accuracy   : {balanced_accuracy:.4f}")
    print(f"Macro Precision     : {precision:.4f}")
    print(f"Macro Recall        : {recall:.4f}")
    print(f"Macro F1            : {macro_f1:.4f}")
    print(f"Macro ROC AUC       : {auc:.4f}")

    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(num_classes)
    )

    print("\nConfusion Matrix")
    print(cm)

    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    cm_df.to_csv(
        os.path.join(
            RESULT_DIR,
            f"confusion_matrix_fold_{fold}.csv"
        )
    )

    print("Confusion Matrix Saved.")

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    importance = pd.DataFrame({

        "Feature": df.drop(columns=["Label"]).columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(

        os.path.join(

            RESULT_DIR,

            f"feature_importance_fold_{fold}.csv"

        ),

        index=False

    )

    print("Feature Importance Saved.")

    # ==========================================================
    # STORE RESULTS
    # ==========================================================

    all_results.append({

        "Fold": fold,

        "Accuracy": accuracy,

        "Balanced Accuracy": balanced_accuracy,

        "Macro Precision": precision,

        "Macro Recall": recall,

        "Macro F1": macro_f1,

        "Macro ROC AUC": auc

    })

    # ==========================================================
    # SAVE BEST MODEL
    # ==========================================================

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        model.save_model(

            os.path.join(

                MODEL_DIR,

                "best_xgboost_model.json"

            )

        )

        print("Best model updated.")

    # ==========================================================
    # DEBUG
    # ==========================================================

    if DEBUG_ONE_FOLD:
        break

# ==========================================================
# FINAL RESULTS
# ==========================================================

results_df = pd.DataFrame(all_results)

print("\n")
print("="*60)
print("10-FOLD RESULTS")
print("="*60)

print(results_df)

print("\nAverage Performance")
print(results_df.mean(numeric_only=True))

print("\nStandard Deviation")
print(results_df.std(numeric_only=True))

results_df.to_csv(

    os.path.join(

        RESULT_DIR,

        "10Fold_Results.csv"

    ),

    index=False

)

summary = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Balanced Accuracy",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "Macro ROC AUC"

    ],

    "Mean":[

        results_df["Accuracy"].mean(),

        results_df["Balanced Accuracy"].mean(),

        results_df["Macro Precision"].mean(),

        results_df["Macro Recall"].mean(),

        results_df["Macro F1"].mean(),

        results_df["Macro ROC AUC"].mean()

    ],

    "Std":[

        results_df["Accuracy"].std(),

        results_df["Balanced Accuracy"].std(),

        results_df["Macro Precision"].std(),

        results_df["Macro Recall"].std(),

        results_df["Macro F1"].std(),

        results_df["Macro ROC AUC"].std()

    ]

})

summary.to_csv(

    os.path.join(

        RESULT_DIR,

        "experiment_summary.csv"

    ),

    index=False

)

print("\nResults saved successfully.")