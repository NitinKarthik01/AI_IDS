import os
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42

CLASSIFICATION = "multiclass"

N_SPLITS = 5                 # use 10 to match the reference script (slower)

DEBUG_ONE_FOLD = False

N_JOBS = -1

DEVICE = "cuda"              # "cuda" or "cpu" (XGBoost only)

USE_STACKING = True

USE_SMOTE = True            # True = tiered SMOTE for XGBoost only (slower)

PRINT_CONFUSION_MATRIX = False

META_MAX_SAMPLES = 300000    # max OOF rows used for weight tuning / stacker

# VAR_THRESHOLD = 0.001

# ==========================================================
# MODEL PARAMETERS
# ==========================================================

# Decision Tree
DT_CRITERION = "gini"

# Random Forest
RF_TREES = 150
RF_MAX_SAMPLES = 0.5         # bootstrap fraction per tree (smaller model)
RF_MAX_FEATURES = "sqrt"

# XGBoost
XGB_N_ESTIMATORS = 400       # upper limit, early stopping picks the best
XGB_EARLY_STOPPING = 20
MAX_DEPTH = 12
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MIN_CHILD_WEIGHT = 3
GAMMA = 0
REG_ALPHA = 0
REG_LAMBDA = 1

BASE_MODELS = ["DecisionTree", "RandomForest", "XGBoost"]

METRICS = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Macro ROC AUC"
]

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = "/home/dhanush2026/dhanush2026/AI_IDS"

DATASET_FILE = (
    f"{PROJECT_ROOT}/DATASETS/"
    f"MERGED_CSV_PCC_Selected.csv"
)

MODEL_DIR = (
    f"{PROJECT_ROOT}/ML/MODELS/"
    f"Ensemble_MULTICLASS"
)

RESULT_DIR = (
    f"{PROJECT_ROOT}/ML/RESULTS/"
    f"Ensemble_MULTICLASS"
)

FOLD_DIR = os.path.join(RESULT_DIR, "per_fold")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FOLD_DIR, exist_ok=True)

# ==========================================================
# SAVE HYPERPARAMETERS
# ==========================================================

hyperparameters = pd.DataFrame({

    "Parameter": [
        "Decision Tree Criterion",
        "RF Trees",
        "RF Max Samples",
        "RF Max Features",
        "XGB Max Trees",
        "XGB Early Stopping Rounds",
        "XGB Learning Rate",
        "XGB Max Depth",
        "XGB Subsample",
        "XGB Colsample",
        "XGB Min Child Weight",
        "XGB Gamma",
        "XGB Regularization Alpha",
        "XGB Regularization Lambda",
        "SMOTE (XGBoost only)",
        "Stacking",
        "Cross Validation"
    ],

    "Value": [
        DT_CRITERION,
        RF_TREES,
        RF_MAX_SAMPLES,
        RF_MAX_FEATURES,
        XGB_N_ESTIMATORS,
        XGB_EARLY_STOPPING,
        LEARNING_RATE,
        MAX_DEPTH,
        SUBSAMPLE,
        COLSAMPLE_BYTREE,
        MIN_CHILD_WEIGHT,
        GAMMA,
        REG_ALPHA,
        REG_LAMBDA,
        USE_SMOTE,
        USE_STACKING,
        f"{N_SPLITS}-Fold Stratified"
    ]

})

hyperparameters.to_csv(
    os.path.join(RESULT_DIR, "hyperparameters.csv"),
    index=False
)

print("Hyperparameters Saved.")

# ======================================================
# RANDOM SEED
# ======================================================

np.random.seed(SEED)
random.seed(SEED)
rng = np.random.RandomState(SEED)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_FILE)

df["Label"] = df["Label"].astype(str).str.strip()

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
    os.path.join(RESULT_DIR, "label_mapping.csv"),
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

feature_names = list(df.drop(columns=["Label"]).columns)

X = df.drop(columns=["Label"]).values.astype(np.float32)

y = df["Label"].values

del df

# ==========================================================
# FEATURE SET  (no variance filtering - all features are used)
# ==========================================================

print("=" * 60)
print("Feature Variance Analysis (informational only)")
print("=" * 60)

variance = np.var(X, axis=0)

variance_df = pd.DataFrame({
    "Feature": feature_names,
    "Variance": variance
})

variance_df = variance_df.sort_values("Variance")

print(variance_df)

variance_df.to_csv(
    os.path.join(RESULT_DIR, "feature_variance.csv"),
    index=False
)

# keep every feature
support_mask = np.ones(X.shape[1], dtype=bool)

selected_features = np.array(feature_names)

print("\nNo features removed.")

print("Features Used :", len(selected_features))

pd.DataFrame({
    "Remaining Features": selected_features
}).to_csv(
    os.path.join(RESULT_DIR, "selected_features.csv"),
    index=False
)

print("\nFeature Matrix :", X.shape)

print("Labels         :", y.shape)

class_counts_all = np.bincount(y)

if num_classes < 3:
    raise ValueError("Multiclass classification needs at least 3 classes")

if class_counts_all.min() < N_SPLITS:
    raise ValueError(
        f"Smallest class has {class_counts_all.min()} samples; "
        f"reduce N_SPLITS"
    )

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


def compute_metrics(y_true, prob):
    """Same metrics as the reference script (macro OVR ROC AUC)."""

    y_pred = prob.argmax(axis=1)

    aucs = []

    for c in range(num_classes):

        binary = (y_true == c).astype(int)

        if binary.min() != binary.max():
            aucs.append(roc_auc_score(binary, prob[:, c]))

    return {

        "Accuracy": accuracy_score(y_true, y_pred),

        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),

        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "Macro ROC AUC": float(np.mean(aucs)) if aucs else np.nan

    }


def print_metrics(m):

    print("\nEvaluation Results")
    print("-" * 40)

    print(f"Accuracy            : {m['Accuracy']:.4f}")
    print(f"Balanced Accuracy   : {m['Balanced Accuracy']:.4f}")
    print(f"Macro Precision     : {m['Macro Precision']:.4f}")
    print(f"Macro Recall        : {m['Macro Recall']:.4f}")
    print(f"Macro F1            : {m['Macro F1']:.4f}")
    print(f"Macro ROC AUC       : {m['Macro ROC AUC']:.4f}")


def save_fold_outputs(name, fold, y_true, y_pred):
    """Classification report + confusion matrix for one model / fold."""

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(FOLD_DIR, f"{name}_classification_report_fold_{fold}.csv")
    )

    print("Classification Report Saved.")

    cm = confusion_matrix(
        y_true, y_pred, labels=np.arange(num_classes)
    )

    if PRINT_CONFUSION_MATRIX:
        print("\nConfusion Matrix")
        print(cm)

    pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    ).to_csv(
        os.path.join(FOLD_DIR, f"{name}_confusion_matrix_fold_{fold}.csv")
    )

    print("Confusion Matrix Saved.")


def save_feature_importance(name, fold, model):

    importance = pd.DataFrame({

        "Feature": selected_features,

        "Importance": model.feature_importances_

    }).sort_values(by="Importance", ascending=False)

    importance.to_csv(
        os.path.join(FOLD_DIR, f"{name}_feature_importance_fold_{fold}.csv"),
        index=False
    )

    print("Feature Importance Saved.")


def full_proba(model, X_in):
    """predict_proba aligned to all num_classes columns."""

    p = model.predict_proba(X_in)

    if p.shape[1] == num_classes:
        return p.astype(np.float32)

    out = np.zeros((len(X_in), num_classes), dtype=np.float32)
    out[:, model.classes_] = p

    return out


def soft_weights(labels):
    """Square-root balanced sample weights."""
    return np.sqrt(compute_sample_weight("balanced", labels))


def apply_smote(X_in, y_in):
    """Tiered SMOTE (same tiers as the reference script), training data only."""

    scaler = StandardScaler()

    Xs = scaler.fit_transform(X_in)

    counts = pd.Series(y_in).value_counts().sort_index()

    sampling_strategy = {}

    for label, count in counts.items():

        if count < 2:
            continue
        elif count < 100:
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

    if not sampling_strategy:
        print("\nNo classes require SMOTE. Skipping oversampling.")
        return X_in, y_in

    print("\nSampling Strategy")

    for k, v in sampling_strategy.items():
        print(f"Class {k:2d}: {counts[k]} -> {v}")

    k_neighbors = max(1, min(2, min(counts[l] for l in sampling_strategy) - 1))

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=SEED,
        k_neighbors=k_neighbors
    )

    Xr, yr = smote.fit_resample(Xs, y_in)

    # back to the original feature scale (trees do not need scaling)
    return scaler.inverse_transform(Xr).astype(np.float32), yr


def make_xgb(n_estimators, early_stopping=None):

    return XGBClassifier(

        objective="multi:softprob",

        num_class=num_classes,

        n_estimators=n_estimators,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        min_child_weight=MIN_CHILD_WEIGHT,

        gamma=GAMMA,

        reg_alpha=REG_ALPHA,

        reg_lambda=REG_LAMBDA,

        tree_method="hist",

        device=DEVICE,

        max_bin=256,

        random_state=SEED,

        eval_metric="mlogloss",

        early_stopping_rounds=early_stopping,

        verbosity=0,

        n_jobs=N_JOBS

    )


def train_xgboost(X_tr, y_tr, n_estimators=None):
    """Train XGBoost. With early stopping unless n_estimators is given
    (used for the final fit). Returns (model, number_of_trees_used)."""

    val_idx = None
    fit_idx = np.arange(len(y_tr))

    if n_estimators is None:

        try:
            fit_idx, val_idx = train_test_split(
                np.arange(len(y_tr)),
                test_size=0.1,
                stratify=y_tr,
                random_state=SEED
            )
        except ValueError:
            fit_idx, val_idx = np.arange(len(y_tr)), None

    X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]

    if USE_SMOTE:
        X_fit, y_fit = apply_smote(X_fit, y_fit)

    w = soft_weights(y_fit)

    if val_idx is not None:

        model = make_xgb(XGB_N_ESTIMATORS, XGB_EARLY_STOPPING)

        model.fit(
            X_fit, y_fit,
            sample_weight=w,
            eval_set=[(X_tr[val_idx], y_tr[val_idx])],
            verbose=False
        )

        return model, int(model.best_iteration) + 1

    n_trees = n_estimators or XGB_N_ESTIMATORS

    model = make_xgb(n_trees)

    model.fit(X_fit, y_fit, sample_weight=w, verbose=False)

    return model, n_trees


def build_and_train(name, X_tr, y_tr, xgb_trees=None):

    if name == "DecisionTree":

        model = DecisionTreeClassifier(
            criterion=DT_CRITERION,
            random_state=SEED
        )

        return model.fit(X_tr, y_tr), None

    if name == "RandomForest":

        model = RandomForestClassifier(
            n_estimators=RF_TREES,
            criterion="gini",
            max_features=RF_MAX_FEATURES,
            max_samples=RF_MAX_SAMPLES,
            class_weight="balanced_subsample",
            n_jobs=N_JOBS,
            random_state=SEED
        )

        return model.fit(X_tr, y_tr), None

    return train_xgboost(X_tr, y_tr, n_estimators=xgb_trees)


def hard_vote(probs):

    soft = np.mean(probs, axis=0)

    counts = np.zeros_like(soft)

    for p in probs:
        counts[np.arange(len(p)), p.argmax(axis=1)] += 1

    out = np.zeros_like(soft)

    # tie-break with soft-voting probability
    out[np.arange(len(soft)), (counts + 1e-6 * soft).argmax(axis=1)] = 1.0

    return out


def tune_weights(probs, labels):
    """Grid search voting weights on macro F1 (on OOF probabilities)."""

    idx = np.arange(len(labels))

    if len(idx) > META_MAX_SAMPLES:
        idx = rng.choice(idx, META_MAX_SAMPLES, replace=False)

    P = [p[idx] for p in probs]

    best, best_score = np.ones(len(P)), -1.0

    for w in [(a, b, c) for a in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3)]:

        pred = np.average(P, axis=0, weights=w).argmax(axis=1)

        score = f1_score(labels[idx], pred, average="macro", zero_division=0)

        if score > best_score:
            best, best_score = np.asarray(w, dtype=float), score

    return best


def stack_features(probs):

    return np.hstack(
        [np.log(np.clip(p, 1e-6, 1.0)) for p in probs]
    ).astype(np.float32)


def fit_stacker(probs, labels):

    idx = np.arange(len(labels))

    if len(idx) > META_MAX_SAMPLES:
        idx = rng.choice(idx, META_MAX_SAMPLES, replace=False)

    lr = LogisticRegression(
        max_iter=300,
        class_weight="balanced"
    )

    lr.fit(stack_features([p[idx] for p in probs]), labels[idx])

    return lr


def stacker_proba(lr, probs):

    p = lr.predict_proba(stack_features(probs))

    if p.shape[1] == num_classes:
        return p

    out = np.zeros((len(p), num_classes))
    out[:, lr.classes_] = p

    return out


def load_and_predict(model_dir, data_frame, method="WeightedVoting"):
    """Load the saved models and predict labels for a DataFrame of raw features.
    method: HardVoting | SoftVoting | WeightedVoting | Stacking"""

    meta = joblib.load(os.path.join(model_dir, "meta.joblib"))

    X_new = (
        data_frame[meta["all_feature_names"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(np.float32)
    )[:, meta["support_mask"]]

    dt = joblib.load(os.path.join(model_dir, "decision_tree.joblib"))
    rf = joblib.load(os.path.join(model_dir, "random_forest.joblib"))

    xg = XGBClassifier()
    xg.load_model(os.path.join(model_dir, "xgboost.ubj"))

    probs = [full_proba(m, X_new) for m in (dt, rf, xg)]

    if method == "Stacking":
        p = stacker_proba(meta["stacker"], probs)
    elif method == "SoftVoting":
        p = np.mean(probs, axis=0)
    elif method == "HardVoting":
        p = hard_vote(probs)
    else:
        p = np.average(probs, axis=0, weights=meta["weights"])

    return meta["label_encoder"].inverse_transform(p.argmax(axis=1))


# ==========================================================
# STRATIFIED K-FOLD
# ==========================================================

skf = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=SEED

)

splits = list(skf.split(X, y))

print(f"\n{N_SPLITS}-Fold Stratified Cross Validation Initialized")

# ==========================================================
# CROSS VALIDATION  (each base model trained ONCE per fold)
# ==========================================================

all_results = []

oof = {
    name: np.zeros((len(y), num_classes), dtype=np.float32)
    for name in BASE_MODELS
}

xgb_best_trees = []

done_folds = []

total_start = time.time()

for fold, (train_idx, test_idx) in enumerate(splits, start=1):

    print("\n" + "=" * 70)
    print(f"Fold {fold}/{N_SPLITS}")
    print("=" * 70)

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Training Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)

    print("Number of classes in y_test :", len(np.unique(y_test)))
    print("Total classes :", num_classes)

    for name in BASE_MODELS:

        print("\n" + "-" * 70)
        print(f"Training {name}...")
        print("-" * 70)

        t0 = time.time()

        model, n_trees = build_and_train(name, X_train, y_train)

        if name == "XGBoost":
            xgb_best_trees.append(n_trees)
            print(f"XGBoost trees used (early stopping) : {n_trees}")

        print(f"Training Completed. ({time.time() - t0:.0f}s)")

        y_prob = full_proba(model, X_test)

        y_pred = y_prob.argmax(axis=1)

        oof[name][test_idx] = y_prob

        print("Prediction Shape :", y_pred.shape)
        print("Probability Shape :", y_prob.shape)

        metrics = compute_metrics(y_test, y_prob)

        print(f"\n[{name}]")

        print_metrics(metrics)

        save_fold_outputs(name, fold, y_test, y_pred)

        save_feature_importance(name, fold, model)

        all_results.append({"Fold": fold, "Model": name, **metrics})

        del model   # fold models are never kept -> no giant model folders

    done_folds.append(fold)

    if DEBUG_ONE_FOLD:
        break

# ==========================================================
# ENSEMBLES FROM OUT-OF-FOLD PROBABILITIES  (no retraining)
# ==========================================================

print("\n" + "=" * 70)
print("Evaluating Ensembles (Voting / Stacking) on OOF Probabilities")
print("=" * 70)

multi_fold = len(done_folds) > 1

ENSEMBLES = ["HardVoting", "SoftVoting", "WeightedVoting"]

if USE_STACKING and multi_fold:
    ENSEMBLES.append("Stacking")

overall_pred = {
    name: np.full(len(y), -1, dtype=int)
    for name in ENSEMBLES
}

weight_rows = []

for fold in done_folds:

    train_idx, test_idx = splits[fold - 1]

    y_test = y[test_idx]

    P_te = [oof[n][test_idx] for n in BASE_MODELS]

    # meta-learners are tuned only on the OOF rows of the OTHER folds
    if multi_fold:

        pool = np.concatenate(
            [splits[f - 1][1] for f in done_folds if f != fold]
        )

        P_pool = [oof[n][pool] for n in BASE_MODELS]

        weights = tune_weights(P_pool, y[pool])

    else:

        pool = None

        weights = np.ones(len(BASE_MODELS))

    weight_rows.append(
        {"Fold": fold, **dict(zip(BASE_MODELS, weights.tolist()))}
    )

    method_probs = {

        "HardVoting": hard_vote(P_te),

        "SoftVoting": np.mean(P_te, axis=0),

        "WeightedVoting": np.average(P_te, axis=0, weights=weights)

    }

    if "Stacking" in ENSEMBLES:

        stacker = fit_stacker(P_pool, y[pool])

        method_probs["Stacking"] = stacker_proba(stacker, P_te)

    print("\n" + "=" * 70)
    print(f"Fold {fold}/{N_SPLITS} - Ensembles   (weights: {weights.tolist()})")
    print("=" * 70)

    for name, prob in method_probs.items():

        y_pred = prob.argmax(axis=1)

        overall_pred[name][test_idx] = y_pred

        metrics = compute_metrics(y_test, prob)

        print(f"\n[{name}]")

        print_metrics(metrics)

        save_fold_outputs(name, fold, y_test, y_pred)

        all_results.append({"Fold": fold, "Model": name, **metrics})

pd.DataFrame(weight_rows).to_csv(
    os.path.join(RESULT_DIR, "selected_voting_weights.csv"),
    index=False
)

# ==========================================================
# FINAL RESULTS
# ==========================================================

results_df = pd.DataFrame(all_results)

print("\n")
print("=" * 60)
print(f"{N_SPLITS}-FOLD RESULTS")
print("=" * 60)

ALL_MODELS = BASE_MODELS + ENSEMBLES

for name in ALL_MODELS:

    print(f"\n{name}")

    print(
        results_df[results_df["Model"] == name]
        .drop(columns=["Model"])
        .to_string(index=False)
    )

mean_df = results_df.groupby("Model")[METRICS].mean().loc[ALL_MODELS]

std_df = results_df.groupby("Model")[METRICS].std().loc[ALL_MODELS]

print("\nAverage Performance")
print(mean_df)

print("\nStandard Deviation")
print(std_df)

results_df.to_csv(
    os.path.join(RESULT_DIR, f"{N_SPLITS}Fold_Results.csv"),
    index=False
)

summary = pd.concat(
    [mean_df.add_suffix(" Mean"), std_df.add_suffix(" Std")],
    axis=1
).reset_index()

summary.to_csv(
    os.path.join(RESULT_DIR, "experiment_summary.csv"),
    index=False
)

comparison = mean_df.sort_values("Macro F1", ascending=False)

comparison.to_csv(
    os.path.join(RESULT_DIR, "model_comparison.csv")
)

print("\nModel Ranking (by Macro F1)")
print(comparison[["Accuracy", "Macro F1", "Balanced Accuracy"]])

# overall out-of-fold confusion matrices
done_mask = np.concatenate([splits[f - 1][1] for f in done_folds])

for name in ALL_MODELS:

    pred = (
        oof[name].argmax(axis=1)
        if name in BASE_MODELS
        else overall_pred[name]
    )

    cm = confusion_matrix(
        y[done_mask],
        pred[done_mask],
        labels=np.arange(num_classes)
    )

    pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    ).to_csv(
        os.path.join(RESULT_DIR, f"{name}_overall_confusion_matrix.csv")
    )

print("\nOverall Confusion Matrices Saved.")

# ==========================================================
# FINAL FIT ON ALL DATA  (3 base models, saved once)
# ==========================================================

if not DEBUG_ONE_FOLD:

    print("\n" + "=" * 70)
    print("Fitting Final Models on All Data...")
    print("=" * 70)

    P_all = [oof[n] for n in BASE_MODELS]

    final_weights = tune_weights(P_all, y)

    final_stacker = (
        fit_stacker(P_all, y)
        if USE_STACKING
        else None
    )

    best_trees = int(np.median(xgb_best_trees))

    print("Final voting weights :", dict(zip(BASE_MODELS, final_weights.tolist())))
    print("Final XGBoost trees  :", best_trees)

    for name, file_name in [
        ("DecisionTree", "decision_tree.joblib"),
        ("RandomForest", "random_forest.joblib"),
        ("XGBoost", "xgboost.ubj")
    ]:

        t0 = time.time()

        model, _ = build_and_train(
            name, X, y,
            xgb_trees=best_trees if name == "XGBoost" else None
        )

        path = os.path.join(MODEL_DIR, file_name)

        if name == "XGBoost":
            model.save_model(path)
        else:
            joblib.dump(model, path, compress=3)

        print(
            f"Saved {file_name}: "
            f"{os.path.getsize(path) / 1e6:.1f} MB "
            f"({time.time() - t0:.0f}s)"
        )

        del model

    joblib.dump(

        {
            "weights": final_weights,
            "stacker": final_stacker,
            "label_encoder": label_encoder,
            "all_feature_names": feature_names,
            "selected_features": selected_features.tolist(),
            "support_mask": support_mask,
            "model_names": BASE_MODELS,
            "default_method": "WeightedVoting"
        },

        os.path.join(MODEL_DIR, "meta.joblib"),

        compress=3

    )

    print("Meta file saved.")

print(f"\nTotal time : {(time.time() - total_start) / 60:.1f} min")

print("\nResults saved successfully.")