import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Flatten,
    Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42

CLASSIFICATION = "binary"

THRESHOLD = "0.10"

N_SPLITS = 10

BATCH_SIZE = 256

EPOCHS = 50

LEARNING_RATE = 0.001

DEBUG_ONE_FOLD = False


# ==========================================================
# BUILD PCC-CNN MODEL
# ==========================================================

def build_pcc_cnn(input_shape, classification="binary", num_classes=None):

    model = Sequential(name="PCC_CNN")

    # ------------------------------------------------------
    # Input Layer
    # ------------------------------------------------------
    model.add(Input(shape=input_shape))

    # ------------------------------------------------------
    # Convolution Block 1
    # ------------------------------------------------------
    model.add(
        Conv1D(
            filters=96,
            kernel_size=4,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv1"
        )
    )

    # ------------------------------------------------------
    # Convolution Block 2
    # ------------------------------------------------------
    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv2"
        )
    )

    # ------------------------------------------------------
    # Convolution Block 3
    # ------------------------------------------------------
    model.add(
        Conv1D(
            filters=32,
            kernel_size=2,
            strides=1,
            padding="same",
            activation="relu",
            name="Conv3"
        )
    )

    # ------------------------------------------------------
    # Flatten
    # ------------------------------------------------------
    model.add(Flatten(name="Flatten"))

    # ------------------------------------------------------
    # Fully Connected Layers
    # ------------------------------------------------------
    model.add(Dense(512, activation="relu", name="Dense_512"))

    model.add(Dense(128, activation="relu", name="Dense_128"))

    model.add(Dense(32, activation="relu", name="Dense_32"))

    # ------------------------------------------------------
    # Output Layer
    # ------------------------------------------------------
    if classification == "binary":

        model.add(Dense(1, activation="sigmoid", name="Output"))

        loss_function = "binary_crossentropy"

    else:

        model.add(
            Dense(
                num_classes,
                activation="softmax",
                name="Output"
            )
        )

        loss_function = "sparse_categorical_crossentropy"

    # ------------------------------------------------------
    # Compile
    # ------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=loss_function,
        metrics=["accuracy"]
    )

    return model
    
# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = "/home/dhanush2026/dhanush2026/AI_IDS"

DATASET_FILE = (
    f"{PROJECT_ROOT}/PCC_RESULTS/"
    f"threshold_{THRESHOLD}/reduced_dataset.csv"
)

MODEL_DIR = (
    f"{PROJECT_ROOT}/MODELS/"
    f"threshold_{THRESHOLD}"
)

RESULT_DIR = (
    f"{PROJECT_ROOT}/RESULTS/"
    f"threshold_{THRESHOLD}"
)


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ======================================================
# RANDOM SEED
# ======================================================
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ===========================
# DATASET PATH
# ===========================

DATASET_FILE = (
    f"/home/dhanush2026/dhanush2026/AI_IDS/"
    f"PCC_RESULTS/threshold_{THRESHOLD}/reduced_dataset.csv"
)

# ===========================
# LOAD DATASET
# ===========================

print("=" * 60)
print("Loading Reduced Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_FILE)

print(DATASET_FILE)
print("\nImmediately after loading:")
print(df["Label"].head())

print("\nUnique labels after loading:")
print(df["Label"].unique()[:10])

print("\nData types:")
print(df.dtypes)

print(f"Threshold : {THRESHOLD}")
print(f"Dataset Shape : {df.shape}")
print()

print(df.head())

print("\nColumns\n")
print(df.columns.tolist())

print(df["Label"].value_counts())

# ==========================================================
# LABEL ENCODING
# ==========================================================

print("\n" + "=" * 60)
print("Label Encoding")
print("=" * 60)

# Keep original labels untouched
y_original = df["Label"].copy()

if CLASSIFICATION == "binary":

    y = y_original.apply(lambda x: 0 if x == "BENIGN" else 1).values

    print("\nBinary Classification Selected")
    print(pd.Series(y).value_counts())

else:

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    print("\nMulti-Class Classification Selected")
    print(f"Number of Classes : {len(encoder.classes_)}")

    for i, label in enumerate(encoder.classes_):
        print(f"{i:2d} --> {label}")        
# ==========================================================
# CLEAN DATASET
# ==========================================================

print("\nCleaning dataset...")

# Replace Inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN with 0
df.fillna(0, inplace=True)

# Verify
assert np.isfinite(
    df.select_dtypes(include=[np.number]).values
).all(), "Dataset still contains NaN/Inf!"

print("Cleaning Completed.")

# ==========================================================
# FEATURES & LABELS
# ==========================================================

X = df.drop(columns=["Label"])

print("\nFeature Matrix Shape :", X.shape)
print("Label Vector Shape   :", y.shape)

# ==========================================================
# CONVERT TO NUMPY
# ==========================================================

X = X.values.astype(np.float32)
y = y

print("\nNumPy Conversion Completed")
print("X Shape :", X.shape)
print("y Shape :", y.shape)

# ==========================================================
# STRATIFIED 10-FOLD CROSS VALIDATION
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
# ==========================================================
# RESULTS CONTAINER
# ==========================================================

all_results = []

best_accuracy = 0.0
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

    print("\n" + "=" * 70)
    print(f"Fold {fold}/{N_SPLITS}")
    print("=" * 70)

    # ------------------------------------------------------
    # Train-Test Split
    # ------------------------------------------------------

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Training Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)
    
    # ==========================================================
    # CHECK TRAINING DATA
    # ==========================================================
    feature_names = df.drop(columns=["Label"]).columns
    print("\nChecking training data...\n")
    for i, feature in enumerate(feature_names):
     col = X_train[:, i]

     if np.isinf(col).any() or np.isnan(col).any():

        print(f"{feature}")

        print(" Positive Inf :", np.isposinf(col).sum())
        print(" Negative Inf :", np.isneginf(col).sum())
        print(" NaN :", np.isnan(col).sum())

    print("\nMaximum value in X_train :", np.nanmax(X_train))
    print("Minimum value in X_train :", np.nanmin(X_train))

    # ------------------------------------------------------
    # Feature Scaling
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nFeature Scaling Completed")
    import joblib
    # Save scaler (only once)
    if fold == 1:
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        joblib.dump(
        scaler,
        os.path.join(
                  MODEL_SAVE_PATH,
                            "scaler.pkl"
        )
       )
       print("Scaler Saved.")
       
    print("Training Mean :", np.mean(X_train))
    print("Training Std  :", np.std(X_train))

    # ------------------------------------------------------
    # Reshape for CNN
    # ------------------------------------------------------

    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    print("\nReshaped Dataset")
    print("X_train :", X_train.shape)
    print("X_test  :", X_test.shape)

    # ------------------------------------------------------
    # Build PCC-CNN Model
    # ------------------------------------------------------

    print("\nBuilding PCC-CNN Model...")

    model = build_pcc_cnn(
        input_shape=(X_train.shape[1], 1),
        classification=CLASSIFICATION,
        num_classes=len(np.unique(y))
    )

    print("Model Created Successfully\n")

    model.summary()
    
    # ==========================================================
    # CALLBACKS
    # ==========================================================
    model_path = os.path.join(
    MODEL_DIR,
    f"fold_{fold}_best.keras"
    )
    
    early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
    filepath=model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
    )
    
    print("\nCallbacks Initialized")
    
    # ==========================================================
    # TRAIN MODEL
    # ==========================================================
    
    print("\nTraining Started...\n")
    
    history = model.fit(

    X_train,
    y_train,

    validation_data=(X_test, y_test),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        model_checkpoint
    ],

    verbose=1
    )
    # ==========================================================
    # SAVE TRAINING HISTORY
    # ==========================================================
    history_df = pd.DataFrame(history.history)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
    os.path.join(
        RESULT_SAVE_PATH,
        f"loss_fold_{fold}.png"
        )
        
      )
      
    plt.close()
    
    plt.figure(figsize=(8,5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
    os.path.join(
        RESULT_SAVE_PATH,
        f"accuracy_fold_{fold}.png"
        )
        
       )
       
    plt.close()
    
    history_df.to_csv(
    os.path.join(
        RESULT_SAVE_PATH,

        f"history_fold_{fold}.csv"
        
        ),
    index=False
    )
    
    print("Training History Saved.")
    
    print("\nTraining Completed.")
    
    # ==========================================================
    # PREDICTION
    # ==========================================================
    
    print("\nGenerating Predictions...")
    # Probability predictions
    y_prob = model.predict(X_test, verbose=0)
    
    # Binary predictions
    if CLASSIFICATION == "binary":
        y_prob = y_prob.flatten()
        y_pred = (y_prob > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_prob, axis=1)
    print("Prediction Completed.")
    
    print("Prediction Shape :", y_pred.shape)
    
    print("\ny_train distribution:")
    print(pd.Series(y_train).value_counts())
    
    print("\ny_test distribution:")
    print(pd.Series(y_test).value_counts())
    
    
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(
    y_test,
    y_prob
    )
    
    roc_df = pd.DataFrame({
    "Threshold": roc_thresholds,

    "FPR": fpr_curve,

    "TPR": tpr_curve
    })
    
    roc_df.to_csv(

    os.path.join(

        RESULT_SAVE_PATH,

        f"roc_fold_{fold}.csv"

    ),

    index=False
    )
    
    # ==========================================================
    # EVALUATION METRICS
    # ==========================================================#
    
    accuracy = accuracy_score(y_test, y_pred)
    
    
    precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
    )
    
    recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
    )
    
    f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
    )
    
    auc = roc_auc_score(
    y_test,
    y_prob
    )
    
    print("\nEvaluation Results")
    print("-" * 40)
    
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    
    
    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================
    cm = confusion_matrix(
    y_test,
    y_pred
    )
    
    TN, FP, FN, TP = cm.ravel()
    print("\nConfusion Matrix")
    print(cm)
    
    print("\nTN :", TN)
    print("FP :", FP)
    print("FN :", FN)
    print("TP :", TP)
    
    # ==========================================================
    # SAVE CONFUSION MATRIX
    # ==========================================================
    cm_df = pd.DataFrame(

    cm,

    index=[
        "Actual Benign",
        "Actual Attack"
    ],

    columns=[
        "Predicted Benign",
        "Predicted Attack"
    ]
    )
    
    cm_df.to_csv(

    os.path.join(

        RESULT_SAVE_PATH,

        f"confusion_matrix_fold_{fold}.csv"
        )
    )
    print("Confusion Matrix Saved.")
    
    # ==========================================================
    # ADDITIONAL METRICS
    # ==========================================================
    
    specificity = TN / (TN + FP)
    
    fpr = FP / (FP + TN)
    
    fnr = FN / (FN + TP)
    
    # ==========================================================
    # STORE CURRENT FOLD RESULTS
    # ==========================================================
    
    print("\nSpecificity :", round(specificity,4))
    print("False Positive Rate :", round(fpr,4))
    print("False Negative Rate :", round(fnr,4))
    
    all_results.append({
    "Fold": fold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "AUC": auc,
    "Specificity": specificity,
    "FPR": fpr,
    "FNR": fnr
    })
    
    # ==========================================================
    # SAVE BEST MODEL
    # ==========================================================
    if accuracy > best_accuracy:
     
     best_accuracy = accuracy
     
     model.save(

        os.path.join(

            MODEL_DIR,

            "best_model.keras"

        )

     )
     
     print("Best model updated.")
    # ------------------------------------------------------
    # Debug Mode
    # ------------------------------------------------------

    if DEBUG_ONE_FOLD:
        break
 # ==========================================================
# SAVE FINAL RESULTS
# ==========================================================

results_df = pd.DataFrame(all_results)

print("\n")
print("=" * 60)
print("10-FOLD RESULTS")
print("=" * 60)

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
        "Precision",
        "Recall",
        "F1",
        "AUC",
        "Specificity"
    ],

    "Mean":[
        results_df["Accuracy"].mean(),
        results_df["Precision"].mean(),
        results_df["Recall"].mean(),
        results_df["F1 Score"].mean(),
        results_df["AUC"].mean(),
        results_df["Specificity"].mean()
    ],

    "Std":[
        results_df["Accuracy"].std(),
        results_df["Precision"].std(),
        results_df["Recall"].std(),
        results_df["F1 Score"].std(),
        results_df["AUC"].std(),
        results_df["Specificity"].std()
    ]

})

summary.to_csv(

    os.path.join(
        RESULT_SAVE_PATH,
        "experiment_summary.csv"
    ),

    index=False

)

print("\nResults saved successfully.")
