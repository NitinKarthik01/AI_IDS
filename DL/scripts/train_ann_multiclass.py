import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf


from architecture import save_model_architecture

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization
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

CLASSIFICATION = "multiclass"

THRESHOLD = "0.10"

N_SPLITS = 10

BATCH_SIZE = 256

EPOCHS = 50

LEARNING_RATE = 0.001

DEBUG_ONE_FOLD = False


# ==========================================================
# BUILD ANN MODEL
# ==========================================================

def build_pcc_ann(input_shape, classification="binary", num_classes=None):

    model = Sequential(name="ANN")

    # ------------------------------------------------------
    # Input Layer
    # ------------------------------------------------------
    model.add(
        Input(shape=(input_shape,))
    )

    # ------------------------------------------------------
    # Hidden Layer 1
    # ------------------------------------------------------
    model.add(
        Dense(
            512,
            activation="relu",
            name="Dense_512"
        )
    )

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    # ------------------------------------------------------
    # Hidden Layer 2
    # ------------------------------------------------------
    model.add(
        Dense(
            128,
            activation="relu",
            name="Dense_128"
        )
    )

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    # ------------------------------------------------------
    # Hidden Layer 3
    # ------------------------------------------------------
    model.add(
        Dense(
            32,
            activation="relu",
            name="Dense_32"
        )
    )

    # ------------------------------------------------------
    # Output Layer
    # ------------------------------------------------------
    if classification == "binary":

        model.add(
            Dense(
                1,
                activation="sigmoid",
                name="Output"
            )
        )

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
    f"{PROJECT_ROOT}/DL/MODELS/ANN_MULTICLASS/"
    f"threshold_{THRESHOLD}"
)

RESULT_DIR = (
    f"{PROJECT_ROOT}/DL/RESULTS/ANN_MULTICLASS/"
    f"threshold_{THRESHOLD}"
)


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
# SAVE HYPERPARAMETERS
# ==========================================================

hyperparameters = pd.DataFrame({

    "Parameter":[
        "Epochs",
        "Batch Size",
        "Learning Rate",
        "Optimizer",
        "Loss Function",
        "Hidden Layers",
        "Activation",
        "Dropout",
        "Cross Validation",
        "Early Stopping",
        "Class Weights"
    ],

    "Value":[
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        "Adam",
        "Sparse Categorical Crossentropy",
        "512 -> 128 -> 32",
        "ReLU + Softmax",
        "0.30",
        f"{N_SPLITS}-Fold Stratified",
        "10",
        "Balanced"
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

print("Original Class Distribution:")
print(df["Label"].value_counts())

# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder = LabelEncoder()

df["Label"] = label_encoder.fit_transform(df["Label"])

num_classes = len(label_encoder.classes_)

# Save label mapping
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

print("\nLabel Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i:2d} --> {label}")

# Features and Labels
X = df.drop(columns=["Label"]).values
y = df["Label"].values
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
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(
        scaler,
        os.path.join(
                  MODEL_DIR,
                            "scaler.pkl"
        )
       )
    print("Scaler Saved.")
       
    print("Training Mean :", np.mean(X_train))
    print("Training Std  :", np.std(X_train))

   

    print("\nANN Input Shape")
    print("X_train :", X_train.shape)
    print("X_test  :", X_test.shape)

    # ------------------------------------------------------
    # Build PCC-ANN Model
    # ------------------------------------------------------

    print("\nBuilding PCC-ANN Model...")

    model = build_pcc_ann(
        input_shape=X_train.shape[1],
        classification=CLASSIFICATION,
        num_classes=num_classes
    )
    
    save_model_architecture(
    model,
    RESULT_DIR,
    "ann_multiclass"
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
    
    
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
    )
    
    class_weights = dict(zip(classes, weights))
    
    print(class_weights)
    
    print("\nTraining Started...\n")
    
    history = model.fit(

    X_train,
    y_train,

    validation_data=(X_test, y_test),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,
    
    class_weight=class_weights,

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
        RESULT_DIR,
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
        RESULT_DIR,
        f"accuracy_fold_{fold}.png"
        )
        
       )
       
    plt.close()
    
    history_df.to_csv(
    os.path.join(
        RESULT_DIR,

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
    y_prob = model.predict(
    X_test,
    verbose=0
    )
    y_pred = np.argmax(
    y_prob,
    axis=1
    )
    
    print("Prediction Shape :", y_pred.shape)
    
    print("\ny_train distribution:")
    print(pd.Series(y_train).value_counts())
    
    print("\ny_test distribution:")
    print(pd.Series(y_test).value_counts())
    
    
    #fpr_curve, tpr_curve, roc_thresholds = roc_curve(
    #y_test,
    #y_prob
    #)
    
    # roc_df = pd.DataFrame({
    # "Threshold": roc_thresholds,

    # "FPR": fpr_curve,

    # "TPR": tpr_curve
    # })
    
    # roc_df.to_csv(

    # os.path.join(

    #     RESULT_DIR,

    #     f"roc_fold_{fold}.csv"

    # ),

    # index=False
    # )
    
    # ==========================================================
    # EVALUATION METRICS
    # ==========================================================#
    
    accuracy = accuracy_score(y_test, y_pred)

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

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1
    
    print("Unique classes in y_test:", len(np.unique(y_test)))
    print("Probability shape:", y_prob.shape)
   # auc = roc_auc_score(
    #    y_test,
     #   y_prob,
      #  multi_class="ovr",
       # average="macro",
        #labels=np.arange(num_classes)
    #)

    print("\nEvaluation Results")
    print("-"*40)

    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Balanced Accuracy   : {balanced_accuracy:.4f}")
    print(f"Macro Precision     : {precision:.4f}")
    print(f"Macro Recall        : {recall:.4f}")
    print(f"Macro F1            : {macro_f1:.4f}")
    #print(f"Macro ROC AUC       : {auc:.4f}")
    
    
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
    
    
    # ==========================================================
    # SAVE CONFUSION MATRIX
    # ==========================================================
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
    # STORE CURRENT FOLD RESULTS
    # ==========================================================
    
    # print("\nSpecificity :", round(specificity,4))
    # print("False Positive Rate :", round(fpr,4))
    # print("False Negative Rate :", round(fnr,4))
    
    all_results.append({
    
    "Fold": fold,

    "Accuracy": accuracy,

    "Balanced Accuracy": balanced_accuracy,

    "Macro Precision": precision,

    "Macro Recall": recall,

    "Macro F1": macro_f1,

    #"Macro ROC AUC": auc

    })
    
    # ==========================================================
    # SAVE BEST MODEL
    # ==========================================================
    if accuracy > best_accuracy:
     
     best_accuracy = accuracy
     
     model.save(

        os.path.join(

            MODEL_DIR,

            "best_ann_multiclass_model.keras"

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

print(results_df.head())

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
        "Balanced Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
       # "Macro ROC AUC"
    ],

    "Mean":[

        results_df["Accuracy"].mean(),

        results_df["Balanced Accuracy"].mean(),

        results_df["Macro Precision"].mean(),

        results_df["Macro Recall"].mean(),

        results_df["Macro F1"].mean(),

     #   results_df["Macro ROC AUC"].mean()

    ],

    "Std":[
        results_df["Accuracy"].std(),

        results_df["Balanced Accuracy"].std(),

        results_df["Macro Precision"].std(),

        results_df["Macro Recall"].std(),

        results_df["Macro F1"].std(),

      #  results_df["Macro ROC AUC"].std()
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
