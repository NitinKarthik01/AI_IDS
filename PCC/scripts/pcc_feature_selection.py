import os
import pandas as pd
import numpy as np

# -----------------------------
# Dataset Path
# -----------------------------

DATASET_PATH = "/home/dhanush2026/dhanush2026/AI_IDS/DATASETS/MERGED_CSV_Sampled_5Percent.csv"

OUTPUT_DIR = "/home/dhanush2026/dhanush2026/AI_IDS/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# PCC Thresholds
# -----------------------------

thresholds = np.arange(0.10, 0.55, 0.05)

# -----------------------------
# Read Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape :", df.shape)

print("\nCleaning Dataset...")

# Replace infinity values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Missing values before filling
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fill missing numeric values using median
df.fillna(df.median(numeric_only=True), inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

print("\nChecking for Infinity values")

numeric_df = df.select_dtypes(include=np.number)

print("Positive Infinity :", np.isposinf(numeric_df).sum().sum())
print("Negative Infinity :", np.isneginf(numeric_df).sum().sum())

from sklearn.preprocessing import LabelEncoder

# -------------------------------------
# Encode Labels
# -------------------------------------

encoder = LabelEncoder()

encoded_labels = encoder.fit_transform(df["Label"])

features = df.drop(columns=["Label"])

# -------------------------------------
# Feature-Label Correlation
# -------------------------------------
from scipy.stats import pearsonr

correlation_scores = {}

for col in features.columns:
    corr, p = pearsonr(features[col], encoded_labels)
    correlation_scores[col] = abs(corr)
    

correlation_df = pd.DataFrame({

    "Feature": correlation_scores.keys(),

    "PCC": correlation_scores.values()

})

correlation_df = correlation_df.sort_values(

    by="PCC",

    ascending=False

)

print(correlation_df)

# Create output directory
OUTPUT_DIR = "/home/dhanush2026/dhanush2026/AI_IDS/PCC_RESULTS"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save feature ranking
correlation_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_ranking.csv"),
    index=False
)

print("\nFeature ranking saved successfully!")
# -----------------------------
# Iterate through PCC thresholds
# -----------------------------

#for threshold in thresholds:

 #   print("=" * 60)
 #   print(f"Processing Threshold : {threshold:.2f}")

    # Create output folder
  #  threshold_folder = os.path.join(
   #     OUTPUT_DIR,
    #    f"threshold_{threshold:.2f}"
    #)

    #os.makedirs(threshold_folder, exist_ok=True)

    #print(f"Output Folder : {threshold_folder}")
