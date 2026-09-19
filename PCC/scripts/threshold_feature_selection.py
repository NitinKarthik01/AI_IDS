import os
import pandas as pd

# ==========================================================
# Load Feature Ranking
# ==========================================================

RANKING_FILE = "/home/dhanush2026/dhanush2026/AI_IDS/PCC_RESULTS/feature_ranking.csv"

# Original cleaned dataset
DATASET_FILE = "/home/dhanush2026/dhanush2026/AI_IDS/DATASETS/MERGED_CSV_Sampled_5Percent.csv"

# Output directory
OUTPUT_DIR = "/home/dhanush2026/dhanush2026/AI_IDS/PCC_RESULTS"

# ==========================================================
# Thresholds (same as PCC-CNN paper)
# ==========================================================

thresholds = [
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

# ==========================================================
# Load Files
# ==========================================================

print("Loading Dataset...")

df = pd.read_csv(DATASET_FILE)

print("Loading Feature Ranking...")

ranking = pd.read_csv(RANKING_FILE)

summary = []

# ==========================================================
# Threshold Loop
# ==========================================================

for threshold in thresholds:

    print("=" * 60)
    print(f"Threshold : {threshold}")
    print("=" * 60)

    selected_features = ranking[
        ranking["PCC"] >= threshold
    ]["Feature"].tolist()

    removed_features = ranking[
        ranking["PCC"] < threshold
    ]["Feature"].tolist()

    # Add Label column
    reduced_df = df[selected_features + ["Label"]]

    # Create Folder

    folder = os.path.join(
        OUTPUT_DIR,
        f"threshold_{threshold:.2f}"
    )

    os.makedirs(folder, exist_ok=True)

    # ---------------------------------------------
    # Save Reduced Dataset
    # ---------------------------------------------

    reduced_df.to_csv(
        os.path.join(folder, "reduced_dataset.csv"),
        index=False
    )

    # ---------------------------------------------
    # Save Selected Features
    # ---------------------------------------------

    with open(
        os.path.join(folder, "selected_features.txt"),
        "w"
    ) as f:

        for feature in selected_features:
            f.write(feature + "\n")

    # ---------------------------------------------
    # Save Removed Features
    # ---------------------------------------------

    with open(
        os.path.join(folder, "removed_features.txt"),
        "w"
    ) as f:

        for feature in removed_features:
            f.write(feature + "\n")

    # ---------------------------------------------
    # Store Summary
    # ---------------------------------------------

    summary.append({

        "Threshold": threshold,

        "Selected Features": len(selected_features),

        "Removed Features": len(removed_features)

    })

    print(f"Selected Features : {len(selected_features)}")
    print(f"Removed Features  : {len(removed_features)}")

# ==========================================================
# Save Summary
# ==========================================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "threshold_summary.csv"
    ),

    index=False

)

print("\nDone!")
print("Threshold feature selection completed.")
