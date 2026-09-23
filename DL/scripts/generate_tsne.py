import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = "/data/dhanush2026/AI_IDS"

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "PCC_RESULTS",
    "threshold_0.10",
    "reduced_dataset.csv"      # Change if filename differs
)

SAVE_PATH = os.path.join(
    PROJECT_ROOT,
    "DL",
    "RESULTS",
    "CNN",
    "threshold_0.10",
    "REPORT"
)

os.makedirs(SAVE_PATH, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading Dataset...")

df = pd.read_csv(DATA_PATH)

print(df.shape)

# ==========================================================
# SAMPLE FOR FASTER t-SNE
# ==========================================================

df = df.sample(
    n=10000,
    random_state=42
)

print("SAMPLED TO 10000 :", df.shape)

# ==========================================================
# FEATURES AND LABELS
# ==========================================================

X = df.drop(columns=["Label"])

y = df["Label"]

# ----------------------------------------------------------
# CONVERT TO BINARY LABELS
# ----------------------------------------------------------

# Everything except Benign is considered Attack

y = y.apply(
    lambda x: 0 if str(x).lower() == "benign" else 1
)

# ==========================================================
# FEATURE SCALING
# ==========================================================

print("Scaling Features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# t-SNE
# ==========================================================

print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

print("Completed.")

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=y,
    cmap="coolwarm",
    s=6,
    alpha=0.7
)

cbar = plt.colorbar(scatter)

cbar.set_ticks([0,1])

cbar.set_ticklabels([
    "Benign",
    "Attack"
])

plt.title("t-SNE Visualization of Binary Intrusion Dataset")

plt.xlabel("t-SNE Component 1")

plt.ylabel("t-SNE Component 2")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_PATH,
        "tSNE.png"
    ),
    dpi=300
)

plt.close()

print("="*60)
print("t-SNE Figure Saved Successfully")
print("="*60)
