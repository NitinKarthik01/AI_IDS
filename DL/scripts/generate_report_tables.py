import os
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# CHANGE ONLY THESE
# ======================================================

PROJECT_ROOT = r"C:\Dhanush\D\MSIS\Mini Project\DL"

MODEL_NAME = "ANN"

# PROJECT_ROOT = r"C:\Dhanush\D\MSIS\Mini Project\ML"

# MODEL_NAME = "RandomForest"

THRESHOLD = "0.10"

# ======================================================

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "RESULTS",
    MODEL_NAME,
    f"threshold_{THRESHOLD}"
)

REPORT_PATH = os.path.join(
    RESULT_PATH,
    "REPORT"
)

os.makedirs(REPORT_PATH, exist_ok=True)

# ======================================================

if MODEL_NAME == "CNN":
    arch_file = "cnn_architecture.csv"
else:
    arch_file = "ann_architecture.csv"

FILES = {

    "Hyperparameters":"hyperparameters.csv",

    "Architecture":arch_file,

    "Experiment Summary":"experiment_summary.csv"
    # "Hyperparameters":"hyperparameters.csv",

    # "Experiment Summary":"experiment_summary.csv"

}

# ======================================================

def save_table(csv_file, title, output):

    df = pd.read_csv(csv_file)

    fig_height = max(2.5, len(df)*0.45)

    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.axis("off")

    table = ax.table(

        cellText=df.values,

        colLabels=df.columns,

        cellLoc="center",

        loc="center"

    )

    table.auto_set_font_size(False)

    table.set_fontsize(10)

    table.scale(1.25,1.6)

    plt.title(
        title,
        fontsize=15,
        weight="bold",
        pad=15
    )

    plt.savefig(

        os.path.join(
            REPORT_PATH,
            output
        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

# ======================================================

for title,file in FILES.items():

    print(f"Generating {title}...")

    save_table(

        os.path.join(
            RESULT_PATH,
            file
        ),

        title,

        file.replace(".csv",".png")

    )

print()

print("Tables Generated Successfully.")