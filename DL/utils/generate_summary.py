import pandas as pd
import os

RESULT_SAVE_PATH = "/home/dhanush2026/dhanush2026/AI_IDS/DL/RESULTS/CNN/threshold_0.10/"

results_df = pd.read_csv(
    os.path.join(
        RESULT_SAVE_PATH,
        "10Fold_Results.csv"
    )
)

summary = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Sensitivity",
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
        results_df["F1"].mean(),
        results_df["Macro F1"].mean(),
        results_df["AUC"].mean(),
        results_df["Specificity"].mean()

    ],

    "Std": [

        results_df["Accuracy"].std(),
        results_df["Balanced Accuracy"].std(),
        results_df["Precision"].std(),
        results_df["Recall"].std(),
        results_df["F1"].std(),
        results_df["Macro F1"].std(),
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

print(summary)

print("\nexperiment_summary.csv created successfully!")
