import os
import pandas as pd

from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    InputLayer
)

# ==========================================================
# SAVE MODEL ARCHITECTURE
# ==========================================================

def save_model_architecture(model, save_path, model_name):

    total_params = model.count_params()

    rows = []

    stage = 1

    for layer in model.layers:

        module = layer.__class__.__name__

        output_shape = tuple(layer.output.shape)

        params = layer.count_params()

        percent = round((params / total_params) * 100, 2)

        # --------------------------------------------------
        # Default Values
        # --------------------------------------------------

        kernel = "-"
        stride = "-"
        activation = "-"
        macs = "-"

        # --------------------------------------------------
        # Conv1D
        # --------------------------------------------------

        if isinstance(layer, Conv1D):

            kernel = layer.kernel_size[0]
            stride = layer.strides[0]
            activation = layer.activation.__name__

        # --------------------------------------------------
        # Dense
        # --------------------------------------------------

        elif isinstance(layer, Dense):

            activation = layer.activation.__name__

        # --------------------------------------------------
        # Batch Normalization
        # --------------------------------------------------

        elif isinstance(layer, BatchNormalization):

            activation = "-"

        # --------------------------------------------------
        # Dropout
        # --------------------------------------------------

        elif isinstance(layer, Dropout):

            activation = "-"
            kernel = "-"
            stride = "-"

        # --------------------------------------------------
        # Flatten
        # --------------------------------------------------

        elif isinstance(layer, Flatten):

            activation = "-"

        rows.append({

            "Stage": stage,
            "Layer": layer.name,
            "Module": module,
            "Output Shape": output_shape,
            "Activation": activation,
            "Kernel": kernel,
            "Stride": stride,
            "Parameters": params,
            "% Total Params": percent,
            "MACs": macs

        })

        stage += 1

    df = pd.DataFrame(rows)

    # ======================================================
    # File Name
    # ======================================================

    file_name = f"{model_name.lower()}_architecture.csv"

    df.to_csv(

        os.path.join(
            save_path,
            file_name
        ),

        index=False

    )

    print("\nArchitecture Saved Successfully.\n")

    print(df)
