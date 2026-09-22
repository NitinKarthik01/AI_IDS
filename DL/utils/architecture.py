import os
import pandas as pd

from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    InputLayer
)


def save_cnn_architecture(model, save_path):

    total_params = model.count_params()

    rows = []

    stage = 0

    for layer in model.layers:

        module = layer.__class__.__name__

        # ======================================================
        # Output Shape
        # ======================================================

        shape = layer.output.shape

        if len(shape) == 3:

            output_shape = f"{shape[1]} × {shape[2]}"

        elif len(shape) == 2:

            output_shape = f"{shape[1]}"

        else:

            output_shape = "-"

        # ======================================================
        # Parameters
        # ======================================================

        params = layer.count_params()

        percent = round((params / total_params) * 100, 2)

        # ======================================================
        # Activation
        # ======================================================

        if hasattr(layer, "activation"):

            activation = layer.activation.__name__

        else:

            activation = "-"

        # ======================================================
        # Conv1D Information
        # ======================================================

        if isinstance(layer, Conv1D):

            kernel = layer.kernel_size[0]

            stride = layer.strides[0]

            padding = layer.padding

            filters = layer.filters

            input_channels = layer.input_shape[-1]

            output_length = layer.output_shape[1]

            macs = (
                output_length
                * filters
                * kernel
                * input_channels
            )

        # ======================================================
        # Dense Information
        # ======================================================

        elif isinstance(layer, Dense):

            kernel = "-"

            stride = "-"

            padding = "-"

            input_units = layer.input_shape[-1]

            output_units = layer.units

            macs = input_units * output_units

        # ======================================================
        # Other Layers
        # ======================================================

        else:

            kernel = "-"

            stride = "-"

            padding = "-"

            macs = "-"

        rows.append({

            "Stage": stage,

            "Module": module,

            "Output Shape": output_shape,

            "Activation": activation,

            "Kernel": kernel,

            "Stride": stride,

            "Padding": padding,

            "Parameters": params,

            "% Total": percent,

            "MACs": macs

        })

        stage += 1

    df = pd.DataFrame(rows)

    df.to_csv(

        os.path.join(
            save_path,
            "cnn_architecture.csv"
        ),

        index=False

    )

    try:

        df.to_excel(

            os.path.join(
                save_path,
                "cnn_architecture.xlsx"
            ),

            index=False

        )

    except:

        pass

    print("\nCNN Architecture Saved Successfully!")
