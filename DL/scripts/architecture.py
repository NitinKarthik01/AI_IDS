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

        output_shape = tuple(layer.output.shape)

        params = layer.count_params()

        percent = (params / total_params) * 100

        # ----------------------------------------
        # Kernel & Stride
        # ----------------------------------------

        if isinstance(layer, Conv1D):

            kernel = layer.kernel_size[0]

            stride = layer.strides[0]

            # Placeholder for now
            macs = "-"

        elif isinstance(layer, Dense):

            kernel = "-"

            stride = "-"

            # Placeholder for now
            macs = "-"

        else:

            kernel = "-"

            stride = "-"

            macs = "-"

        rows.append({

            "Stage": stage,

            "Module": module,

            "Output Shape": output_shape,

            "Kernel": kernel,

            "Stride": stride,

            "Parameters": params,

            "% Total": round(percent, 2),

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
    
    print(df)
