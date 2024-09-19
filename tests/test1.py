from ftnirsml.code import read_and_clean_data,saveModelWithMetadata,loadModelWithMetadata,TrainingModeWithHyperband,TrainingModeWithoutHyperband,InferenceMode,TrainingModeFinetuning
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import load_model

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="whitegrid", font_scale=2)

def main():
    print("running!")

    filepath='./Data/sample_data.csv'
    data = pd.read_csv(filepath)
    raw_data = read_and_clean_data(data)
    training_outputs_hyperband, additional_outputs_hyperband = TrainingModeWithHyperband(
        raw_data=raw_data,
        filter_CHOICE='savgol',
        scaling_CHOICE='minmax'
    )

    training_outputs_manual, additional_outputs_manual = TrainingModeWithoutHyperband(
        raw_data=raw_data,
        filter_CHOICE='savgol',
        scaling_CHOICE='minmax',
        model_parameters=[2, 101, 51, 0.1, False, 50, 256, 0.1]
    )

    model = training_outputs_manual['trained_model']

    model.save("./Models/my_model.keras")

    prediction = InferenceMode(model, raw_data, scaler_y=training_outputs_manual['scaler_y'], scaler_x=training_outputs_manual['scaler_x'])

    column_names = ['col1', 'col2', 'col3']
    description = 'This model is trained on data columns col1, col2, col3'

    metadata = {"column_names": column_names, "description":description,
                "training_approach":'a training approach!',"scaler_x":training_outputs_manual['scaler_x'],"scaler_y":training_outputs_manual['scaler_y']}

    model_w_metadata_path = "./Models/my_model_with_metadata.keras.zip"

    #use modelToZipIO for non-disk option
    saveModelWithMetadata(model,model_w_metadata_path, metadata=metadata)
    model, metadata = loadModelWithMetadata(model_w_metadata_path)

    training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(model=model,raw_data=raw_data,
                           filter_CHOICE='savgol',
                           scaling_CHOICE='minmax',
                           seed_value=42)

    print('test complete!')

if __name__ == "__main__":
    main()