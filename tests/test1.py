from ftnirsml.code import read_and_clean_data,TrainingModeWithHyperband,TrainingModeWithoutHyperband,saveModel,saveModelWithMetadata,InferenceMode,TrainingModeFinetuning,readModelMetadata
import tensorflow as tf
import numpy as np
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="whitegrid", font_scale=2)

def main():
    print("running!")

    filepath='./Data/sample_data.csv'
    raw_data = read_and_clean_data(filepath)
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

    saveModel(model, "./Models/my_model.h5")

    prediction = InferenceMode(model, raw_data, row_number=5, scaler_y=training_outputs_manual['scaler_y'], scaler_x=training_outputs_manual['scaler_x'])
    print(f"Prediction for row {5}: {prediction}")

    column_names = ['col1', 'col2', 'col3']
    description = 'This model is trained on data columns col1, col2, col3'
    saveModelWithMetadata(path="my_model_with_metadata.h5", 
                          model=model, 
                          old_metadata_path="my_model_with_metadata.h5",
                          column_names=column_names, 
                          description=description,
                          training_approach='a training approach!',
                          scaler_x=training_outputs_manual['scaler_x'],
                          scaler_y=training_outputs_manual['scaler_y']
                          )

    metadata_path = './Models/my_model_with_metadata.h5'
    column_names, description, training_approach, scaler_x, scaler_y = readModelMetadata(metadata_path)
 
    training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(raw_data=raw_data, 
                           filter_CHOICE='savgol',
                           scaling_CHOICE='minmax', 
                           file_path='my_model_with_metadata.h5',
                           seed_value=42)

    print('test complete!')


if __name__ == "__main__":
    main()