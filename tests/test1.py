from ftnirsml.code import saveModelWithMetadata,loadModelWithMetadata,TrainingModeWithHyperband,TrainingModeWithoutHyperband,InferenceMode,TrainingModeFinetuning
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

    #filepath='./Data/NWFSC_data_sample_trunc.csv'
    filepath = './Data/AFSC_data_sample_trunc.csv'
    data = pd.read_csv(filepath)

    training_outputs_hyperband, additional_outputs_hyperband = TrainingModeWithHyperband(
        data=data,
        filter_CHOICE='savgol',
        scaling_CHOICE='minmax'
    )

    training_outputs_manual, additional_outputs_manual = TrainingModeWithoutHyperband(
        data=data,
        filter_CHOICE='savgol',
        scaling_CHOICE='minmax',
        model_parameters=[2, 101, 51, 0.1, False, 50, 256, 0.1]
    )

    model = training_outputs_manual['trained_model']

    prediction1 = InferenceMode(model, data.loc[1:5], scaler_y=training_outputs_manual['scaler_y'], scaler_x=training_outputs_manual['scaler_x'])

    model.save("./Models/my_model.keras")

    metadata = training_outputs_manual #use the training outputs as the metadata, as they contain many required fields.
    metadata["description"] = 'Very cool model, the concept came to me in a dream last night.'

    model_w_metadata_path = "./Models/my_model_with_metadata.keras.zip"

    #use packModelWithMetadata for non-disk option
    saveModelWithMetadata(model,model_w_metadata_path, metadata=metadata)
    model, metadata = loadModelWithMetadata(model_w_metadata_path)

    #try with a model trainined 1 previous time
    prediction2 = InferenceMode(model, data.loc[1:5], scaler_y=metadata['scaler_y'], scaler_x=metadata['scaler_x']) #should be the same as 1
    assert all(prediction1 == prediction2)

    training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(model=model,data=data,previous_metadata = metadata,
                           filter_CHOICE='savgol',
                           scaling_CHOICE='maxabs',
                           seed_value=42)

    new_metadata = training_outputs_finetuning
    new_metadata["description"] = 'Improve upon the last model. Used maxabs for scaling'

    model_w_metadata_path = "./Models/my_model_with_metadata_2nd_train.keras.zip"
    saveModelWithMetadata(model,model_w_metadata_path, metadata=new_metadata,previous_metadata=metadata)

    #try with a model trainined 2 previous times
    model, metadata = loadModelWithMetadata(model_w_metadata_path)

    training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(model=model, data=data,
                                                                                        previous_metadata=metadata,
                                                                                        filter_CHOICE='moving_average',
                                                                                        scaling_CHOICE='minmax',
                                                                                        seed_value=42)

    new_metadata = training_outputs_finetuning
    new_metadata["description"] = 'Improve upon the last AGAIN. Used minmax for scaling, moving average for filter choice'

    #if you don't want to keep previous model in metadata, can delete it...
    del new_metadata['model']

    model_w_metadata_path = "./Models/my_model_with_metadata_3rd_train.keras.zip"
    saveModelWithMetadata(model, model_w_metadata_path, metadata=new_metadata, previous_metadata=metadata)

    model, metadata = loadModelWithMetadata(model_w_metadata_path)

    print("last metadata:")
    print(metadata[-1])

    print("previous metadata:")
    print(metadata[-2])

    print("original metadata:")
    print(metadata[-3])

    print('test complete!')

if __name__ == "__main__":
    main()