from ftnirsml.code import saveModelWithMetadata,loadModelWithMetadata,TrainingModeWithHyperband,TrainingModeWithoutHyperband,InferenceMode,TrainingModeFinetuning,format_data
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
from copy import deepcopy
from tensorflow.keras.models import load_model

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="whitegrid", font_scale=2)

def main():
    print("running!")

    #filepath1='./Data/NWFSC_data_sample_trunc.csv'
    #data1 = pd.read_csv(filepath1)
    filepath2 = './Data/AFSC_data_sample_trunc.csv'
    data2 = pd.read_csv(filepath2)
    #filepath3 = './Data/SEFSC_data_sample_trunc.csv'
    #data3 = pd.read_csv(filepath3)
    #data = [data1,data2,data3]

    data = pd.read_csv(filepath2)

    #filepath = './Data/SEFSC_data_sample_trunc.csv'
    #data = pd.read_csv(filepath)

    #artifacts: scalers and metadata for the scalers.
    #og_data_info: identifying data back to the original dataset, like dataset hashes
    formatted_data,format_metadata,og_data_info = format_data(data,filter_CHOICE='savgol',scaler='minmax',splitvec=[40,70])

    training_outputs_hyperband, additional_outputs_hyperband = TrainingModeWithHyperband(
        data=formatted_data,
        bio_idx = format_metadata["datatype_indices"]["bio_indices"],
        wn_idx = format_metadata["datatype_indices"]["wn_indices"],
        extra_bio_columns=5,
        max_epochs = 1
    )

    training_outputs_manual, additional_outputs_manual = TrainingModeWithoutHyperband(
        data=formatted_data,
        bio_idx = format_metadata["datatype_indices"]["bio_indices"],
        wn_idx = format_metadata["datatype_indices"]["wn_indices"],
        total_bio_columns=100
    )

    model = training_outputs_manual['trained_model']

    metadata = training_outputs_manual

    metadata.update(format_metadata) #use a combination of the training outputs and the format data metadata as the metadata
    metadata["description"] = 'Very cool model, the concept came to me in a dream last night.'

    #reload the data, and format with existing scalers.
    formatted_data2,_,_ = format_data(data, filter_CHOICE=metadata['filter'], scaler=metadata['scaler'],splitvec=[0, 0])

    prediction1 = InferenceMode(model, formatted_data2.loc[1:5], metadata['scaler'],metadata['names_ordered'])

    test_data = deepcopy(data)

    #see what happens when we drop a columm (this one used in AFSC data sample, could make the test more generic
    test_data.drop("gear_depth",axis=1,inplace=True)

    formatted_data3,_,_ = format_data(test_data, filter_CHOICE=metadata['filter'], scaler=metadata['scaler'],
                                  splitvec=[0, 0])

    prediction1_drop = InferenceMode(model, formatted_data3.loc[1:5], metadata['scaler'])

    #see what happens when I load in another dataset entirely, format it, and plug it into inference.

    filepath1 = './Data/NWFSC_data_sample_trunc.csv'
    data1 = pd.read_csv(filepath1)

    #not sure if the values (bad) represent an incorrect approach or natural poor performance.
    formatted_data1, _, _ = format_data(data1, filter_CHOICE=metadata['filter'], scaler=metadata['scaler'], splitvec=[0, 0])
    prediction1_alt = InferenceMode(model, formatted_data1.loc[1:5], metadata['scaler'],metadata['names_ordered'])

    model.save("./Models/my_model.keras")

    model_w_metadata_path = "./Models/my_model_with_metadata.keras.zip"

    #use packModelWithMetadata for non-disk option
    saveModelWithMetadata(model,model_w_metadata_path, metadata=metadata)
    model, metadata = loadModelWithMetadata(model_w_metadata_path)

    #try with a model trainined 1 previous time
    prediction2 = InferenceMode(model,formatted_data.loc[1:5], metadata['scaler'],metadata['names_ordered']) #should be the same as 1

    #assert all(prediction1 == prediction2)

    #todo: to fine tune, need to bring in an existing scaler and provide to a specific format_data call to modify it. use the exported
    #scaler in the finetune function.
    formatted_data1, outputs, _ = format_data(data1, filter_CHOICE=metadata['filter'], scaler=metadata['scaler'],
                                        splitvec=[0, 0],add_scale=True)

    training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(model=model,data=formatted_data,previous_metadata = metadata,
                           filter_CHOICE='savgol',
                           scaling_CHOICE='maxabs',
                           metadata['names_ordered'],
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