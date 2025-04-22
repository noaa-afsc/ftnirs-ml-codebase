from ftnirsml.code import saveModelWithMetadata,loadModelWithMetadata,TrainingModeWithHyperband,TrainingModeWithoutHyperband,InferenceMode,TrainingModeFinetuning,format_data,plot_training_history,plot_residuals_heatmap
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
from copy import deepcopy
from tensorflow.keras.callbacks import Callback as tk_callbacks
from tensorflow.keras.callbacks import EarlyStopping

TOTAL_EPOCH = 0
class CustomCallback(tk_callbacks):

    def on_epoch_begin(self,epoch, logs=None):
        global TOTAL_EPOCH
        TOTAL_EPOCH=TOTAL_EPOCH+1
        print("TOTAL EPOCH:"+str(TOTAL_EPOCH))

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="whitegrid", font_scale=2)

def main():
    print("running!")

    earlystop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)

    #train a model with one hot, and then try to apply it back on original dataset.

    filepath = './Data/AFSC_2017_pollock_20perc_truth.csv'
    data = pd.read_csv(filepath)

    formatted_data, format_metadata, og_data_info = format_data(data, filter_CHOICE='savgol',scaler='MinMaxScaler',bio_scaler='MinMaxScaler',wn_scaler='MinMaxScaler',response_scaler='MinMaxScaler',
                                                                splitvec=[40, 70], interp_minmaxstep=[3952.0, 8000.0, 8.0])

    model, training_outputs_manual, additional_outputs_manual = TrainingModeWithoutHyperband(
        data=formatted_data,
        scaler=format_metadata['scaler'],
        bio_idx=format_metadata["datatype_indices"]["bio_indices"],
        wn_idx=format_metadata["datatype_indices"]["wn_indices"],
        total_bio_columns=100,
        callbacks=[CustomCallback(), earlystop]
    )



    #does model train without error? Seems to.

    filepath1='./Data/NWFSC_data_sample_trunc.csv'
    data1 = pd.read_csv(filepath1)

    #load in a model output from the app to test if it is interoperable
    model_from_app, metadata_ = loadModelWithMetadata("./Models/unnamed.keras.zip")

    #attempt to run a model trained using one hot expansion (aware of already expanded columns) back on the original data (non expanded columns)
    #filepath1 = './Data/NWFSC_data_sample_trunc.csv'
    #different_data = pd.read_csv(filepath1)
    formatted_data1, outputs, _ = format_data(data1, filter_CHOICE=metadata_['filter'],
                                              scaler=metadata_['scaler'], splitvec=[0, 0])

    test_pred,statz = InferenceMode(model_from_app, formatted_data1.loc[1:5], metadata_['scaler'],metadata_['model_col_names'])

    data1.loc[1, "sex"] =  pd.NA
    #data1 = data1.fillna(pd.NA)

    formatted_data, format_metadata, og_data_info = format_data(data1, filter_CHOICE='savgol',scaler='MinMaxScaler',bio_scaler='MinMaxScaler',wn_scaler='MinMaxScaler',response_scaler='MinMaxScaler',
                                                                splitvec=[40, 70], interp_minmaxstep=[3952.0, 8000.0, 8.0])

    model, training_outputs_manual, additional_outputs_manual = TrainingModeWithoutHyperband(
        data=formatted_data,
        scaler=format_metadata['scaler'],
        bio_idx=format_metadata["datatype_indices"]["bio_indices"],
        wn_idx=format_metadata["datatype_indices"]["wn_indices"],
        total_bio_columns=100,
        callbacks=[CustomCallback(), earlystop]
    )

    #import code
    #code.interact(local=dict(globals(), **locals()))


    data1 = pd.read_csv(filepath1)
    data1.loc[1, "sex"] = pd.NA
    #data1 = data1.fillna(pd.NA)

    formatted_data, format_metadata, og_data_info = format_data(data1, filter_CHOICE=format_metadata['filter'], scaler=format_metadata['scaler'],splitvec=[0, 0])

    #import code
    #code.interact(local=dict(globals(), **locals()))

    prediction,_ = InferenceMode(model, formatted_data.loc[1:5], format_metadata['scaler'],training_outputs_manual['model_col_names'])

    #test- take a dataset without ages and run it through inference.

    data1 = pd.read_csv(filepath1)
    data1 = data1.drop('age', axis=1)

    formatted_data_no_age, format_metadata_no_age, og_data_info = format_data(data1, filter_CHOICE=format_metadata['filter'],
                                                                scaler=format_metadata['scaler'], splitvec=[0, 0])




    prediction,stats = InferenceMode(model, formatted_data_no_age.loc[1:5], format_metadata_no_age['scaler'],training_outputs_manual['model_col_names'])

    print(stats)

    #filepath1='./Data/NWFSC_data_sample_trunc.csv'
    #data1 = pd.read_csv(filepath1)
    filepath2 = './Data/AFSC_data_sample_trunc.csv'
    data2 = pd.read_csv(filepath2)

    formatted_data2, format_metadata2, og_data_info = format_data(data2, filter_CHOICE=format_metadata['filter'],
                                                                scaler=format_metadata['scaler'], splitvec=[0, 0])

    prediction, _ = InferenceMode(model, formatted_data2.loc[1:5], format_metadata2['scaler'],training_outputs_manual['model_col_names'])

    #train model on two datasets, then apply back to original individual as well as combined. Ensure they match.
    data1 = pd.read_csv(filepath1)
    formatted_data_comb, comb_meta, og_data_info = format_data([data1,data2],filter_CHOICE='savgol',scaler='MinMaxScaler',splitvec=[40,70])

    model_comb,training_outputs_manual, additional_outputs_manual = TrainingModeWithoutHyperband(
        data=formatted_data_comb,
        scaler=comb_meta['scaler'],
        bio_idx = comb_meta["datatype_indices"]["bio_indices"],
        wn_idx = comb_meta["datatype_indices"]["wn_indices"],
        total_bio_columns=100,
        callbacks=[CustomCallback(),earlystop]
    )

    #apply it back to both of the datasets

    single_dataset, single_meta, og_data_info = format_data(data2, filter_CHOICE=comb_meta['filter'],
                                                                  scaler=comb_meta['scaler'], splitvec=[0, 0])


    #filepath3 = './Data/SEFSC_data_sample_trunc.csv'
    #data3 = pd.read_csv(filepath3)
    #data = [data1,data2,data3]

    data = pd.read_csv(filepath2)

    #filepath = './Data/SEFSC_data_sample_trunc.csv'
    #data = pd.read_csv(filepath)

    #artifacts: scalers and metadata for the scalers.
    #og_data_info: identifying data back to the original dataset, like dataset hashes
    formatted_data,format_metadata,og_data_info = format_data(data,filter_CHOICE='savgol',scaler='MinMaxScaler',splitvec=[40,70])

    model1,training_outputs_hyperband, additional_outputs_hyperband = TrainingModeWithHyperband(
        data=formatted_data,
        scaler=format_metadata['scaler'],
        bio_idx = format_metadata["datatype_indices"]["bio_indices"],
        wn_idx = format_metadata["datatype_indices"]["wn_indices"],
        total_bio_columns=100,
        #extra_bio_columns=5,
        max_epochs = 1, #,
        callbacks=[CustomCallback()]
    )



    model2,training_outputs_manual, additional_outputs_manual = TrainingModeWithoutHyperband(
        data=formatted_data,
        scaler=format_metadata['scaler'],
        bio_idx = format_metadata["datatype_indices"]["bio_indices"],
        wn_idx = format_metadata["datatype_indices"]["wn_indices"],
        total_bio_columns=100,
        callbacks=[CustomCallback(),earlystop]
    )

    metadata = training_outputs_manual

    metadata.update(format_metadata) #use a combination of the training outputs and the format data metadata as the metadata
    metadata["description"] = 'Very cool model, the concept came to me in a dream last night.'

    #test out inference on the same dataset as training

    #import code
    #code.interact(local=dict(globals(), **locals()))

    prediction1,stats = InferenceMode(model2,formatted_data.loc[1:5], metadata['scaler'],metadata['model_col_names']) #should be the same as 1
    print(stats)
    print(prediction1)

    #test out inference on the same dataset as training, but this time using the existing scaler.
    formatted_data1, _, _ = format_data(data, filter_CHOICE=metadata['filter'], scaler=metadata['scaler'],splitvec=[0, 0])
    prediction2,stats = InferenceMode(model2, formatted_data1.loc[1:5], metadata['scaler'],metadata['model_col_names'])
    print(stats)
    print(prediction2)

    #should be the same or very close.
    print('is each prediction the same:')
    all(prediction1 == prediction2)

    #see what happens when we drop a columm.
    test_data = deepcopy(data)
    test_data.drop("gear_depth",axis=1,inplace=True)
    formatted_data2,_,_ = format_data(test_data, filter_CHOICE=metadata['filter'], scaler=metadata['scaler'],splitvec=[0, 0])

    #import code
    #code.interact(local=dict(globals(), **locals()))

    prediction1_drop,_ = InferenceMode(model2, formatted_data2.loc[1:5], metadata['scaler'],metadata['model_col_names'])
    print(prediction1_drop)

    #see what happens when I load in another dataset entirely, format it, and plug it into inference.
    filepath1 = './Data/NWFSC_data_sample_trunc.csv'
    different_data = pd.read_csv(filepath1)

    #not sure if the values (bad) represent an incorrect approach or natural poor performance.
    formatted_diff_data, _, _ = format_data(different_data, filter_CHOICE=metadata['filter'], scaler=metadata['scaler'], splitvec=[0, 0])
    prediction1_alt,_ = InferenceMode(model2, formatted_diff_data.loc[1:5], metadata['scaler'],metadata['model_col_names'])

    print(prediction1_alt)

    model2.save("./Models/my_model.keras")

    model_w_metadata_path = "./Models/my_model_with_metadata.keras.zip"

    #use packModelWithMetadata for non-disk option
    saveModelWithMetadata(model2,model_w_metadata_path, metadata=metadata)
    model2, metadata = loadModelWithMetadata(model_w_metadata_path)

    model3, training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(
        model=model2,
        scaler=metadata['scaler'],
        data=formatted_data,
        bio_idx = metadata["datatype_indices"]["bio_indices"],
        names_ordered=metadata['model_col_names'],
        seed_value=42)

    new_metadata = training_outputs_finetuning
    new_metadata["description"] = 'Train the original model again using the same data in a second round.'

    new_metadata.update(format_metadata)

    model_w_metadata_path = "./Models/my_model_with_metadata_2nd_train.keras.zip"
    saveModelWithMetadata(model3,model_w_metadata_path, metadata=new_metadata,previous_metadata=metadata)

    #try with a model trainined 2 previous times
    model3, metadata2 = loadModelWithMetadata(model_w_metadata_path)

    #create and transform with a scaler that encompasses the different data
    formatted_data1, outputs, _ = format_data(different_data, filter_CHOICE=metadata2[-1]['filter'], scaler=metadata2[-1]['scaler'],splitvec=[40, 70],add_scale=True)

    #bio idx and names ordered both needed because bio index describes latest ds, names_ordered describes previous. A little clunky.
    model4,training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(
        model=model3,
        scaler=metadata2[-1]['scaler'],
        data=formatted_data1,
        bio_idx = outputs["datatype_indices"]["bio_indices"],
        names_ordered=metadata2[-1]['model_col_names'],
        seed_value=42)

    new_metadata = training_outputs_finetuning
    new_metadata["description"] = 'retrain last model (trained on AFSC data) using new data from NWFSC'

    new_metadata.update(outputs)

    model_w_metadata_path = "./Models/my_model_with_metadata_3rd_train.keras.zip"
    saveModelWithMetadata(model4, model_w_metadata_path, metadata=new_metadata, previous_metadata=metadata2)

    model, metadata3 = loadModelWithMetadata(model_w_metadata_path)

    #attempt to run a model trained using one hot expansion (aware of already expanded columns) back on the original data (non expanded columns)
    #filepath1 = './Data/NWFSC_data_sample_trunc.csv'
    #different_data = pd.read_csv(filepath1)
    formatted_data1, outputs, _ = format_data(different_data, filter_CHOICE=metadata3[-1]['filter'],
                                              scaler=metadata3[-1]['scaler'], splitvec=[0, 0])

    prediction1_alt,_ = InferenceMode(model, formatted_data1.loc[1:5], metadata3[-1]['scaler'],metadata3[-1]['model_col_names'])

    #also, attempt to run back on afsc (original) training data

    formatted_data1_, outputs_, _ = format_data(data, filter_CHOICE=metadata3[-1]['filter'],
                                              scaler=metadata3[-1]['scaler'], splitvec=[0, 0])

    prediction1_alt_,_ = InferenceMode(model, formatted_data1_, metadata3[-1]['scaler'],metadata3[-1]['model_col_names'])

    #test condition: remove a category for a one-hot encoded variable to see if it is handled correctly.

    #first, lets check that this data doesn't have side effects...


    different_data['sex'] = different_data['sex'].replace('M', 'I') #swap out a category to simulate both removing and adding.

    formatted_data1, outputs, _ = format_data(different_data, filter_CHOICE=metadata3[-1]['filter'],
                                              scaler=metadata3[-1]['scaler'], splitvec=[39, 61],add_scale=True)

    #bio idx and names ordered both needed because bio index describes latest ds, names_ordered describes previous. A little clunky.
    model5,training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(
        model=model,
        scaler=metadata3[-1]['scaler'],
        data=formatted_data1,
        bio_idx = outputs["datatype_indices"]["bio_indices"],
        names_ordered=metadata3[-1]['model_col_names'],
        seed_value=42)



    #plot_training_history(training_outputs_finetuning['training_history'])

    print("last metadata:")
    print(metadata3[-1])

    print("previous metadata:")
    print(metadata3[-2])

    print("original metadata:")
    print(metadata3[-3])



    #test: fully drop a bio column- does it get accounted for as inactive in metadata cols_active?
    data1 = pd.read_csv(filepath1)
    data1 = data1.drop(columns=['gear_depth'])

    formatted_data, format_metadata, og_data_info = format_data(data1, filter_CHOICE=metadata_['filter'],
                                              scaler=metadata_['scaler'], splitvec=[0, 0])

    #test: fully drop a ohc colum

    data1 = pd.read_csv(filepath1)
    data1 = data1.drop(columns=['sex'])

    formatted_data, format_metadata, og_data_info = format_data(data1, filter_CHOICE=metadata_['filter'],
                                                                scaler=metadata_['scaler'], splitvec=[0, 0])

    #test: fully drop a two columns

    data1 = pd.read_csv(filepath1)
    data1 = data1.drop(columns=['sex','gear_depth'])

    formatted_data, format_metadata, og_data_info = format_data(data1, filter_CHOICE=metadata_['filter'],
                                                                scaler=metadata_['scaler'], splitvec=[0, 0])


    print('test complete!')

if __name__ == "__main__":
    main()