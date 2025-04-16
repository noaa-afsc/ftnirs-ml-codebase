import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import pyCompare
import pickle
import base64
import os
import io
import zipfile
import warnings
import hashlib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Normalizer, RobustScaler, MaxAbsScaler, FunctionTransformer
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from keras_tuner.tuners import  Hyperband
from tensorflow.keras.layers import LeakyReLU, Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
from copy import deepcopy
#PyWavelets
import pywt
from sklearn.decomposition import PCA
import sys
from sklearn.model_selection import KFold
import shap
# Use for internal functions which rely on intermediates
import tempfile

from .constants import REQUIRED_METADATA_FIELDS_ORIGINAL,WN_MATCH,INFORMATIONAL,RESPONSE_COLUMNS,SPLITNAME,WN_STRING_NAME,IDNAME,STANDARD_COLUMN_NAMES,MISSING_DATA_VALUE,ONE_HOT_FLAG,MISSING_DATA_VALUE_UNSCALED

# Set seeds for reproducibility 
np.random.seed(42)
tf.random.set_seed(42)

# Seaborn configuration
sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="whitegrid", font_scale=2)

#define a do nothing scaler, for use in one-hot encoding, to allow for columns names to still be registered and handled equally by
#the transformer protocol

# Filter functions
def savgol(data, window_length=17, polyorder=2, deriv=1):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    return savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=deriv)

def moving_average(data, size=5):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    return uniform_filter1d(data, size=size, axis=1)

def gaussian(data, sigma=2):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    return gaussian_filter1d(data, sigma=sigma, axis=1)

def median(data, size=5):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    return median_filter(data, size=(1, size))

def wavelet(data, wavelet='db1', level=1):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    def apply_wavelet(signal):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        coeffs[1:] = [pywt.threshold(i, value=0.5 * max(i)) for i in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)
    return np.apply_along_axis(apply_wavelet, axis=1, arr=data)

def fourier(data, threshold=0.1):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    def apply_fft(signal):
        fft_data = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        fft_data[np.abs(frequencies) > threshold] = 0
        return np.fft.ifft(fft_data).real
    return np.apply_along_axis(apply_fft, axis=1, arr=data)

def pca(data, n_components=5):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    pca_ = PCA(n_components=n_components)
    transformed = pca_.fit_transform(data)
    return pca_.inverse_transform(transformed)

# Scaling functions

def assign_scaler(scaler):
    if isinstance(scaler, str):

        scalers = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'MaxAbsScaler': MaxAbsScaler,
            'RobustScaler': RobustScaler,
            'Normalizer': Normalizer  # Note: Normalizer might not be suitable for y, use with caution
        }

        if scaler not in scalers:
            raise ValueError(f"Unsupported scaling method: {scaler}")

        return scalers[scaler]()

def create_scale(data,bio_scaler,response_scaler=None,wn_scaler=None):

    if wn_scaler is None:
        wn_scaler = bio_scaler

    if response_scaler is None:
        response_scaler = bio_scaler

    #required to not have side effects on the original data object
    data = deepcopy(data)

    if isinstance(bio_scaler, str):
        bio_scaler = assign_scaler(bio_scaler)
    if isinstance(wn_scaler, str):
        wn_scaler = assign_scaler(wn_scaler)
    if isinstance(response_scaler, str):
        response_scaler = assign_scaler(response_scaler)

    #otherwise, assume the object is the correct scaler object

    #inject in the value representing missing data
    #provide missing values to scaler (use one hot flag to assume the position of one hot
    #for the wn columns, which aren't expected to have missing data later, fill in with the median value to not influence scale
    data.loc[len(data)] = [0 if ONE_HOT_FLAG in x else data[x].median() if WN_MATCH in x else MISSING_DATA_VALUE for x in data.columns]

    #if providing data_indeces, assume that only some of the data are getting scaled (such as in first round, when changing data in place.)

    if any([WN_MATCH in m for m in data.columns]):
        _, _, _, _, _, wn_inds = wnExtract(data.columns)

        column_scaler = ColumnTransformer([(x,FunctionTransformer(func=None) if ONE_HOT_FLAG in x else bio_scaler if x not in RESPONSE_COLUMNS else response_scaler,[x]) for x in [data.columns[m] for m in range(min(wn_inds))]]+[(WN_STRING_NAME,wn_scaler,[data.columns[n] for n in wn_inds])])
    else:
        column_scaler = ColumnTransformer([(x,FunctionTransformer(func=None) if ONE_HOT_FLAG in x else bio_scaler if x not in RESPONSE_COLUMNS else response_scaler, [x]) for x in [m for m in data.columns]])

    column_scaler.set_output(transform='pandas')
    column_scaler.fit(data)

    #tranform and rename/reorder to fix column scaler behavior
    data = transform(data,column_scaler)

    #remove the dummy row
    data.drop(data.tail(1).index, inplace=True)

    return data, [column_scaler]

def transform(data,column_scaler):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    data = column_scaler.transform(data)

    #rename and reorder, output of columns scaler doesn't preserve names or order.
    data.columns = [i.split("__",1)[1] for i in data.columns]

    return data



# Model-building functions
def build_model(hp, input_dim_A, input_dim_B):
    input_A = Input(shape=(input_dim_A,))
    x = input_A

    input_B = Input(shape=(input_dim_B, 1))
    
    # Define the hyperparameters
    num_conv_layers = hp.Int('num_conv_layers', 1, 4, default=1)
    kernel_size = hp.Int('kernel_size', 51, 201, step=10, default=101)
    stride_size = hp.Int('stride_size', 26, 101, step=5, default=51)
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.05, default=0.1)
    use_max_pooling = hp.Boolean('use_max_pooling', default=False)
    num_filters = hp.Int('num_filters', 50, 100, step=10, default=50)

    y = input_B
    for i in range(num_conv_layers):
        y = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=stride_size,
            activation='relu',
            padding='same')(y)
        
        # Ensure the input size is appropriate for max pooling
        if use_max_pooling and y.shape[1] > 1:
            y = MaxPooling1D(pool_size=2)(y)
        
        y = Dropout(dropout_rate)(y)

    y = Flatten()(y)
    y = Dense(4, activation="relu", name='output_B')(y)

    con = concatenate([x, y])

    z = Dense(
        hp.Int('dense', 4, 640, step=32, default=256),
        activation='relu')(con)
    z = Dropout(hp.Float('dropout-2', 0.0, 0.5, step=0.05, default=0.0))(z)

    output = Dense(1, activation="linear")(z)
    model = Model(inputs=[input_A, input_B], outputs=output)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mse', 'mae'])
    return model

# Training, evaluation, and plotting functions
def train_and_optimize_model(tuner, data, nb_epoch, batch_size,bio_names_ordered,wn_columns_names_ordered,**kwargs):

    tuner.search([data.loc[data[SPLITNAME] == 'training', bio_names_ordered],
                  data.loc[data[SPLITNAME] == 'training',wn_columns_names_ordered]],
                  data.loc[data[SPLITNAME] == 'training', RESPONSE_COLUMNS],
                 epochs=nb_epoch,
                 batch_size=batch_size,
                 shuffle=True,
                 validation_data=[[data.loc[data[SPLITNAME] == 'validation', bio_names_ordered],
                                  data.loc[data[SPLITNAME] == 'validation',wn_columns_names_ordered]],
                                  data.loc[data[SPLITNAME] == 'validation',RESPONSE_COLUMNS]],
                 verbose=1,
                 **kwargs)

    model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]

    return model, best_hp

def plot_training_history(history):
    plt.figure(figsize=(10, 10))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def evaluate_model(model,scaler, data, bio_names, wn_names,splitnames=["training","validation","test"]):

    print(data.columns)

    bio_and_wn_data_list = [data.loc[:, bio_names], data.loc[:, wn_names]]
    preds_all = model.predict(bio_and_wn_data_list)
    preds_all = scaler[0].named_transformers_[RESPONSE_COLUMNS[0]].inverse_transform(preds_all)

    perf_stats_split = {}

    perf_stats_split['unaged'] = {}
    for i in splitnames:
        perf_stats_split[i] = {}
        perf_stats_split[i]['nrow']=0

    #split specific stats:
    for i in list(data['split'].unique()):

        #make sure any of the rows within the split have valid ages (not NA)
        if any([a in data.columns for a in RESPONSE_COLUMNS]):

            if data.loc[data['split'] == i, RESPONSE_COLUMNS].notna().any()[0]:

                #subset to just the relevant data- correct split and w/ ages.

                datasub = data[(data['split'] == i) & (data[RESPONSE_COLUMNS].notna()['age'])]

                perf_stats_split[i]={}

                #a little inneffecient to reculate preds...

                bio_and_wn_data_list = [datasub.loc[:, bio_names], datasub.loc[:, wn_names]]
                response_data  = datasub.loc[:, RESPONSE_COLUMNS]
                evaluation = model.evaluate(bio_and_wn_data_list, response_data)

                response_data = scaler[0].named_transformers_[RESPONSE_COLUMNS[0]].inverse_transform(response_data)

                preds = model.predict(bio_and_wn_data_list)
                preds = scaler[0].named_transformers_[RESPONSE_COLUMNS[0]].inverse_transform(preds)
                #note: don't expect this to behave properly if multiple response columns are present
                r2 = r2_score(response_data, preds)

                perf_stats_split[i]['r2']=r2
                perf_stats_split[i]['loss'] = evaluation[0]
                perf_stats_split[i]['mse'] = evaluation[1]
                perf_stats_split[i]['mae'] = evaluation[2]
                perf_stats_split[i]['nrow']=len(datasub)

    #get nrow for any present non aged data. since preds calculated for whole thing can subset downstream.
    if any([a in data.columns for a in RESPONSE_COLUMNS]):
        perf_stats_split['unaged']['nrow']=int(data[RESPONSE_COLUMNS].isna().sum())
    else:
        perf_stats_split['unaged']['nrow']=len(data)

    return perf_stats_split, preds_all

def plot_predictions(y_test, preds):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    lims = [min([min(y_test),min(preds)])-0.1, max([max(y_test),max(preds)])+0.1]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()

def plot_prediction_error(preds, y_test):
    preds = np.array(preds).flatten()
    y_test = y_test.to_numpy().flatten()
    error = preds - y_test

    plt.figure(figsize=(6, 6))
    plt.hist(error, bins=20)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.show()

def evaluate_training_set(model, data, scaler_y):

    #required to not have side effects on the original data object
    data = deepcopy(data)

    X_train_biological_data = data.loc[data['sample'] == 'training', data.columns[3:100]]
    X_train_wavenumbers = data.loc[data['sample'] == 'training', data.columns[100:1100]]
    y_train = data.loc[data['sample'] == 'training', 'age']
    f_train = data.loc[data['sample'] == 'training', 'file_name']

    y_train = np.array(y_train).reshape(-1, 1)
    preds_t = model.predict([X_train_biological_data, X_train_wavenumbers])
    
    preds_t = preds_t.reshape(-1, 1)
    y_train_reshaped = y_train.reshape(-1, 1)
    
    y_pr_transformed = scaler_y.inverse_transform(preds_t)
    y_tr_transformed = scaler_y.inverse_transform(y_train_reshaped)

    r_squared_tr = r2_score(y_tr_transformed, y_pr_transformed)
    rmse_tr = sqrt(mean_squared_error(y_tr_transformed, y_pr_transformed))

    y_tr_df = pd.DataFrame(y_tr_transformed, columns=['train'])
    y_tr_df['pred'] = y_pr_transformed
    y_tr_df['file'] = f_train.reset_index(drop=True)

    return r_squared_tr, rmse_tr, y_tr_df

# Plotting functions with no direct file saving 
def training_set_plot(y_tr_transformed, y_pr_transformed):
    sns.set_style("white")
    sns.set(style="ticks")
    sns.set_context("poster")

    f, ax = plt.subplots(figsize=(12, 12))
    p = sns.regplot(x=y_tr_transformed, y=y_pr_transformed, ci=None,
                    scatter_kws={"edgecolor": 'b', 'linewidths': 2, "alpha": 0.5, "s": 150},
                    line_kws={"alpha": 0.5, "lw": 4})
    ax.plot([y_tr_transformed.min(), y_tr_transformed.max()], [y_tr_transformed.min(), y_tr_transformed.max()], 'k--', lw=2)

    p.set(xlim=(-1, 24))
    p.set(ylim=(-1, 24))
    sns.despine()
    plt.title('Training Set', fontsize=25)
    plt.xlabel('Traditional Age (years)')
    plt.ylabel('FT-NIR Age (years)')
    return plt

def test_set_plot(y_test_transformed, y_pred_transformed):
    f, ax = plt.subplots(figsize=(12, 12))
    p = sns.regplot(x=y_test_transformed, y=y_pred_transformed, ci=None,
                    scatter_kws={"edgecolor": 'b', 'linewidths': 2, "alpha": 0.5, "s": 150},
                    line_kws={"alpha": 0.5, "lw": 4})
    ax.plot([y_test_transformed.min(), y_test_transformed.max()], [y_test_transformed.min(), y_test_transformed.max()], 'k--', lw=2)

    p.set(xlim=(-1, 24))
    p.set(ylim=(-1, 24))
    sns.despine()
    plt.title('Test Set', fontsize=25)
    plt.xlabel('Traditional Age (years)')
    plt.ylabel('FT-NIR Age (years)')
    return plt

#ddoesn't do anything the original doesn't
#def bland_altman_plot(y_test_transformed, y_pred_transformed):
#    plt = pyCompare.blandAltman(y_test_transformed, y_pred_transformed,
#                          limitOfAgreement=1.96, confidenceInterval=95,
#                          confidenceIntervalMethod='approximate',
#                          detrend=None, percentage=False,
#                          title='Bland-Altman Plot\n')
#    return plt

# Manual model building for training without hyperband 
def build_model_manual(input_dim_A, input_dim_B, num_conv_layers, kernel_size, stride_size, dropout_rate, use_max_pooling, num_filters, dense_units, dropout_rate_2):
    input_A = Input(shape=(input_dim_A,))
    x = input_A

    input_B = Input(shape=(input_dim_B, 1))
    y = input_B
    for i in range(num_conv_layers):
        y = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=stride_size,
            activation='relu',
            padding='same')(y)
        
        if use_max_pooling and y.shape[1] > 1:
            y = MaxPooling1D(pool_size=2)(y)
        
        y = Dropout(dropout_rate)(y)

    y = Flatten()(y)
    y = Dense(4, activation="relu", name='output_B')(y)

    con = concatenate([x, y])

    z = Dense(dense_units, activation='relu')(con)
    z = Dropout(dropout_rate_2)(z)

    output = Dense(1, activation="linear")(z)
    model = Model(inputs=[input_A, input_B], outputs=output)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mse', 'mae'])
    return model

# Inference function 
def InferenceMode(model, data, scaler,names_ordered):

    padded_data, _ = pad_bio_columns(data, names_ordered['bio_column_names_ordered'],
                                                            total_bio_columns=len(names_ordered['bio_column_names_ordered_padded']))

    #do the normal eval
    stats, preds = evaluate_model(model, scaler, padded_data, names_ordered['bio_column_names_ordered_padded'], names_ordered['wn_columns_names_ordered'])

    return preds,stats

def wnExtract(data_columns):

    #rely on naming convention

    wn_inds = [i for i, x in enumerate(WN_MATCH in c for c in data_columns) if x]

    wn_array = [float(c[len(WN_MATCH):]) for c in data_columns if WN_MATCH in c]

    if(len(wn_array)==0):
        return None, None, None, None, None, None

    wn_min = min(wn_array)
    wn_max = max(wn_array)

    wn_order = "asc" if wn_array[0]<wn_array[1] else 'desc'

    #calculate average step size
    step = (wn_max-wn_min)/(len(wn_array)-1)

    return wn_order, wn_array, wn_min, step, wn_max, wn_inds


def resample_spectroscopy_data(wave_numbers, intensities, target_wave_numbers):

    # Linear interpolation for resampling. TODO: assess what should be user parameters
    interp_func = interp1d(wave_numbers, intensities, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Apply the interpolation function to the target wave numbers
    resampled_intensities = interp_func(target_wave_numbers)

    return resampled_intensities

def getWNSteps(min,max,size):
    return np.arange(min, max + size, size)
def interpolateWN(data,wn_inds, wn, wn_order,desired_min, desired_max,desired_step_size):

    #assumes that only data for wn are being given to this function.
    #tolerant of desc/asc ordering, but needs to be consistent and consecutive.

    data = deepcopy(data)

    data_wn = data.iloc[:, wn_inds]

    target_wave_numbers = getWNSteps(desired_min, desired_max, desired_step_size)

    if wn_order == "desc":
        wn.reverse()

    outwn = [resample_spectroscopy_data(wn, data_wn.iloc[[i]].to_numpy()[:,::-1] if wn_order=="desc" else data_wn.iloc[[i]].to_numpy(), target_wave_numbers) for i in range(data_wn.shape[0])]

    #best, I think, if we just delete the wn columns, then append them to right here. That will be most consistent for later easier assumptions.

    data_no_wn = data.drop(data.columns[wn_inds],axis=1) #delete all previous wn

    out = pd.concat([data_no_wn.reset_index(drop=True),pd.DataFrame(np.vstack(outwn),columns=[f'{WN_MATCH}{i}' for i in target_wave_numbers])],axis=1)

    return out #stick new wn onto the right

#this function will take either a list of datasets or a single dataset, apply interoploation (either automated or predefined)
#and export a single dataset with standard interpolation and all other original columns included.
#requires that the wave number columns are named with the following pattern: [WN_MATCH][wav_num]. The wav numbers must be consecutive.
def standardize_data(data,interp_minmaxstep = None):

    if isinstance(data,list):

        if interp_minmaxstep is None:

            #identify ranges from the source data
            interp_min = []
            interp_max = []
            interp_step_size = []

            for dataset in data:
                # required to not have side effects on the original data object
                dataset = deepcopy(dataset)

                wn_order,wn_array,wn_min,wn_step,wn_max,wn_inds = wnExtract(dataset.columns)

                interp_min.append(wn_min)
                interp_max.append(wn_max)
                interp_step_size.append(wn_step)

            #test for mixed types, indicating assumption error:
            if any(isinstance(x,float) for x in interp_min) and any(x is None for x in interp_min):
                raise ValueError(f"Datasets must either all have {WN_STRING_NAME} or all lack it, mixed datasets provided.")

            interp_min = min(interp_min)
            interp_max = max(interp_max)
            interp_step_size = min(interp_step_size)
        else:
            interp_min,interp_max,interp_step_size = interp_minmaxstep

        #interpolate each dataset:
        standardized_wn_data = []

        for dataset in data:

            wn_order,wn_array,wn_min,wn_step,wn_max,wn_inds = wnExtract(dataset.columns) #mildly inneffecient to call again
            standardized_wn_data.append(interpolateWN(dataset,wn_inds,wn_array,wn_order, interp_min, interp_max,interp_step_size))

        data = pd.concat(standardized_wn_data, axis = 0, join='outer')
        _, _, _, _, _, wn_inds = wnExtract(data.columns)

        #rearrange the final ds to make sure that the wav numbers are stuck on the right
        data = pd.concat([data.drop(data.columns[wn_inds], axis=1),data.iloc[:, wn_inds]],axis=1)
        _, _, _, _, _, wn_inds = wnExtract(data.columns) #get the final inds for the resulting data, which also contains additional biological columns.

    else:

        wn_order, wn_array, wn_min, wn_step_size, wn_max, wn_inds = wnExtract(data.columns)

        if wn_order is not None and interp_minmaxstep is not None:

            interp_min, interp_max, interp_step_size = interp_minmaxstep

            data = interpolateWN(data, wn_inds,wn_array,wn_order, interp_min, interp_max,interp_step_size)

            _, _, _, _, _, wn_inds = wnExtract(data.columns)

        #if wn order is descending, standardize to ascending but don't interpolate. Ensure that wn go on the right of ds.
        elif wn_order == "desc":

            data = data[list(np.delete(data.columns,wn_inds))+list(data.columns[wn_inds[::-1]])]

            interp_min, interp_max, interp_step_size = wn_min, wn_max, wn_step_size

    return data, [interp_min,interp_max,interp_step_size], wn_inds

def autoOneHot(data,expand_nonstandard_str=True,NA_as_one_hot_category=True):

    data = deepcopy(data)

    _, _, _, _, _, wn_inds = wnExtract(data.columns) #just do again to let the fxn be more indepentent

    if wn_inds is None:
        wn_inds_set = {}
    else:
        wn_inds_set = set(wn_inds)

    inf_inds = []
    resp_inds = []
    biological_expanded = []

    for i in (set(range(data.shape[1])) - wn_inds_set):

        if data.columns[i] in INFORMATIONAL:
            inf_inds.append(i)
        elif data.columns[i] in RESPONSE_COLUMNS:
            resp_inds.append(i)
        else:
            #can assume it's biological- see if it is categorical, and if so one-hot expand.
            if data.columns[i] in STANDARD_COLUMN_NAMES:
                if STANDARD_COLUMN_NAMES[data.columns[i]]["data_type"] == "categorical":

                    #fill NA if present
                    if NA_as_one_hot_category:
                        data[data.columns[i]] = data[data.columns[i]].fillna("<NA>")

                    #expand into one hot.
                    biological_expanded.append(pd.get_dummies(data[data.columns[i]],prefix=data.columns[i]+ONE_HOT_FLAG).astype(int))

                else:
                    data[data.columns[i]] = data[data.columns[i]].fillna(MISSING_DATA_VALUE)
                    biological_expanded.append(data[data.columns[i]])

            elif data[data.columns[i]].apply(lambda x: isinstance(x,str)).any():

                #if there are any NA values, replace with text so they can be expanded as their own category (may allow model to better assess effect of missing data, needs testing and possible a parameter
                #if variable performance compared to just leaving NA data out of one-hot.

                if NA_as_one_hot_category:
                    data[data.columns[i]] = data[data.columns[i]].fillna("<NA>")

                # if the column is a string, perform the exansion based on category names.
                if expand_nonstandard_str:
                    warnings.warn(f"{data.columns[i]} was treated as a categorical biological variable (argument 'expand_nonstandard_str' set to true)")
                    biological_expanded.append(pd.get_dummies(data[data.columns[i]],prefix=data.columns[i]+ONE_HOT_FLAG).astype(int))
                else:
                    #maybe want to be an explicit 'warning', leave for later if helpful
                    warnings.warn(f"{data.columns[i]} was eliminated from the dataset (argument 'expand_nonstandard_str' set to false)")

            else:

                #if it's not standard, or if it's not a string, assume it's numeric / integer and thus can interpret a -1 value (caution with this assumption)

                data[data.columns[i]] = data[data.columns[i]].fillna(MISSING_DATA_VALUE)
                biological_expanded.append(data[data.columns[i]])

    bio = pd.concat(biological_expanded,axis = 1)
    bio.reset_index(drop=True, inplace=True)

    #check it's not bigger than max cols, and pad extra cols on.

    #if bio.shape[1] < total_bio_columns:

    #    padding = pd.DataFrame(MISSING_DATA_VALUE_UNSCALED,index=range(bio.shape[0]),columns = [f"_UNDECLARED_{m}" for m in range(bio.shape[1]+1,total_bio_columns+1)])

    #    bio.reset_index(drop=True, inplace=True)
    #    padding.reset_index(drop=True, inplace=True)

    #    bio = pd.concat([bio, padding], axis=1)

    responsedat = data.iloc[:, resp_inds]
    responsedat.reset_index(drop=True, inplace=True)

    infdat = data.iloc[:, inf_inds]
    infdat.reset_index(drop=True, inplace=True)

    wn_dat = data.iloc[:, wn_inds]
    wn_dat.reset_index(drop=True, inplace=True)

    all = pd.concat([responsedat,infdat,bio,wn_dat],axis=1)

    return all, (list(range(0,len(resp_inds))),list(range(len(resp_inds),len(resp_inds+inf_inds))),[len(resp_inds+inf_inds) + m for m in list(range(bio.shape[1]))],[m + (len(resp_inds+inf_inds)+bio.shape[1]) for m in range(len(wn_inds))])
    #disinguish the non-wn and informational columns
    #assess which of those is defined as categorical, expand with one-hot.
    #assert that the resulting total # of columns is < than "total_bio_columns"
    #tack on unnassigned columns
    #rearrange full dataset, in order of informational, biological, wn.
    #export information for their easier later identification by index...?

def hash_dataset(data: pd.DataFrame) -> str:

    data = deepcopy(data)

    # Sort columns and rows to ignore order
    data = data.sort_index(axis=1).sort_values(by=data.columns.tolist()).reset_index(drop=True)

    # Convert to a string representation
    data_string = data.to_string(index=False, header=False)

    # Generate a hash from the string
    data_hash = hashlib.md5(data_string.encode()).hexdigest()

    return data_hash

def format_data(data,filter_CHOICE=None,scaler=None,bio_scaler=None,wn_scaler=None,response_scaler=None,splitvec=None, interp_minmaxstep = None,add_scale=False):
    data = deepcopy(data)
    inactive_cols = []


    if isinstance(data,list):
        og_data_hashes = [hash_dataset(i) for i in data]
    else:
        og_data_hashes = hash_dataset(data)

    # standardize data, return the interpolation parameters (calculated automatically if None is supplied), and the indices of the wn_inds, which can
    # from here be assumed to be in ascending order, and corresponding to their original position. If multiple dataset were provided, a union operation
    # was performed

    #condition: if scaling has already been provided, look into the column values to assess if interpolation is necessary. If it is necessary, provide the values into below function.
    if not isinstance(scaler,str) and interp_minmaxstep is None:

        _, _, wn_min, step, wn_max, _ = wnExtract(scaler[0].transformers[-1][2]) #sum([i[2] for i in scaler.transformers],[])
        interp_minmaxstep = [wn_min, wn_max, step]

    data,interp_minmaxstep,wn_inds = standardize_data(data,interp_minmaxstep)

    #check that wn names are now equivalent, fix if not
    if not isinstance(scaler, str):

        data_feature_columns = [x for x in data.columns if x not in INFORMATIONAL + RESPONSE_COLUMNS]
        data_bio_columns = [x for x in data_feature_columns if WN_MATCH not in x]

        model_bio_features = [x[0] for x in sum([z.transformers for z in scaler],[]) if x[0] not in RESPONSE_COLUMNS and WN_MATCH not in x[0]]
        model_wn_features = scaler[0].transformers[-1][2]

        # check if the col names representing # are similar or not the same. Safe to assume at this point both are consecutive and on the right side of the ds.
        assert len(wn_inds) == len(model_wn_features)

        # check they are immaterially different
        assert all([abs(float(data.iloc[:, wn_inds[x]].name[len(WN_MATCH):]) - float(model_wn_features[x][len(WN_MATCH):])) < 0.001 for x in range(len(wn_inds))])

        # this resolves any naming differences due to rounding.
        data = data.rename(columns={data.iloc[:, wn_inds[x]].name: model_wn_features[x] for x in range(len(wn_inds))})

        #also while in this conditional, can add in extra bio columns that will be needed for both inference and fine-tune
        cols_active = {x: True for x in model_bio_features if ONE_HOT_FLAG not in x} #base case, will change below if not

        if not all([x in data_bio_columns for x in model_bio_features]):
            dummy_col = [col for col in model_bio_features if col not in data_bio_columns and ONE_HOT_FLAG not in col]
            data[dummy_col] = MISSING_DATA_VALUE

            [cols_active.update({i: False}) for i in dummy_col]

            #for those with ohc in col.
            model_one_hot_features = set(["".join(col.split("_")[:-2]) for col in model_bio_features if ONE_HOT_FLAG in col])

            for m in model_one_hot_features:

                if m in data:
                    present_one_hot = [f"{m}{ONE_HOT_FLAG}_{x}" for x in data[m].unique()]
                    cols_active.update({m: True})
                else:
                    present_one_hot = []
                    #set as inactive.
                    cols_active.update({m:False})
                missing_one_hot = [col for col in [l for l in model_bio_features if m in l and ONE_HOT_FLAG in l] if col not in present_one_hot]
                #if len(missing_one_hot)>0:
                    #make sure to add in an <NA> column, if previously present or not, and give it value of 1.
                data[missing_one_hot] = 0
                data[[i for i in missing_one_hot if '<NA>' in i]] = 1

        # for one hot: this should indeed consider whether certain categories are no longer present in the provided column- for instance, in the sex category,
        # if there is M,F and I (immature), we will want to establish here if 'I' column needs to be added as dummy col. for one hot, this should be given a value
        # of 0 instead of the MISSING_DATA_VALUE.

    if not SPLITNAME in data and splitvec is None:
        raise ValueError(f"Either a column of name: {SPLITNAME} or a 'splitvec' argument of [x,y], where x-0 = train %, x-y = val %, and 100-y = test % must be provided")
    if splitvec is not None:
        #reassign based on operator defined splits.

        data[SPLITNAME] = pd.Series(np.random.choice(a=["training","validation","test"],size=data.shape[0],p=[splitvec[0]/100,(splitvec[1]-splitvec[0])/100,1-(splitvec[1]/100)]))

        split_behavior = "randomly_assigned"
    else:
        #test for any NA in splitname, require that if split column is included all data needs to have it present. provide outputs for splits, sourced from the data.

        if data[SPLITNAME].isna().any():
            raise ValueError(f"Data {SPLITNAME} column is only partially defined: either provide datasets with {SPLITNAME} fully defined, or specify splitvec argument")

        split_behavior = "determined_from_dataset(s)"

        percentages = data[SPLITNAME].value_counts(normalize=True)*100

        #estimate ratio based on provided data.
        splitvec = [np.round(percentages.get('training',0)),np.round(100- percentages.get('test', 0))]

    #check ids- make sure there is an id column present, and assert that the id field does not contain duplicates.
    if IDNAME in data:
        if data[IDNAME].duplicated().any():
            raise ValueError(f"Data {IDNAME} column is not globally unique")
    else:
        raise ValueError(f"Data must contain an {IDNAME} column to distinguish it from all other samples in dataset(s)")

    #Also require string bio columns to be marked as categorical, and have the same one-hot behavior applied
    #(maybe that's an optional behavior, alternative behavior would be to include extra columns as simply informational?)
    data,dt_indices = autoOneHot(data)

    data = preprocess_spectra(data, filter_CHOICE)

    #think we might need to fill <NA> here with missing data value- pretty sure it is being converted to min val
    #by transformers.
    #if data.isnull().values.any():
    #    import code
    #    code.interact(local=dict(globals(), **locals()))
    #    data = data.fillna(MISSING_DATA_VALUE)

    data_feature_columns = [x for x in data.columns if x not in INFORMATIONAL + RESPONSE_COLUMNS]
    data_bio_columns = [x for x in data_feature_columns if WN_MATCH not in x]

    #import code
    #code.interact(local=dict(globals(), **locals()))

    if isinstance(scaler,str):

        bio_scaler = bio_scaler if bio_scaler is not None else scaler
        wn_scaler = wn_scaler if wn_scaler is not None else scaler
        response_scaler = response_scaler if response_scaler is not None else scaler

        data_mod, scaler = create_scale(data[[[i for i in data.columns][x] for x in dt_indices[0]+dt_indices[2]+dt_indices[3]]], bio_scaler,wn_scaler,response_scaler)
        data[data_mod.columns]=data_mod

        cols_active = {x: True for x in data_bio_columns}

    else:

        # assess new feature columns

        model_bio_features = [x[0] for x in sum([z.transformers[:-1] for z in scaler],[]) if x[0] not in RESPONSE_COLUMNS]
        new_features = [x for x in data_bio_columns if x not in model_bio_features and 'UNDECLARED_' not in x]

        cols_in_order = []

        #[cols_active.update({i:False}) for i in inactive_cols]

        #if splitvec == [39, 61]:

        # scale existing
        for i in list(range(1,len(scaler)+1))[::-1]:
            #print(i)
            #print(data_bio_columns)
            dummy_age = False
            if 'age' not in data.columns:
                dummy_age = True
                #easier to just make a column temporarily than try to change behavior of transform method
                data['age']=0.1

            data_mod = transform(data, scaler[-i]) #list(data.iloc[:, dt_indices[1]].columns), dt_indices[1]
            data[data_mod.columns] = data_mod

            if dummy_age:
                data = data.drop('age', axis=1)

        if add_scale and len(new_features) > 0:

            # create a scaler for new columns
            #
            if bio_scaler is None:
                bio_scalers = [[str(p.named_transformers_[i])[:-2] for i in p.named_transformers_ if i != WN_STRING_NAME or i != response_scaler] for p in scaler]
                bio_scalers = set([x for xs in bio_scalers for x in xs])
                if 'FunctionTransformer' in bio_scalers:
                    bio_scalers.remove('FunctionTransformer')

                bio_scalers = list(bio_scalers)
                bio_scaler = bio_scalers[0] if len(bio_scalers) >= 1 else 'MinMaxScaler' #haven't formally defined this as default, but w/e

            data_new,new_scaler = create_scale(data[new_features],bio_scaler)
            scaler.append(new_scaler[0])

            data[data_new.columns] = data_new

            [cols_active.update({i: True}) for i in data_new.columns if ONE_HOT_FLAG not in i]

            new_one_hot_features =  set(["".join(col.split("_")[:-2]) for col in data_new.columns if ONE_HOT_FLAG in col])

            cols_active.update({i:True} for i in new_one_hot_features)


    #if adding new
    outputs = {"scaler":scaler,"filter":filter_CHOICE,"splits":{"vec":splitvec,"origination":split_behavior}, "cols_active":cols_active,
                "datatype_indices":{"response_indices":dt_indices[0],"informational_indices":dt_indices[1],"bio_indices":dt_indices[2],"wn_indices":dt_indices[3]}}

    return data,outputs,og_data_hashes

def pad_bio_columns(data,bio_names_ordered,total_bio_columns=None,extra_bio_columns=None):

    bio_col_len = len(bio_names_ordered)
    if total_bio_columns is None and extra_bio_columns is None:
        bio_col_len = bio_col_len
    elif total_bio_columns is not None:
        bio_col_len = total_bio_columns
    else:
        bio_col_len = bio_col_len + extra_bio_columns

        # required to not have side effects on the original data object
    data = deepcopy(data)

    bio_data = data[bio_names_ordered]

    data_minus_bio_data = data.drop(bio_names_ordered, axis=1)
    data_minus_bio_data.reset_index(drop=True, inplace=True)

    # zero pad the bio data to make sure the model trains at the correct size.
    if bio_data.shape[1] < bio_col_len:
        padding = pd.DataFrame(MISSING_DATA_VALUE_UNSCALED, index=range(bio_data.shape[0]),
                               columns=[f"_UNDECLARED_{m}" for m in range(bio_data.shape[1] + 1, bio_col_len + 1)])

        bio_data.reset_index(drop=True, inplace=True)
        padding.reset_index(drop=True, inplace=True)

        bio_data = pd.concat([bio_data, padding], axis=1)

        data = pd.concat([bio_data, data_minus_bio_data], axis=1) #not the same as before, but completely internal, slightly easier logic for idx predictability

    return data,list(bio_data.columns)

# Training Mode with Hyperband 
def TrainingModeWithHyperband(data: pd.DataFrame,scaler,bio_idx, wn_idx,total_bio_columns=None,extra_bio_columns=None,max_epochs=35, epochs = 30, batch_size = 32, seed_value=42,**kwargs):

    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    #named order of bio and wn columns:
    bio_names_ordered = [data.columns[x] for x in bio_idx]
    wn_columns_names_ordered = [data.columns[x] for x in wn_idx]

    padded_data, bio_names_ordered_padded = pad_bio_columns(data, bio_names_ordered, total_bio_columns=total_bio_columns,
                                                   extra_bio_columns=extra_bio_columns)

    input_dim_A = len(bio_names_ordered_padded)
    input_dim_B = len(wn_idx)

    def model_builder(hp):
        return build_model(hp, input_dim_A, input_dim_B)

    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = Hyperband(
            model_builder,
            objective='val_loss',
            max_epochs=max_epochs,
            directory=tmpdir,
            project_name='mmcnn',
            seed=seed_value
        )

        #drop any NA in response for training
        model, best_hp = train_and_optimize_model(tuner, padded_data.dropna(subset=RESPONSE_COLUMNS), epochs, batch_size, bio_names_ordered_padded, wn_columns_names_ordered,**kwargs)
        history = final_training_pass(model, padded_data.dropna(subset=RESPONSE_COLUMNS), epochs, batch_size, bio_names_ordered_padded, wn_columns_names_ordered,**kwargs)


    stats, preds = evaluate_model(model, scaler, padded_data,bio_names_ordered_padded,wn_columns_names_ordered)

    model.summary()

    training_outputs = {
        'training_history': history,
        'stats': stats,
        'predictions': preds,
        'model_col_names': {'bio_column_names_ordered':bio_names_ordered,'bio_column_names_ordered_padded':bio_names_ordered_padded,'wn_columns_names_ordered':wn_columns_names_ordered}
    }

    #could return top 3 in extra outputs, etc.
    return model,training_outputs, {best_hp}

# Training Mode without Hyperband
def TrainingModeWithoutHyperband(data: pd.DataFrame, scaler, bio_idx, wn_idx, epochs=35, batch_size = 32, seed_value=42,total_bio_columns=None,extra_bio_columns=None, \
                                 num_conv_layers = 2, kernel_size = 101, stride_size = 51, dropout_rate = 0.1, use_max_pooling = False, num_filters = 50, dense_units = 256, dropout_rate_2 = 0.1,\
                                 **kwargs):

    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # named order of bio and wn columns:
    bio_names_ordered = [data.columns[x] for x in bio_idx]
    wn_columns_names_ordered = [data.columns[x] for x in wn_idx]

    padded_data,bio_names_ordered_padded = pad_bio_columns(data,bio_names_ordered,total_bio_columns=total_bio_columns,extra_bio_columns=extra_bio_columns)

    input_dim_A = len(bio_names_ordered_padded)
    input_dim_B = len(wn_idx)

    model = build_model_manual(
        input_dim_A,
        input_dim_B,
        num_conv_layers,
        kernel_size,
        stride_size,
        dropout_rate,
        use_max_pooling,
        num_filters,
        dense_units,
        dropout_rate_2
    )

    history = final_training_pass(model, padded_data.dropna(subset=RESPONSE_COLUMNS), epochs, batch_size,bio_names_ordered_padded,wn_columns_names_ordered,**kwargs)

    #evalution is 3 item array of metrics in .fit - loss,mse,mae
    stats, preds = evaluate_model(model, scaler, padded_data,bio_names_ordered_padded,wn_columns_names_ordered)

    model.summary()
    
    training_outputs = {
        'training_history': history,
        'stats': stats,
        'predictions': preds,
        'model_col_names': {'bio_column_names_ordered':bio_names_ordered,'bio_column_names_ordered_padded':bio_names_ordered_padded,'wn_columns_names_ordered':wn_columns_names_ordered}
    }
    
    return model,training_outputs, {}

# Training Mode with Fine-tuning 
def TrainingModeFinetuning(model,scaler, data,bio_idx,names_ordered, epochs = 35, batch_size = 32, seed_value=42,**kwargs):

    bio_names_ordered = [data.columns[x] for x in bio_idx] #this gets us the current dataset columns. can assume due to format_data dummy col behavior that previous columns are represented and values are scaled properly to
    #represent MISSING_DATA_VALUE.

    #names_ordered['bio_column_names_ordered'] #this is the old dataset columns.

    #this ensures that it respects previous order.
    bio_names_ordered_match = names_ordered['bio_column_names_ordered']+[i for i in bio_names_ordered if i not in names_ordered['bio_column_names_ordered']]

    padded_data, bio_names_ordered_padded = pad_bio_columns(data, bio_names_ordered_match, total_bio_columns=len(names_ordered['bio_column_names_ordered_padded']))

    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    history = final_training_pass(model, padded_data.dropna(subset=RESPONSE_COLUMNS), epochs, batch_size,bio_names_ordered_padded,names_ordered['wn_columns_names_ordered'],**kwargs)
    
    # Evaluate the model
    stats, preds = evaluate_model(model,scaler,padded_data,bio_names_ordered_padded,names_ordered['wn_columns_names_ordered'])

    model.summary()
    
    # Prepare the output
    training_outputs = {
        'training_history': history,
        'stats': stats,
        'predictions': preds,
        'model_col_names': {'bio_column_names_ordered': bio_names_ordered,
                            'bio_column_names_ordered_padded': bio_names_ordered_padded,
                            'wn_columns_names_ordered': names_ordered['wn_columns_names_ordered']}
    }
    
    return model,training_outputs, {}

# Spectra preprocessing function 
def preprocess_spectra(data, filter_type='savgol'):

    if filter_type is not None:

        #required to not have side effects on the original data object
        data = deepcopy(data)

        _, _, _, _, _, wn_inds = wnExtract(data.columns)

        filter_functions = {
            'savgol': savgol,
            'moving_average': moving_average,
            'gaussian': gaussian,
            'median': median,
            'wavelet': wavelet,
            'fourier': fourier,
            'pca': pca
        }

        filter_func = filter_functions.get(filter_type, savgol)

        data.iloc[:, wn_inds] = filter_func(data.iloc[:, wn_inds].values)

    return data

# Final training pass function
def final_training_pass(model, data, nb_epoch, batch_size,bio_names_ordered,wn_columns_names_ordered,**kwargs):

    history = model.fit([data.loc[data[SPLITNAME] == 'training', bio_names_ordered],
                  data.loc[data[SPLITNAME] == 'training', wn_columns_names_ordered]],
                 data.loc[data[SPLITNAME] == 'training', RESPONSE_COLUMNS],
                 epochs=nb_epoch,
                 batch_size=batch_size,
                 shuffle=True,
                 validation_data=[[data.loc[data[SPLITNAME] == 'validation', bio_names_ordered],
                                   data.loc[data[SPLITNAME] == 'validation', wn_columns_names_ordered]],
                                  data.loc[data[SPLITNAME] == 'validation', RESPONSE_COLUMNS]],
                 verbose=1,
                 **kwargs).history
    
    return history

# will  write if given a dirpath, otherwise will export the zipfile as in memory io object.
# not the prettiest behavior, open to suggestion.

def formatMetadata(metadata=None,previous_metadata=None,mandate_some_fields=True):

    if metadata is None and previous_metadata is None:
        raise ValueError("need to provide metadata to use this function: otherwise, just save with vanilla keras model.save()")

    #allow for previous metadata to be a dict, which will be interpreted as a list of len 1
    if previous_metadata is not None:
        if not isinstance(previous_metadata, list):
            print("Previous metadata is a dict: assuming only one previous training event. If not a correct assumption, supply a list of all previous metadata dicts.")
            previous_metadata = [previous_metadata]

        if metadata is not None:
            previous_metadata.append(metadata)

        metadata_all = previous_metadata
    else:
        #allow for metadata of list len 1 being supplied, or just a dict
        if isinstance(metadata, list):
            assert len(metadata)==1
            metadata_all = metadata
        else:
            metadata_all = [metadata]


    #check that mandatory fields are supplied:
    if mandate_some_fields:
        latest_metadata = metadata_all[-1]

        if not all([y in latest_metadata for y in REQUIRED_METADATA_FIELDS_ORIGINAL]):
            raise ValueError("Missing metadata: supply all following metadata: " + \
                             [x for x in REQUIRED_METADATA_FIELDS_ORIGINAL] + \
                             ", or, set 'mandate_some_metadata_fields' to FALSE in saveModelWithMetadata")

    return metadata_all
def modelToZipIO(model, model_name,metadata_all):

    with tempfile.TemporaryDirectory() as tmpdir:
        modelpathtmp = os.path.join(tmpdir, model_name)
        model.save(modelpathtmp)

        metadata_path = os.path.join(tmpdir,"metadata.pickle")
        with open(metadata_path, "w") as mf:
            mf.write(base64.b64encode(pickle.dumps(metadata_all)).decode('utf-8'))
        mf.close()

        zipdest = io.BytesIO()

        with zipfile.ZipFile(zipdest,"w",compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(modelpathtmp,arcname=model_name)
            zipf.write(metadata_path,arcname="metadata.pickle")

        zipdest.seek(0)

    return zipdest

def packModelWithMetadata(model, model_long_name,metadata=None, previous_metadata = None,mandate_some_metadata_fields=True):

    model_name = os.path.basename(model_long_name)[:-4]
    metadata_all = formatMetadata(metadata=metadata, previous_metadata=previous_metadata,mandate_some_fields=mandate_some_metadata_fields)
    zipdest = modelToZipIO(model, model_name, metadata_all =metadata_all)

    return zipdest

def saveModelWithMetadata(model, model_path,metadata=None, previous_metadata = None,mandate_some_metadata_fields=True):

    zipdest = packModelWithMetadata(model, model_path, metadata=metadata, previous_metadata=previous_metadata,mandate_some_metadata_fields=mandate_some_metadata_fields)

    with open(model_path, "wb") as f:
        f.write(zipdest.getvalue())

def loadModelWithMetadata(zip_path,model_name=None):

    #if don't supply model name, assume it inherits from the path name
    if model_name is None:
        model_name = os.path.basename(zip_path[:-4])

    with tempfile.TemporaryDirectory() as tmpdir:

        with zipfile.ZipFile(zip_path,"r") as zipf:
            zipf.extractall(tmpdir)

        model = load_model(os.path.join(tmpdir,model_name))
        with open(os.path.join(tmpdir,"metadata.pickle"),"rb") as f:
            metadata = pickle.loads(base64.b64decode(f.read()))

    #make metadata into a dict if len 1 to make behaviors more consistent to end users.
    if len(metadata)==1:
        metadata = metadata[0]

    return model,metadata

#dan comment: this works but very ugly. bugged or just not a great vis?
def plot_residuals_heatmap(y_test, preds):
    residuals = y_test - preds.flatten()
    plt.figure(figsize=(10, 6))
    sns.heatmap(np.reshape(residuals, (-1, 1)), cmap='coolwarm', annot=True, fmt='.2f')
    plt.title("Heatmap of Residuals")
    plt.xlabel("Samples")
    plt.ylabel("Residuals")
    plt.show()

def explain_model_predictions(model, X_test_biological, X_test_wavenumbers):
    explainer = shap.KernelExplainer(model.predict, [X_test_biological, X_test_wavenumbers])
    shap_values = explainer.shap_values([X_test_biological, X_test_wavenumbers])
    shap.summary_plot(shap_values, X_test_biological)

