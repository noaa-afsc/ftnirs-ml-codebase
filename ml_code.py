import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import pyCompare
import matplotlib
import keras_tuner as kt
import h5py
import json
import joblib
import pickle
import base64

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Normalizer, RobustScaler, MaxAbsScaler
from yellowbrick.regressor import PredictionError, ResidualsPlot
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from keras_tuner.tuners import BayesianOptimization, Hyperband
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU, Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
import pywt
from sklearn.decomposition import PCA
import sys
from io import StringIO


# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="whitegrid", font_scale=2)

def enforce_data_format(data: pd.DataFrame):
    expected_first_10 = ['filename', 'sample', 'age', 'weight', 'length', 'latitude', 'longitude', 'sex_M', 'sex_F', 'sex_immature']

    if len(data.columns) < 100:
        raise ValueError(f"DataFrame should have at least 100 columns, but it has {len(data.columns)}.")

    for i, expected_name in enumerate(expected_first_10):
        if data.columns[i] != expected_name:
            raise ValueError(f"Column {i+1} should be named '{expected_name}', but found '{data.columns[i]}'.")

    # Columns 11 to 100 can be named anything

    # Check that the remaining columns (101 onwards) contain 'wavenumber' or 'wn' (case-insensitive)
    for col in data.columns[100:]:
        if not any(keyword in col.lower() for keyword in ['wavenumber', 'wn']):
            raise ValueError(f"Column '{col}' does not contain 'wavenumber' or 'wn'.")

    return True  

def read_and_clean_data(data_source, drop_outliers=True):
    if isinstance(data_source, str):
        data = pd.read_csv(data_source)
    elif isinstance(data_source, StringIO):
        data = pd.read_csv(data_source)
    else:
        raise ValueError("data_source must be a file path (str) or a StringIO object.")
    
    enforce_data_format(data) # throws error if format not followed
    
    if data.isnull().values.any():
        raise ValueError("Data contains NaN values")

    return data

# Savitzky-Golay filter function
def savgol_filter_func(data, window_length=17, polyorder=2, deriv=1):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=deriv)

# Moving Average filter function
def moving_average_filter(data, size=5):
    return uniform_filter1d(data, size=size, axis=1)

# Gaussian filter function
def gaussian_filter_func(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma, axis=1)

# Median filter function
def median_filter_func(data, size=5):
    return median_filter(data, size=(1, size))

# Wavelet filter function
def wavelet_filter_func(data, wavelet='db1', level=1):
    def apply_wavelet(signal):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        coeffs[1:] = [pywt.threshold(i, value=0.5 * max(i)) for i in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)
    
    return np.apply_along_axis(apply_wavelet, axis=1, arr=data)

# Fourier filter function
def fourier_filter_func(data, threshold=0.1):
    def apply_fft(signal):
        fft_data = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        fft_data[np.abs(frequencies) > threshold] = 0
        return np.fft.ifft(fft_data).real
    
    return np.apply_along_axis(apply_fft, axis=1, arr=data)

# PCA filter function
def pca_filter_func(data, n_components=5):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return pca.inverse_transform(transformed)

def apply_normalization(data, columns):
    normalizer = Normalizer()
    data[columns] = normalizer.fit_transform(data[columns])
    return data

def apply_robust_scaling(data, feature_columns):
    scaler_y = RobustScaler()
    data['age'] = scaler_y.fit_transform(data[['age']])
    data[feature_columns] = data[feature_columns].apply(lambda col: RobustScaler().fit_transform(col.values.reshape(-1, 1)))
    return data, scaler_y

def apply_minmax_scaling(data, feature_columns):
    scaler_y = MinMaxScaler()
    data['age'] = scaler_y.fit_transform(data[['age']])
    data[feature_columns] = data[feature_columns].apply(lambda col: MinMaxScaler().fit_transform(col.values.reshape(-1, 1)))
    return data, scaler_y

def apply_maxabs_scaling(data, feature_columns):
    scaler_y = MaxAbsScaler()
    data['age'] = scaler_y.fit_transform(data[['age']])
    data[feature_columns] = data[feature_columns].apply(lambda col: MaxAbsScaler().fit_transform(col.values.reshape(-1, 1)))
    return data, scaler_y

def apply_scaling(data, scaling_method='standard'):
    scalers = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'maxabs': MaxAbsScaler,
        'robust': RobustScaler,
        'normalize': Normalizer  # Note: Normalizer might not be suitable for y, use with caution
    }
    
    if scaling_method not in scalers:
        raise ValueError(f"Unsupported scaling method: {scaling_method}")
    
    feature_columns = data.columns.difference(['filename', 'sample', 'age'])
    
    # Create and fit a separate scaler for the 'age' column
    scaler_y = scalers[scaling_method]()
    data['age'] = scaler_y.fit_transform(data[['age']])
    
    # Create and fit a separate scaler for the feature columns
    scaler_x = scalers[scaling_method]()
    data[feature_columns] = scaler_x.fit_transform(data[feature_columns])
    
    return data, scaler_x, scaler_y

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
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model

def train_and_optimize_model(tuner, data, nb_epoch, batch_size):
    outputFilePath = 'Estimator'
    checkpointer = ModelCheckpoint(filepath=outputFilePath, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)

    X_train_biological_data = data.loc[data['sample'] == 'training', data.columns[3:100]]
    X_train_wavenumbers = data.loc[data['sample'] == 'training', data.columns[100:1100]]
    y_train = data.loc[data['sample'] == 'training', 'age']

    tuner.search([X_train_biological_data, X_train_wavenumbers], y_train,
                 epochs=nb_epoch,
                 batch_size=batch_size,
                 shuffle=True,
                 validation_split=0.25,
                 verbose=1,
                 callbacks=[earlystop])

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

def evaluate_model(model, data):
    X_test_biological_data = data.loc[data['sample'] == 'test', data.columns[3:100]]
    X_test_wavenumbers = data.loc[data['sample'] == 'test', data.columns[100:1100]]
    y_test = data.loc[data['sample'] == 'test', 'age']

    evaluation = model.evaluate([X_test_biological_data, X_test_wavenumbers], y_test)
    preds = model.predict([X_test_biological_data, X_test_wavenumbers])
    r2 = r2_score(y_test, preds)
    return evaluation, preds, r2

def plot_predictions(y_test, preds):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    lims = [-2.5, 5]
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

    y_tr_df.to_csv('./Output/Data/train_predictions.csv', index=False)

    return r_squared_tr, rmse_tr, y_tr_df

def plot_training_set(y_tr_transformed, y_pr_transformed):
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
    plt.savefig('./Output/Figures/TrainingSet.png')
    plt.show()

def plot_test_set(y_test_transformed, y_pred_transformed):
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
    plt.savefig('./Output/Figures/TestSet.png')
    plt.show()

def plot_bland_altman(y_test_transformed, y_pred_transformed):
    pyCompare.blandAltman(y_test_transformed.flatten(), y_pred_transformed.flatten(),
                          limitOfAgreement=1.96, confidenceInterval=95,
                          confidenceIntervalMethod='approximate',
                          detrend=None, percentage=False,
                          title='Bland-Altman Plot\n',
                          savePath='./Output/Figures/BlandAltman.png')

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
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model

def InferenceMode(model_or_path, data, row_number, scaler_y=None, scaler_x=None):
    if isinstance(model_or_path, str):
        # Load model from disk
        model = load_model(model_or_path)
    else:
        # Use the provided model object
        model = model_or_path
    
    # Extract the specific row for inference
    sample_data = data.iloc[row_number]

    # Extract the full set of features (columns 3 to 1100)
    full_data = sample_data[data.columns[3:1100]].values.reshape(1, -1)
    
    # Apply the same scaling used during training
    if scaler_x:
        full_data = scaler_x.transform(full_data)
    
    # Split the scaled data into biological and wavenumber data
    biological_data = full_data[:, :97]  # Columns 3-100 (biological variables)
    wavenumber_data = full_data[:, 97:].reshape(1, -1, 1)  # Columns 101-1100 (wavenumber variables)
    
    # Run inference
    prediction = model.predict([biological_data, wavenumber_data])
    
    # Inverse transform the prediction to the original scale
    if scaler_y:
        prediction = scaler_y.inverse_transform(prediction)
    
    return prediction


def TrainingModeWithHyperband(raw_data, filter_CHOICE, scaling_CHOICE, seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
    data = preprocess_spectra(raw_data, filter_type=filter_CHOICE)
    
    scaling_method = scaling_CHOICE  # 'minmax', 'standard', 'maxabs', 'robust', or 'normalize'
    data, scaler_x, scaler_y = apply_scaling(data, scaling_method)

    input_dim_A = data.columns[3:100].shape[0]
    input_dim_B = data.columns[100:1100].shape[0]
    
    def model_builder(hp):
        return build_model(hp, input_dim_A, input_dim_B)
    
    tuner = Hyperband(
        model_builder,
        objective='val_loss',
        max_epochs=1,
        directory='Tuners',
        project_name='mmcnn',
        seed=42
    )
        
    nb_epoch = 1 
    batch_size = 32
    model, best_hp = train_and_optimize_model(tuner, data, nb_epoch, batch_size)
    
    nb_epoch = 1 
    batch_size = 32
    history = final_training_pass(model, data, nb_epoch, batch_size)
    
    evaluation, preds, r2 = evaluate_model(model, data)
    
    model.summary()
    
    training_outputs = {
        'trained_model': model,
        'scaler_x' : scaler_x,
        'scaler_y' : scaler_y,
        'training_history': history,
        'evaluation': evaluation,
        'predictions': preds,
        'r2_score': r2
    }
    
    # top_5_models = tuner.get_best_models(num_models=5)
    
    additional_outputs = {
        # 'top5models': top_5_models
    }
    
    return training_outputs, additional_outputs

def TrainingModeWithoutHyperband(raw_data, filter_CHOICE, scaling_CHOICE, model_parameters, seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    enforce_data_format(raw_data)

    if len(model_parameters) != 8:
        raise ValueError("model_parameters must be a list of 8 values.")
    
    num_conv_layers, kernel_size, stride_size, dropout_rate, use_max_pooling, num_filters, dense_units, dropout_rate_2 = model_parameters
    
    if not all(isinstance(param, (int, float, bool)) for param in model_parameters):
        raise ValueError("All model parameters must be either int, float, or bool.")
    
    data = preprocess_spectra(raw_data, filter_type=filter_CHOICE)
    
    scaling_method = scaling_CHOICE  # 'minmax', 'standard', 'maxabs', 'robust', or 'normalize'
    data, scaler_x, scaler_y = apply_scaling(data, scaling_method)

    input_dim_A = 97  # Columns 3-100
    input_dim_B = 1000  # Columns 101-1100

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
    
    nb_epoch = 1  # !@!
    batch_size = 32
    history = final_training_pass(model, data, nb_epoch, batch_size)
    
    evaluation, preds, r2 = evaluate_model(model, data)
    print(f"Evaluation: {evaluation}, R2: {r2}")
    
    model.summary()
    
    training_outputs = {
        'trained_model': model,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'training_history': history,
        'evaluation': evaluation,
        'predictions': preds,
        'r2_score': r2
    }
    
    additional_outputs = {}
    
    return training_outputs, additional_outputs

def TrainingModeFinetuning(raw_data, filter_CHOICE, scaling_CHOICE, file_path, seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Ensure data format
    enforce_data_format(raw_data)

    # Preprocess the data
    data = preprocess_spectra(raw_data, filter_type=filter_CHOICE)
    
    # Apply scaling
    scaling_method = scaling_CHOICE  # 'minmax', 'standard', 'maxabs', 'robust', or 'normalize'
    data, scaler_x, scaler_y = apply_scaling(data, scaling_method)

    # Load the pretrained model
    model = load_model(file_path)
    print(f"Loaded model from {file_path}")

    # Set the input dimensions based on the data format
    input_dim_A = 97  # Columns 3-100
    input_dim_B = 1000  # Columns 101-1100

    # Fine-tune the model on the new data
    nb_epoch = 1  # Adjust the number of epochs as needed
    batch_size = 32
    history = final_training_pass(model, data, nb_epoch, batch_size)
    
    # Evaluate the model
    evaluation, preds, r2 = evaluate_model(model, data)
    print(f"Evaluation: {evaluation}, R2: {r2}")
    
    model.summary()
    
    # Prepare the output
    training_outputs = {
        'trained_model': model,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'training_history': history,
        'evaluation': evaluation,
        'predictions': preds,
        'r2_score': r2
    }
    
    additional_outputs = {}
    
    return training_outputs, additional_outputs


def preprocess_spectra(data, filter_type='savgol'):
    filter_functions = {
        'savgol': savgol_filter_func,
        'moving_average': moving_average_filter,
        'gaussian': gaussian_filter_func,
        'median': median_filter_func,
        'wavelet': wavelet_filter_func,
        'fourier': fourier_filter_func,
        'pca': pca_filter_func
    }
    
    filter_func = filter_functions.get(filter_type, savgol_filter_func)
    
    data.loc[data['sample'] == 'training', data.columns[100:1100]] = filter_func(data.loc[data['sample'] == 'training', data.columns[100:1100]].values)
    data.loc[data['sample'] == 'test', data.columns[100:1100]] = filter_func(data.loc[data['sample'] == 'test', data.columns[100:1100]].values)
    
    return data

def final_training_pass(model, data, nb_epoch, batch_size):
    outputFilePath = 'Estimator'
    checkpointer = ModelCheckpoint(filepath=outputFilePath, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)

    X_train_biological_data = data.loc[data['sample'] == 'training', data.columns[3:100]]
    X_train_wavenumbers = data.loc[data['sample'] == 'training', data.columns[100:1100]]
    y_train = data.loc[data['sample'] == 'training', 'age']

    history = model.fit([X_train_biological_data, X_train_wavenumbers], y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.25,
                        verbose=1,
                        callbacks=[checkpointer, earlystop]).history
    
    return history


def saveModel(model, path):
    model.save(path)
    print(f"Model saved to {path}")    

def saveModelWithMetadata(path, model, old_metadata_path, column_names, description, training_approach, scaler_x, scaler_y):

    # Save the Keras model
    model.save(path)
    print(f"Model only saved to {path}")

    # Serialize scalers to strings using pickle and encode with base64
    scaler_x_str = base64.b64encode(pickle.dumps(scaler_x)).decode('utf-8')
    scaler_y_str = base64.b64encode(pickle.dumps(scaler_y)).decode('utf-8')

    # Read the old metadata from the old metadata path
    try:
        old_metadata = readModelMetadata(old_metadata_path)
    except:
        old_metadata = []

    # Create the new metadata entry
    new_metadata = [
        json.dumps(column_names),
        description,
        training_approach,
        scaler_x_str,
        scaler_y_str
    ]

    # Append the new metadata to the old metadata list
    old_metadata.append(new_metadata)

    # Append the updated metadata to the HDF5 file
    with h5py.File(path, 'a') as f:
        if 'metadata' not in f:
            metadata_group = f.create_group('metadata')
        else:
            metadata_group = f['metadata']

        # Convert the entire metadata to a string representation
        metadata_str = json.dumps(old_metadata)

        # Save the entire metadata as a new dataset
        metadata_group.create_dataset('metadata_history', data=metadata_str)

    print(f"Metadata added to {path}")

def readModelMetadata(path):
    with h5py.File(path, 'r') as f:
        if 'metadata' in f:
            metadata_str = f['metadata']['metadata_history'][()]
            # Convert the metadata string back to a list of lists
            metadata = json.loads(metadata_str)
        else:
            metadata = []

    # Extract the latest metadata entry
    if metadata:
        latest_metadata = metadata[-1]
        
        # Deserialize each component
        column_names = json.loads(latest_metadata[0])
        description = latest_metadata[1]
        training_approach = latest_metadata[2]
        scaler_x = pickle.loads(base64.b64decode(latest_metadata[3]))
        scaler_y = pickle.loads(base64.b64decode(latest_metadata[4]))
    else:
        raise ValueError("No metadata found in the provided path.")

    return column_names, description, training_approach, scaler_x, scaler_y



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

    saveModel(model, "my_model.h5")


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

    metadata_path = 'my_model_with_metadata.h5'
    column_names, description, training_approach, scaler_x, scaler_y = readModelMetadata(metadata_path)
 
    training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(raw_data=raw_data, 
                           filter_CHOICE='savgol',
                           scaling_CHOICE='minmax', 
                           file_path='my_model_with_metadata.h5',
                           seed_value=42)


if __name__ == "__main__":
    main()