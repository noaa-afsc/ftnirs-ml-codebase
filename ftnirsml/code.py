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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Normalizer, RobustScaler, MaxAbsScaler
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from keras_tuner.tuners import BayesianOptimization, Hyperband
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU, Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
#PyWavelets
import pywt
from sklearn.decomposition import PCA
import sys
from sklearn.model_selection import KFold
import shap
# Use for internal functions which rely on intermediates
import tempfile

# Set seeds for reproducibility 
np.random.seed(42)
tf.random.set_seed(42)

# Seaborn configuration
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

def read_and_clean_data(data, drop_outliers=True):

    enforce_data_format(data) # throws error if format not followed
    
    if data.isnull().values.any():
        raise ValueError("Data contains NaN values")

    return data

# Filter functions
def savgol_filter_func(data, window_length=17, polyorder=2, deriv=1):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=deriv)

def moving_average_filter(data, size=5):
    return uniform_filter1d(data, size=size, axis=1)

def gaussian_filter_func(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma, axis=1)

def median_filter_func(data, size=5):
    return median_filter(data, size=(1, size))

def wavelet_filter_func(data, wavelet='db1', level=1):
    def apply_wavelet(signal):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        coeffs[1:] = [pywt.threshold(i, value=0.5 * max(i)) for i in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)
    return np.apply_along_axis(apply_wavelet, axis=1, arr=data)

def fourier_filter_func(data, threshold=0.1):
    def apply_fft(signal):
        fft_data = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        fft_data[np.abs(frequencies) > threshold] = 0
        return np.fft.ifft(fft_data).real
    return np.apply_along_axis(apply_fft, axis=1, arr=data)

def pca_filter_func(data, n_components=5):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return pca.inverse_transform(transformed)

# Scaling functions
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
def train_and_optimize_model(tuner, data, nb_epoch, batch_size):
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

def bland_altman_plot(y_test_transformed, y_pred_transformed):
    plt = pyCompare.blandAltman(y_test_transformed.flatten(), y_pred_transformed.flatten(),
                          limitOfAgreement=1.96, confidenceInterval=95,
                          confidenceIntervalMethod='approximate',
                          detrend=None, percentage=False,
                          title='Bland-Altman Plot\n')
    return plt

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
def InferenceMode(model_or_path, data, scaler_y=None, scaler_x=None):
    if isinstance(model_or_path, str):
        # Load model from disk
        model = load_model(model_or_path)
    else:
        # Use the provided model object
        model = model_or_path

    feature_columns = data.columns.difference(['filename', 'sample', 'age'])

    #print(scaler_x)

    # Apply the same scaling used during training
    if scaler_x:
        data[feature_columns] = scaler_x.transform(data[feature_columns])

    biological_data = data[data.columns[3:100]]
    wavenumber_data = data[data.columns[100:1100]]

    #import hashlib
    #print('hash value of data struct into predict:')
    #hasher = hashlib.md5()
    #test1 = biological_data
    #test1[0][:]= np.empty(len(test1[0][:]), dtype=object)
    #hasher.update((pickle.dumps([test1, wavenumber_data])))
    #print(hasher.hexdigest())
    #should equal: '805313e71322ad2fcfe6cd6892147704'
    
    # Run inference
    prediction = model.predict([biological_data, wavenumber_data])
    
    # Inverse transform the prediction to the original scale
    if scaler_y:
        prediction = scaler_y.inverse_transform(prediction)
    
    return prediction

# Training Mode with Hyperband 
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = Hyperband(
            model_builder,
            objective='val_loss',
            max_epochs=1,
            directory=tmpdir,
            project_name='mmcnn',
            seed=42
        )

        nb_epoch = 1
        batch_size = 32
        model, best_hp = train_and_optimize_model(tuner, data, nb_epoch, batch_size)
        history = final_training_pass(model, data, nb_epoch, batch_size)
    
    evaluation, preds, r2 = evaluate_model(model, data)
    
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

    return training_outputs, {}

# Training Mode without Hyperband
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
    
    return training_outputs, {}

# Training Mode with Fine-tuning 
def TrainingModeFinetuning(model, raw_data, filter_CHOICE, scaling_CHOICE, seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Ensure data format
    enforce_data_format(raw_data)

    # Preprocess the data
    data = preprocess_spectra(raw_data, filter_type=filter_CHOICE)
    
    # Apply scaling
    scaling_method = scaling_CHOICE  # 'minmax', 'standard', 'maxabs', 'robust', or 'normalize'
    data, scaler_x, scaler_y = apply_scaling(data, scaling_method)

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
    
    return training_outputs, {}

# Spectra preprocessing function 
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

# Final training pass function
def final_training_pass(model, data, nb_epoch, batch_size):
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
                        callbacks=[earlystop]).history
    
    return history


# will  write if given a dirpath, otherwise will export the zipfile as in memory io object.
# not the prettiest behavior, open to suggestion.

def modelToZipIO(model, model_name,metadata=None, previous_metadata = None):

    if metadata is None and previous_metadata is None:
        return ValueError("need to provide metadata to use this function: otherwise, just save with vanilla keras model.save()")

    if previous_metadata is not None:
        assert isinstance(previous_metadata, list)  # the metadata object can be w/e, but needs to store lineage in a list.
        previous_metadata.append(metadata)
        metadata_all = previous_metadata
    else:
        metadata_all = [metadata]


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
def saveModelWithMetadata(model, model_path,metadata=None, previous_metadata = None):

    model_name = os.path.basename(model_path)[:-4]

    zipdest = modelToZipIO(model, model_name, metadata=metadata, previous_metadata=previous_metadata)

    with open(model_path, "wb") as f:
        f.write(zipdest.getvalue())

def loadModelWithMetadata(path):

    with tempfile.TemporaryDirectory() as tmpdir:

        with zipfile.ZipFile(path,"r") as zipf:
            zipf.extractall(tmpdir)

        model = load_model(os.path.join(tmpdir,os.path.basename(path[:-4])))
        with open(os.path.join(tmpdir,"metadata.pickle"),"rb") as f:
            metadata = pickle.loads(base64.b64decode(f.read()))

    return model,metadata

def cross_validate_model(model, data, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    X_biological = data[data.columns[3:100]]
    X_wavenumbers = data[data.columns[100:1100]]
    y = data['age']
    
    scores = []
    for train_index, test_index in kf.split(X_biological):
        X_train_biological, X_test_biological = X_biological.iloc[train_index], X_biological.iloc[test_index]
        X_train_wavenumbers, X_test_wavenumbers = X_wavenumbers.iloc[train_index], X_wavenumbers.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit([X_train_biological, X_train_wavenumbers], y_train, epochs=5, verbose=0)
        score = model.evaluate([X_test_biological, X_test_wavenumbers], y_test, verbose=0)
        scores.append(score)
    
    return scores

def compare_models(models, data):
    comparison_results = {}
    X_test_biological = data[data.columns[3:100]]
    X_test_wavenumbers = data[data.columns[100:1100]]
    y_test = data['age']
    
    for model_name, model in models.items():
        preds = model.predict([X_test_biological, X_test_wavenumbers])
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        comparison_results[model_name] = {'R2': r2, 'MSE': mse}
    
    return comparison_results

from sklearn.model_selection import GridSearchCV

# Probably don't let people run this on shared resources/web hosting but let them run it on their own systems
def grid_search_model(data, param_grid):
    X_biological = data[data.columns[3:100]]
    X_wavenumbers = data[data.columns[100:1100]]
    y = data['age']
    
    model = KerasRegressor(build_fn=build_model_manual, epochs=10, batch_size=32, verbose=0)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_result = grid.fit([X_biological, X_wavenumbers], y)
    
    return grid_result.best_params_, grid_result.best_score_

def plot_residuals_heatmap(y_test, preds):
    residuals = y_test - preds.flatten()
    plt.figure(figsize=(10, 6))
    sns.heatmap(np.reshape(residuals, (-1, 1)), cmap='coolwarm', annot=True, fmt='.2f')
    plt.title("Heatmap of Residuals")
    plt.xlabel("Samples")
    plt.ylabel("Residuals")
    plt.show()

import logging

def setup_logging(logfile='model_training.log'):
    logging.basicConfig(filename=logfile, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def log_training_event(event_message):
    logging.info(event_message)

def handle_missing_data(data):
    if data.isnull().values.any():
        missing_data_report = data.isnull().sum()
        print(f"Missing Data Report:\n{missing_data_report}")
        # Filling missing values with mean
        data.fillna(data.mean(), inplace=True)
    return data

def explain_model_predictions(model, X_test_biological, X_test_wavenumbers):
    explainer = shap.KernelExplainer(model.predict, [X_test_biological, X_test_wavenumbers])
    shap_values = explainer.shap_values([X_test_biological, X_test_wavenumbers])
    shap.summary_plot(shap_values, X_test_biological)
