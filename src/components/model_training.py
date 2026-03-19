import pandas as pd
import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Activation, LayerNormalization
from keras.optimizers import Adam
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    """Load parameters from yaml file"""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from yaml file")
        return params
    except FileNotFoundError as e:
        logger.error(f"file not found:{e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML error:{e}")
        raise
    except Exception as e:
        logger.error(f"unknown error occured while loading params file: {e}")
        raise

def load_data(file_path:str)->np.array:
    """Load the transformed data from files."""
    try:
        data = pd.read_csv(file_path).values
        logger.debug("Transformed data loaded successfully.")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the transformed data files: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the transformed data: {e}")
        raise

def build_model(X_train: np.array, y_train:np.array, X_test:np.array, y_test:np.array, params):
    """Build and compile LSTM model"""
    try:
        lstm_units = params["model_training"]["lstm_units"]
        dense_units = params["model_training"]["dense_units"]
        dropout = params["model_training"]["dropout_ratio"]
        lr = params["model_training"]["learning_rate"]
        epochs = params["model_training"]["epochs"]
        patience = params["model_training"]["patience"]
        batch_size = params["model_training"]["batch_size"]

        beta_1 = params["model_training"]["beta_1"]
        beta_2 = params["model_training"]["beta_2"]
        epsilon = params["model_training"]["epsilon"]
        clipnorm = params["model_training"]["clipnorm"]

        activation = params["model_training"]["activation"]
        loss = params["model_training"]["loss"]

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        model = Sequential([
            LSTM(lstm_units, return_sequences=False, input_shape=(30,1)),
            Dropout(dropout),
            Dense(dense_units),
            LayerNormalization(),
            Activation(activation),
            Dropout(dropout),
            Dense(1)
        ])

        opt = Adam(
            learning_rate=lr,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            clipnorm=clipnorm
        )

        model.compile(optimizer=opt, loss=loss)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1, batch_size = batch_size
        )

        return model

    except Exception as e:
        logger.error(f"Error in model building: {e}")
        raise

def save_model(model:Sequential,model_path:str)->None:
    """Save the trained model to a file."""
    try:
        model_dir = os.path.join(model_path, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, 'lstm_model.h5'))
        logger.debug("Model saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while saving the model: {e}")
        raise

def main():
    """Main function to execute the model training steps."""
    try:
        params = load_params("params.yaml")
        X_train = load_data("./data/transformed/X_train.csv")
        y_train = load_data("./data/transformed/y_train.csv")
        X_test = load_data("./data/transformed/X_test.csv")
        y_test = load_data("./data/transformed/y_test.csv")
        logger.debug("Successfully loaded the transformed data for training.")
        X_train = X_train.reshape(X_train.shape[0], 30, 1)
        X_test = X_test.reshape(X_test.shape[0], 30, 1)
        model = build_model(X_train,y_train,X_test,y_test,params)
        save_model(model, "./model")
        logger.debug("Model training process completed successfully.")
    except Exception as e:
        logger.error(f"failed to train and save the model: {e}")
        raise

if __name__ == '__main__':
    main()

        


        
