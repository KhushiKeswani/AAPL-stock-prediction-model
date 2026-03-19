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

def build_model(X_train:np.array,y_train:np.array,X_test:np.array,y_test:np.array)->Sequential:
    """Build and compile the LSTM model."""
    try:
        if(len(X_train) != len(y_train)):
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20,             #hyperparameter to be tuned
        restore_best_weights=True  # Automatically reverts to best weights
        )
        model = model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(30, 1)),  # Tomorrow only!
        Dropout(0.2), #hyperparameter to be tuned
        Dense(32),
        LayerNormalization(),
        Activation('relu'),
        Dropout(0.2),
        Dense(1) # Single price!
        ])
        logger.debug("Model architecture defined successfully.")
        opt= Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,clipnorm =1.0)
        model.compile(optimizer=opt, loss='mse')
        model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks = [early_stop],epochs=100, verbose=1)
        logger.debug("Model built and trained successfully.")
        return model
    except Exception as e:
        logger.error(f"An error occurred while building the model: {e}")
        raise

def save_model(model:Sequential,model_path:str)->None:
    """Save the trained model to a file."""
    try:
        model_dir = os.path.join(model_path, 'model')
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, 'lstm_model.h5'))
        logger.debug("Model saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while saving the model: {e}")
        raise

def main():
    """Main function to execute the model training steps."""
    try:
        X_train = load_data("./data/transformed/X_train.csv")
        y_train = load_data("./data/transformed/y_train.csv")
        X_test = load_data("./data/transformed/X_test.csv")
        y_test = load_data("./data/transformed/y_test.csv")
        logger.debug("Successfully loaded the transformed data for training.")
        X_train = X_train.reshape(X_train.shape[0], 30, 1)
        X_test = X_test.reshape(X_test.shape[0], 30, 1)
        model = build_model(X_train,y_train,X_test,y_test)
        save_model(model, "./model")
        logger.debug("Model training process completed successfully.")
    except Exception as e:
        logger.error(f"failed to train and save the model: {e}")
        raise

if __name__ == '__main__':
    main()

        


        
