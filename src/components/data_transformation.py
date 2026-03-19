import pandas as pd
import numpy as np
import os
import logging
import yaml
import pickle
from sklearn.preprocessing import MinMaxScaler

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_transformation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_transformation.log')
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


def scaling_data(train_data:pd.DataFrame,test_data:pd.DataFrame)->tuple:
    """Scale the data using MinMaxScaler."""
    try:
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        logger.debug('The data has been moved in range of 0 and 1 successfully.')
        return train_scaled,test_scaled,scaler
    except Exception as e:
        logger.error(f"An error occured while scaling data:{e}")
        raise

def save_scaler(scaler, scaler_path:str)->None:
    """Save the scaler object to a file."""
    try:
        scaler_dir = os.path.dirname(scaler_path)
        os.makedirs(scaler_dir, exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.debug("Scaler saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while saving the scaler: {e}")
        raise

def create_sequences(data:pd.DataFrame,n_steps:int)->tuple:
    """Create sequences of data for training the model."""
    try:
        X,y = [],[]
        for i in range(len(data)-n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps])
        logger.debug("Sequences created successfully.")
        return np.array(X),np.array(y)
    except Exception as e:
        logger.error(f"An error occured while creating sequences:{e}")
        raise

def save_data(X_train:np.array,y_train:np.array,X_test:np.array,y_test:np.array,data_path:str)->None:
    """Save the transformed data to files."""
    try:
        transformed_data_dir = os.path.join(data_path, 'transformed')
        os.makedirs(transformed_data_dir, exist_ok=True)
        X_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))  # Reshape for saving
        y_train = pd.DataFrame(y_train)
        X_test = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
        y_test = pd.DataFrame(y_test)
        X_train.to_csv(os.path.join(transformed_data_dir, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(transformed_data_dir, 'y_train.csv'), index=False)
        X_test.to_csv(os.path.join(transformed_data_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(transformed_data_dir, 'y_test.csv'), index=False)
        logger.debug("Transformed data saved successfully.")
    except Exception as e:
        logger.error(f"An error occurred while saving the transformed data: {e}")
        raise

def main():
    """Main function to execute the data transformation steps."""
    try:
        df_train = pd.read_csv("./data/raw/train_data.csv")
        df_test = pd.read_csv("./data/raw/test_data.csv")
        params = load_params("params.yaml")
        logger.debug("Successfully loaded the preprocessed data.")
        train_scaled, test_scaled,scaler = scaling_data(df_train[['close']], df_test[['close']])
        save_scaler(scaler, "./model/scaler/scaler.pkl")
        #hyperparamater 
        n_steps = params['data_transformation']['n_steps']
        X_train, y_train = create_sequences(train_scaled, n_steps)
        X_test, y_test = create_sequences(test_scaled, n_steps)
        save_data(X_train, y_train, X_test, y_test, "./data")
        logger.debug("Data transformation completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()