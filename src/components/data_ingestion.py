import pandas as pd
import os
import logging
import yaml


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path:str)->pd.DataFrame:
    """Load data from a csv file."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Successfully loaded the data from the csv file.")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the data: {e}")
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """preprocess the data for making it ready for training."""
    try:
        df['date'] = pd.to_datetime(df['date'])

        df = df.dropna()
        logger.debug("Successfully preprocessed the data.")
        return df
    except Exception as e:
        logger.error(f"An error occurred while preprocessing the data: {e}")
        raise

def save_preprocessed_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_dir = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_dir, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_dir, 'test_data.csv'), index=False)
        logger.debug("Successfully saved the preprocessed data.")
    except Exception as e:
        logger.error(f"An error occurred while saving the preprocessed data: {e}")
        raise

def main():
    try:
        base_path = r"C:\Users\DELL\OneDrive\Desktop\Documents\dataaaa"
        data_path = os.path.join(base_path, "AAPL_featuress.csv")
        df = load_data(data_path)
        final_df = preprocess_data(df)
        split_idx = int(0.8 * len(df))

        train_data = final_df[:split_idx]
        test_data = final_df[split_idx:]
        save_preprocessed_data(train_data, test_data, "./data")
        logger.debug("Data ingestion process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == '__main__':
    main()