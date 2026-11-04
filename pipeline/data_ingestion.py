import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Log Config
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path: str) -> pd.DataFrame:
    """Load the data from CSV file"""
    try:
        df = pd.read_csv(data_path)
        logger.debug(f"Loaded data from {data_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load the csv file: {e}")
        raise

def preprocess_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-process the Dataframe"""
    try:
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
        df = df.rename(columns = {'v1': 'target', 'v2': 'text'})
        logger.debug("Data processing success!")
        return df
    except KeyError as e:
        logger.error(f"Missing column in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to preprocess the csv file: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test data to local files"""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug(f"Data saved successfully to {raw_data_path}")
    except Exception as e:
        logger.error(f"Failed to save the data: {e}")
        raise


def main():
    try:
        test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(data_path)
        processed_data = preprocess_csv(df)
        train_data, test_data = train_test_split(processed_data, test_size=test_size)
        save_data(train_data, test_data, "./data")
    except Exception as e:
        logger.error(f"Failed to complete the data ingestion: {e}")


if __name__ == "__main__":
    main()