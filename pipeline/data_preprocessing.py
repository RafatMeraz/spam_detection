import logging
import os
import string

import nltk
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text: str) -> str:
    """Transform the input text by converting into lowercase, tokenizing, removing stopwords and punctuation, and stemming"""
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_dataframe(df: pd.DataFrame, text_column = "text", target_column = "target") -> pd.DataFrame:
    """
        Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
        """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error(f'Column not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error during text normalization: {e}')
        raise

def main():
    try:
        # Fetch the data from files
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug('Data loaded successfully')

        # Data transformation
        train_processed_data = preprocess_dataframe(train_data)
        test_processed_data = preprocess_dataframe(test_data)

        # Save the processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"))
        test_processed_data.to_csv(os.path.join(data_path, "text_processed.csv"))

        logger.debug('Processed data saved successfully')

    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
    except pd.errors.EmptyDataError as e:
        logger.error(f'Empty data error: {e}')
    except Exception as e:
        logger.error(f'Error during data preprocessing: {e}')

if __name__ == "__main__":
    main()