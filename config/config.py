import os

DATA_PATH = 'data'
DATA_PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw')

ASPECT_DATA_SET_FILE_NAME = 'aspect_dataset.csv'
CROSS_DATA_SET_FILE_NAME = 'reviews_dataset.csv'
CROSS_DATA_SET_FILE_PATH = os.path.join(DATA_PROCESSED_PATH, CROSS_DATA_SET_FILE_NAME)
ASPECT_DATA_SET_FILE_PATH = os.path.join(DATA_PROCESSED_PATH, ASPECT_DATA_SET_FILE_NAME)

SENTIMENT_ANALYSIS_POLARITY_CLASSIFICATION_CONFIG = os.path.join('data', 'models', 'sa_polarity')
ASPECT_CLASSIFICATION_MODEL_PATH = os.path.join('data', 'models', 'aspect_classifier')


COMMON_MODELS = {
                'logs': './logs',
                'temp': './temp',
                 }