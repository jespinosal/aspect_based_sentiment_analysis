import pandas as pd
from config.config import CROSS_DATA_SET_FILE_PATH
from src.constants import DataStructure, PolarityLabels
#from sklearn.utils import shuffle


def data_set_binary_balance(df):
    #df = shuffle(df)
    df_pos = df[df[DataStructure.POLARITY].isin([PolarityLabels.POSITIVE])]
    df_neg = df[df[DataStructure.POLARITY].isin([PolarityLabels.NEGATIVE])]
    if len(df_pos) > len(df_neg):
        df_pos = df_pos[0:len(df_neg)]

    else:
        df_neg = df_neg[0:len(df_pos)]

    return pd.concat([df_pos, df_neg])


def filter_size(df):
    df = df.copy()
    df['token_lenght'] = df[DataStructure.REVIEW].apply(lambda x: len(x.split()))
    q1 = df['token_lenght'].quantile(q=0.25)
    q2 = df['token_lenght'].quantile(q=0.75)
    df = df[(df['token_lenght'] > q1) & (df['token_lenght'] < q2)]
    del df['token_lenght']
    return df


def data_reader_sentiment_analysis(file_path=None):
    """
    Columns : ;source;product_name;review_text;rating;polarity
    :return:
    """
    file_path = CROSS_DATA_SET_FILE_PATH if file_path is None else file_path
    df_sentiment_analysis = pd.read_csv(file_path, sep=';',
                                        dtype={'source': str, "product_name": str,
                                               'review_text': str, 'rating': float,
                                               'polarity': str})
    df_sentiment_analysis = df_sentiment_analysis[~df_sentiment_analysis[DataStructure.REVIEW].isna()]
    df_sentiment_analysis = filter_size(df=df_sentiment_analysis)
    df_sentiment_analysis = df_sentiment_analysis[df_sentiment_analysis[DataStructure.POLARITY].isin([
        PolarityLabels.POSITIVE, PolarityLabels.NEGATIVE])]
    df_sentiment_analysis = data_set_binary_balance(df=df_sentiment_analysis)

    return df_sentiment_analysis


if __name__ == "__main__":
    df_ = data_reader_sentiment_analysis()
