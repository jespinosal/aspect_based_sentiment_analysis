"""
Note: text pre processing pipelines will include different methods according the model purpose.
For transformer models lemma, stop word removal and word tokenization are not necessary.
"""
import re
import string
import unicodedata
from nltk import tokenize


def word_tokenizer_basic(text):
    """
    Perform a simpler word tokenization
    :param text:
    :return:
    """
    return tokenize.word_tokenize(text=text, language='english')


def convert_to_lower(sentence):
    """
    Lower case for the text documents
    """
    return sentence.lower()


def unicode_normalizer(text):
    """
    Convert unicode characters to ASCII.
    :return:
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def white_space_normalizer(text: str) -> str:
    pattern = re.compile(r'\s+')
    text = re.sub(pattern, ' ', text)
    return text.strip()


def delete_punctuation(text):
    """Delete puntuation tokens in text string """
    # it generates white spaces when punctuation is separated from text: "are you ok ?" -> "are you ok  "
    translation_dict = str.maketrans('', '', string.punctuation)
    return text.translate(translation_dict)


def format_abbreviation(text):
    """
    Source: https://www.programcreek.com/python/?CodeExample=normalize+text
    :param text:
    :return:
    """
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    return text


def format_punctuation(text):
    """
    Source: https://www.programcreek.com/python/?CodeExample=normalize+text
    :param text:
    :return:
    """
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    return text


def alphanumeric_normalizer(text):
    """

    :param text:
    :return:
    """
    return re.sub(r"[^A-Za-z0-9(),:;!?\'\`]", " ", text)


def basic_pre_processing(text):
    if type(text) is str:
        text = unicode_normalizer(text)
        text = alphanumeric_normalizer(text)
        text = white_space_normalizer(text)
    else:
        text = None
    return text


if __name__ == "__main__":
    text_ = "你好吗 I'm happy to test. I'll check #1000 documents!  :) ;) :P Right?"
    text_ = basic_pre_processing(text=text_)
