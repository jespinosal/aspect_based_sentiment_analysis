"""
from config.config import COMMON_MODELS, ASPECT_CLASSIFICATION_MODEL_PATH
from src.models.aspect_extraction_classifier.model_mgr import AspectClassifierManager

self.aspect_module_classifier: AspectClassifierManager = None
    def _load_aspect_classifier_model(self):
        self.aspect_module_classifier = AspectClassifierManager(model_path=ASPECT_CLASSIFICATION_MODEL_PATH)

"""

import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from src.models.aspect_extraction_rule_based.aspect_extractor_rules import AspectExtractorMgr, word_id_sentence_getter
from src.models.sentiment_polarity_classifier.model_mgr import SentimentAnalysisPolarityClassifierMgr
from config.config import ASPECT_CLASSIFICATION_MODEL_PATH
from src.models.aspect_extraction_classifier.model_mgr import AspectClassifierManager
from spacy.tokens.span import Span
import logging
logger = logging.getLogger(__name__)


def get_aspect_sentences(text_review, word_ids):
    aspect_sentences = []
    if word_ids:
        for word_id in word_ids:
            aspect_sentence = word_id_sentence_getter(text_review,
                                                      word_id=word_id,
                                                      n=7,
                                                      i_s=word_ids)
            aspect_sentences.append(aspect_sentence)
    else:
        aspect_sentences = [text_review]
    return aspect_sentences


class AspectSentimentManager:
    def __init__(self):
        self.sentiment_module: SentimentAnalysisPolarityClassifierMgr = None
        self.aspect_module: AspectSentimentManager = None
        self.aspect_module_classifier: AspectClassifierManager = None
        self.window: int = 6
        self.sentiment_neutral_threshold = 0.7
        self.sentiment_neutral_label = 'neutral'
        self.prediction = None
        self._load_aspect_extraction_model()
        self._load_sentiment_analysis_model()
        self._load_aspect_classifier_model()

    def _load_aspect_classifier_model(self):
        self.aspect_module_classifier = AspectClassifierManager(model_path=ASPECT_CLASSIFICATION_MODEL_PATH)

    def _load_aspect_extraction_model(self):
        self.aspect_module = AspectExtractorMgr(window=self.window)

    def _load_sentiment_analysis_model(self):
        self.sentiment_module = SentimentAnalysisPolarityClassifierMgr(production_mode=True)

    def _prediction_threshold(self, raw_prediction: List[Tuple[str, int]]) -> str:
        label, prob = raw_prediction[0]
        return label if prob >= self.sentiment_neutral_threshold else self.sentiment_neutral_label, prob

    @staticmethod
    def _get_aspect_text(aspect: Span) -> str:
        return aspect['aspect_sentence'].text

    def _get_sentiment_polarity(self, aspect: Span, text_sentence=None) -> str:
        if text_sentence is None:
            text = self._get_aspect_text(aspect)
        else:
            text = text_sentence
        raw_prediction = self.sentiment_module.predict(text=text, raw=True)
        polarity_label, prob = self._prediction_threshold(raw_prediction)
        return polarity_label, prob

    @staticmethod
    def _flatten_prediction(aspects) -> dict:
        aspects_ = {}
        aspect_ = {}
        aspects_list = []
        polarity_list = []
        aspects_aux = []

        for aspect in aspects:
            aspect_ = aspect.copy()
            aspects_list.append(aspect_['aspect'])
            polarity_list.append(aspect_['sentiment'])

        aspects_['id_review'] = [aspect_['id_review']]
        aspects_['review'] = [aspect_['review']]
        aspects_['aspects'] = [aspects_list]  # to str in order to conserve data in same row when parse to df
        aspects_['polarity'] = [polarity_list]  # to str in order to conserve data in same row when parse to df
        aspects_['aspects_aux'] = [aspect_['aspects_aux']]
        aspects_['polarity_aux'] = [aspect_['polarity_aux']]

        return aspects_

    def predict(self, text_review: str, id_review: int = 0, flatten: bool = False) -> pd.DataFrame:

        aspects = self.aspect_module.extract_aspect(text_review)

        for aspect in aspects:
            aspect['id_review'] = id_review
            aspect['review'] = text_review
            aspect['sentiment'], aspect['sentiment_prob'] = self._get_sentiment_polarity(aspect)
            aspects_aux, first_word_ids = self.aspect_module_classifier.predict(text=[text_review])
            sentences_aux = get_aspect_sentences(text_review=text_review, word_ids=first_word_ids)
            aspect['aspects_aux'] = aspects_aux
            sentiments_and_probs = [self._get_sentiment_polarity(aspect=None, text_sentence=text.text)
                                                                 for text in sentences_aux]
            aspect['polarity_aux'] = [polarity for polarity, prob in sentiments_and_probs]
            aspect['polarity_prob_aux'] = [prob for polarity, prob in sentiments_and_probs]

        if flatten:
            aspects = self._flatten_prediction(aspects)

        return pd.DataFrame(aspects)

    def predict_batch(self, text_reviews: List[str], flatten: bool) -> pd.DataFrame:

        output = pd.DataFrame()

        for id_review, text_review in tqdm(enumerate(text_reviews)):
            try:
                output = output.append(self.predict(text_review=text_review, id_review=id_review, flatten=flatten))
            except:
                logger.info(f'Exception in prediction of id_review: {id_review} with text_review: {text_review}')

        return output


if __name__ == "__main__":

    review_batch = [
                    'The food we had yesterday was delicious. The breakfast was delivered late',
                    'My time in Italy was very enjoyable',
                    'I found the meal to be tasty',
                    'The internet connection was terribly slow.',
                    'Our experience was suboptimal',
                    'It was a surprise, the screen is not bad',
                    'Not sure if the movie sound was the most suitable one',
                    'Air france food is delicious',
                    'I think Gabriel García Márquez is not boring. The first book I read was funny'
                    ]

    aspect_sentiment_manager = AspectSentimentManager()
    df_aspect_sentiment = aspect_sentiment_manager.predict(text_review=review_batch[0], id_review=0)
    df_aspects_sentiment = aspect_sentiment_manager.predict_batch(text_reviews=review_batch, flatten=False)
    df_aspects_sentiment_flatten = aspect_sentiment_manager.predict_batch(text_reviews=review_batch, flatten=True)