import os
import torch
import logging
from transformers import AutoModelForSequenceClassification
from src.models.common.pre_processing import auto_tokenizer
from typing import List, Union
from src.models.common.model_mgr import ClassifierMgr
from src.models.sentiment_polarity_classifier.data_parser import data_reader_sentiment_analysis
from config.config import SENTIMENT_ANALYSIS_POLARITY_CLASSIFICATION_CONFIG
from src.constants import DataStructure
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


class SentimentAnalysisPolarityClassifierMgr(ClassifierMgr):

    def __init__(self, model_path: str = None, production_mode: bool = True):
        self.model_path = SENTIMENT_ANALYSIS_POLARITY_CLASSIFICATION_CONFIG if \
            model_path is None else model_path
        self.production_mode = production_mode
        self.model_label_mgr = None
        self.model_classifier = None
        self.model_tokenizer = None
        self.model_loaded_status = False
        self.model_name = 'roberta-base'
        self.max_length = 256
        self.target_label_name = DataStructure.POLARITY
        self.text_label_name = DataStructure.REVIEW
        self.config = {
            'num_train_epochs': 3,
            'per_device_train_batch_size': 32,
            'per_device_eval_batch_size': 32,
            'warmup_steps': 2000,
            'weight_decay': 0.01,
            'logging_steps': 2000,
            'learning_rate': 5e-05,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'layers_freeze': None}

        if os.path.isdir(self.model_path):
            self.set_pretrained_model()

    def _model_loader(self, num_labels, model_name=None):
        """
        Load from disc existing model
        :param num_labels:
        :return:
        """
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_path if model_name is None else model_name,
            num_labels=num_labels).to(DEVICE)

    @staticmethod
    def _data_loader(raw_dataset_path, training_dataset):
        """
        Create the train dataset por the intent classifier model.
        If raw_dataset_path is None the training_dataset will be generate from the raw data available on
        data/interim/intent_detection.csv.
        If the training_dataset is the input it will be used for training.
        :param raw_dataset_path:
        :param training_dataset:
        :return:
        """
        if training_dataset is not None:
            dataset = training_dataset
        elif raw_dataset_path is None:
            raw_dataset_path = 'data/processed/reviews_dataset.csv'
            dataset = data_reader_sentiment_analysis(file_path=raw_dataset_path)
        elif raw_dataset_path is not None:
            dataset = data_reader_sentiment_analysis(file_path=raw_dataset_path)
        else:
            raise FileExistsError

        return dataset

    def _tokenizer_loader(self, model_name=None):
        return auto_tokenizer(pretrained_model_name_or_path=self.model_path if model_name is None
                              else model_name)

    def predict(self, text: str, raw: bool = False) -> Union[List[str], List[tuple]]:
        """
        Perform a prediction of sentiment polarity for class negative and positive from text.
        The model will predict one the following labels:
        --> 'positive' or 'negative'.
        self.predict(text='I am so happy in my new job', raw=1))
        >> [('positive',0.9933562278747559)]
        self.predict(text='I am so happy in my new job', raw=0))
        >> ['positive']
        :param text: text string to evaluate
        :param raw: flag to get probabilities if true/1 or false/0 default to get just the predicted categories
        :return:
        """

        output = ['None']

        if self.model_loaded_status:
            try:
                text_tokens = self.model_tokenizer([text],
                                                   padding=True,
                                                   truncation=True,
                                                   max_length=self.max_length,
                                                   return_tensors="pt").to(DEVICE)
                predictions = self.model_classifier(**text_tokens)
                probabilities = predictions[0].softmax(1)
                label_encoded = probabilities.argmax(axis=1).tolist()
                label_decoded = self.model_label_mgr.decoder(label_encoded)

                if raw:
                    output = [(label_decoded[0], probabilities.tolist()[0][label_encoded[0]])]
                else:
                    output = label_decoded

                logger.info(f'Inference of text segment {text} performed using model available on {self.model_path} '
                            f'predicting: {output} ')
            except ValueError:
                logger.info(f'{type(text)} is not was no valid vale, please introduce string type value')

        else:
            logger.info(f'model {self.model_path} was not loaded')
            raise FileExistsError

        return output


if __name__ == "__main__":

    sa_model = SentimentAnalysisPolarityClassifierMgr(production_mode=False)
    sa_model.train()
    sa_model.predict(text='I really like the screen but I hate the battery')



