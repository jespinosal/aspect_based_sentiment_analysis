"""
Define base structure for the model management
"""

import os
import json
import torch
import logging
import pandas as pd
from abc import ABC
import abc
from transformers.modeling_outputs import SequenceClassifierOutput
from src.models.common.encoders_decoders import LabelsMgr
from src.models.common.utils_train import build_model_components, save_model_configuration, \
    training_metrics_getter
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from src.models.common.pre_processing import auto_tokenizer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


class ClassifierMgr(ABC):

    def __init__(self, model_path: str = None,
                 production_mode: bool = None,
                 model_name: str = None,
                 max_length: int = None,
                 target_label_name: str = None,
                 text_label_name: str = None,
                 config: dict = None):

        self.model_path: str = model_path
        self.production_mode: bool = production_mode
        self.model_name: str = model_name
        self.max_length: int = max_length
        self.target_label_name: str = target_label_name
        self.text_label_name: str = text_label_name
        self.config = dict()
        self.config['num_train_epochs']: int = config.get('num_train_epochs', None)
        self.config['per_device_train_batch_size']: int = config.get('per_device_train_batch_size', None)
        self.config['per_device_eval_batch_size']: int = config.get('per_device_eval_batch_size', None)
        self.config['warmup_steps']: int = config.get('warmup_steps', None)
        self.config['weight_decay']: float = config.get('weight_decay', None)
        self.config['logging_steps']: int = config.get('logging_steps', None)
        self.config['learning_rate']: float = config.get('learning_rate', None)
        self.config['hidden_dropout_prob']: float = config.get('hidden_dropout_prob', None)
        self.config['attention_probs_dropout_prob']: float = config.get('attention_probs_dropout_prob', None)
        self.config['layers_freeze']: int = config.get('layers_freeze', None)
        self.model_label_mgr: LabelsMgr = None
        self.model_loaded_status: bool = False
        self.trainer: Trainer = None
        self.train_metrics: dict = None
        self.model_classifier = None
        self.model_tokenizer = None

        if os.path.isdir(self.model_path):
            self.set_pretrained_model()

    @staticmethod
    @abc.abstractmethod
    def _data_loader(raw_dataset_path, training_dataset) -> pd.DataFrame:
        """
        Create the train dataset for classification model.
        """
        return

    def _model_loader(self, num_labels, model_name=None):
        """
        Load from disc existing model
        :param num_labels:
        :return:
        """
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_path if model_name is None else model_name,
            num_labels=num_labels).to(DEVICE)

    def _tokenizer_loader(self, model_name=None):
        return auto_tokenizer(pretrained_model_name_or_path=self.model_path if model_name is None
                              else model_name)


    def _get_raw_probabilities(self, predictions: SequenceClassifierOutput, softmax: bool = True):
        """
        Given the array of probabilities of pytorch models parse the
        :param predictions:
        :return:
        """
        output_raw = {}
        if softmax:
            probabilities = predictions[0].softmax(1)
        else:
            probabilities = predictions[0].sigmoid()

        for index_, prob_ in enumerate(probabilities.tolist()[0]):
            label_ = self.model_label_mgr.decoder([index_])[0]
            output_raw[label_] = prob_

        sorted_tuples = sorted(output_raw.items(), key=lambda kv: kv[1], reverse=True)

        return sorted_tuples

    @abc.abstractmethod
    def predict(self, text):
        """
        Perform prediction over text sentence
        :param text:
        :return:
        """
        return

    def set_pretrained_model(self):
        self.model_label_mgr = self.__load_labels_map()
        self.model_classifier = self._model_loader(num_labels=len(self.model_label_mgr))
        self.model_tokenizer = self._tokenizer_loader()
        self.model_loaded_status = True
        logger.info(f'Loaded pre trained model from {self.model_path}')

    def __load_labels_map(self):
        label_map_file_name = 'labels_map.json'
        with open(os.path.join(self.model_path, label_map_file_name), 'r') as fp:
            labels_map = json.load(fp)

        return LabelsMgr(labels_map=labels_map)

    def __model_save(self, trainer, model, tokenizer, label_mgr, train_data, train_metrics):
        logger.info(f'Save model components on {self.model_path}')
        return save_model_configuration(trainer=trainer, model=model, tokenizer=tokenizer,
                                        label_name_map=label_mgr.label_name_map,
                                        model_data_frame=train_data,
                                        metrics_summary=train_metrics,
                                        model_path=self.model_path,
                                        )

    def get_labels_length(self, data):
        return len(set(data[self.target_label_name]))

    def model_modifier(self, model):
        """
        This model implement changes to modify model paramaters such as:
        - hidden_dropout_prob:
        - attention_probs_dropout_prob:
        - embeddings layer:
        - encoder layers:

        If hidden_dropout_prob and attention_probs_dropout_prob is not defined will use the default values 0.1.
        If layers_freeze is defined the model will freeze since first layer to freeze_layer_count layer in the encoder
        of the current base_model. IF layers_freeze=0  only freeze embeddings.

        :param model:
        :return:
        """

        if self.config is not None:
            model.config.hidden_dropout_prob = self.config.get('hidden_dropout_prob', 0.1)
            model.config.attention_probs_dropout_prob = self.config.get('attention_probs_dropout_prob', 0.1)
            freeze_layer_count = self.config.get('layers_freeze', None)  # -> 0, 2, 4, 6, 8
            logger.info(f'model dropout props hidden = {model.config.hidden_dropout_prob}'
                        f'and attention_probs {model.config.attention_probs_dropout_prob}')

        if freeze_layer_count is not None:
            logger.info(f'Freeze embeddings layer and encoder layers from 0 to {freeze_layer_count} layer ')
            # We freeze here the embeddings of the model

            for param in model.base_model.embeddings.parameters():
                param.requires_grad = False

            if freeze_layer_count >= 1:
                # if freeze_layer_count <= 1 (for instance "0"), we only freeze the embedding layer
                # otherwise we freeze since the first `freeze_layer_count` encoder layers to freeze_layer_count layer
                for layer in model.base_model.encoder.layer[:freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False
        return model

    def train(self, raw_dataset_path: str = None, training_dataset: pd.DataFrame = None) -> None:
        """
        Given a train_data and model_path this method will perform a model training saving all the model components
        on model_path. Using the transformer model model_name. production_mode: Flag to indicate if model will be
        trained with all data or some partition. Production mode = TRUE involves training with all the available data.
        FALSE involves training with a data partition.
        :param raw_dataset_path: mix of dataset from different sources most of them from home assistants
        :param training_dataset: the dataset available on raw_dataset_path must be transformed in training_dataset
        :return: None
        """
        train_data = self._data_loader(raw_dataset_path, training_dataset)
        num_labels = self.get_labels_length(data=train_data)
        model = self._model_loader(model_name=self.model_name, num_labels=num_labels)
        model = self.model_modifier(model)

        tokenizer = self._tokenizer_loader(model_name=self.model_name)
        self.trainer, label_mgr = build_model_components(config=self.config,
                                                         model=model,
                                                         tokenizer=tokenizer,
                                                         data=train_data,
                                                         max_length_tokenization=self.max_length,
                                                         production_mode=self.production_mode,
                                                         target_label_name=self.target_label_name,
                                                         text_label_name=self.text_label_name)
        _ = self.trainer.train()

        self.train_metrics = training_metrics_getter(self.trainer)

        self.__model_save(trainer=self.trainer, model=model, tokenizer=tokenizer,
                          label_mgr=label_mgr,
                          train_data=train_data,
                          train_metrics=self.train_metrics)

        self.set_pretrained_model()  # --> set model reading the save model on self.model_path

        logger.info(f'Model training performed, model available on {self.model_path}')
        return model


if __name__ == "__main__":
    pass


