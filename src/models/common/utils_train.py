import os
import pandas as pd
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from src.models.common.encoders_decoders import LabelsMgr
from src.utils import set_seed
from config.config import COMMON_MODELS
from src.models.common.metrics import compute_metrics_multiclass
from src.models.common.constants import FileNames
from itertools import chain

"""
Note: By default this script will perform training for production "MASTER BRANCH", using almost all the available data 
for training. Before running this script on production you have to test the same model configuration with the same 
dataset using dev/test/train partitions on "development" mode to guarantee the model on production will be enough good.
Set the variable production_mode = False for testing purpose and production_mode = True for production env.
"""
SEED = 1
set_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataSetParser(torch.utils.data.Dataset):
    """
    Class to parse data set in torch Dataset structure in order to automatize
    the data management on Trainer methods.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def data_partitions(data: pd.DataFrame,
                    production_mode: bool = True,
                    target_label_name='target_column',
                    stratify=None,
                    token_level=False):
    if token_level:
        unique_labels = set(chain(*data[target_label_name].to_list()))
    else:
        unique_labels = set(data[target_label_name])
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=100, stratify=stratify)
    if production_mode:
        train_data = data.copy()
    return train_data, valid_data, unique_labels


def set_up_model_bert_model_trainer(config):

    training_args = TrainingArguments(
        output_dir=COMMON_MODELS['temp'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        warmup_steps=config['warmup_steps'],  # number of warmup steps for learning rate scheduler
        weight_decay=config['weight_decay'],  # strength of weight decay
        logging_dir=COMMON_MODELS['logs'],  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        logging_steps=config['logging_steps'],  # log & save weights each logging_steps
        evaluation_strategy="steps",  # steps or epoch -> evaluate each `logging_steps`
        learning_rate=config['learning_rate'],  # learning rate from 0 to 1
        save_steps=config['logging_steps'],
        seed=SEED
    )

    return training_args


def data_encoder(tokenizer, train_data, valid_data, max_length, label_mgr, target_label_name, text_label_name):
    # Involves -> __call__(text: Union[str, List[str], List[List[str]]], text_pair: Optional[Union[str,
    # List[str], List[List[str]]]] = None,

    train_labels = label_mgr.encoder(train_data[target_label_name])

    valid_labels = label_mgr.encoder(valid_data[target_label_name])

    train_encodings = tokenizer(text=train_data[text_label_name].to_list(), truncation=True, padding=True,
                                max_length=max_length)
    valid_encodings = tokenizer(text=valid_data[text_label_name].to_list(), truncation=True, padding=True, max_length=max_length)

    return train_encodings, valid_encodings, train_labels, valid_labels


def build_classifier(model, tokenizer, train_data, valid_data, label_mgr, max_length, config, target_label_name,
                     text_label_name):

    train_encodings, valid_encodings, train_labels, valid_labels = data_encoder(tokenizer, train_data,
                                                                                valid_data, max_length,
                                                                                label_mgr=label_mgr,
                                                                                target_label_name=target_label_name,
                                                                                text_label_name=text_label_name)
    train_dataset = DataSetParser(train_encodings, train_labels)
    valid_dataset = DataSetParser(valid_encodings, valid_labels)

    training_args = set_up_model_bert_model_trainer(config)

    trainer = Trainer(
                      model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=valid_dataset,
                      compute_metrics=compute_metrics_multiclass,
                     )

    return trainer


def training_metrics_getter(trainer):
    metrics_validation = trainer.evaluate()
    metrics_train = trainer.state.log_history
    metrics_summary = {'validation': metrics_validation, 'train': metrics_train}
    return metrics_summary


def save_model_configuration(trainer, model, tokenizer, label_name_map: dict,
                             model_data_frame: pd.DataFrame,
                             metrics_summary: dict,
                             model_path: str):

    sep = ';'

    # Save training steps on log
    trainer.save_state()  # --> log available on /WeVoiceLabML/logs/trainer_state.json
    trainer.save_metrics(metrics=metrics_summary['validation'], split='all',
                         combined=True)  # --> log available on /WeVoiceLabML/logs/all_results.json

    # Save model and tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # json.dumps(labels_map, sort_keys=True, indent=4)
    with open(os.path.join(model_path, FileNames.LABEL_MAP), 'w') as fp:
        json.dump(label_name_map, fp)

    with open(os.path.join(model_path, FileNames.METRICS), 'w') as fp:
        json.dump(metrics_summary, fp)

    train_dataset_filepath = os.path.join(model_path, FileNames.TRAINING_DATASET)
    if not os.path.isfile(train_dataset_filepath):
        model_data_frame.to_csv(train_dataset_filepath, sep=sep)


def build_model_components(config, model, tokenizer, data, max_length_tokenization=16, production_mode=True,
                           target_label_name=None, text_label_name=None):

    train_data, valid_data, unique_labels = data_partitions(data=data, production_mode=production_mode,
                                                            target_label_name= target_label_name)
    label_mgr = LabelsMgr(unique_labels=unique_labels)
    trainer = build_classifier(model=model, tokenizer=tokenizer, train_data=train_data, valid_data=valid_data,
                               label_mgr=label_mgr,
                               max_length=max_length_tokenization, config=config, target_label_name=target_label_name,
                               text_label_name=text_label_name)

    return trainer, label_mgr


if __name__ == "__main__":
    pass
