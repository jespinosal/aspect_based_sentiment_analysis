import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from src.models.aspect_extraction_classifier.constants import AspectDatasetFields, AspectDatasetColumns
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize_and_align_labels(examples,
                              input_column,
                              label_column,
                              label_all_tokens,
                              max_length=None,
                              padding=False,
                              truncation=False,
                              tokenizer=None):
    """
     Source: HF
     This method add a new key in the tokenized_inputs dict with the allinged
     lables using the tokenier method word_ids(). You can choose to methods to
     to alling the label, ignoring/considering the subwords with same word_id.
     For instance:
     >> tags.    = [0, 1, 0, 0, 1]
     >> word_ids =                 [None, 0, 1,    1,    1, 2, 3, 4, None]
     >> labels_all_tokens_true   = [-101, 0, 1,    1,    1, 0, 0, 1, -101]
     >> labels_all_tokens_false  = [-101, 0, 1, -101, -101, 0, 0, 1, -101]
    """
    tokenized_inputs = tokenizer(examples[input_column].to_list(), truncation=truncation,
                                 is_split_into_words=True,  # to support token tokenizer
                                 padding=padding, max_length=max_length)
    #  ,return_tensors="pt").to("cuda")
    labels = []
    for i, label in enumerate(examples[label_column].to_list()):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class AspDat(Dataset):
    def __init__(self, items):
        """
        items is a tokenized object type(items_train)
        :param items:
        """
        self.items = items

    def __len__(self):
        """
        Return the amount of samples of items
        len(items['input_ids']) == len(items.input_ids)
        :return:
        """
        return len(self.items.input_ids)

    def __getitem__(self, idx):
        """
        Return
        :param idx:
        :return:
        """
        return {k: torch.tensor(v[idx]) for k, v in self.items.items()}


def token_tokenizer(model_checkpoint, add_prefix_space):
    # model_name = model_checkpoint.split("/")[-1]
    # add_prefix_space = True if "roberta" in model_name else False @todo automatize add_prefix_space setting
    token_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=add_prefix_space)
    return token_tokenizer


def build_enconde_dataset(df_partition, tokenizer, tokenizer_config):
    items_partition = tokenize_and_align_labels(
                  examples=df_partition,
                  input_column=AspectDatasetColumns.input_column,
                  label_column=AspectDatasetColumns.label_column,
                  label_all_tokens=True,
                  max_length=tokenizer_config['max_length'],
                  padding=tokenizer_config['padding'],
                  truncation=tokenizer_config['truncation'],
                  tokenizer=tokenizer)
    df_aspect_torch = AspDat(items_partition)
    return df_aspect_torch


def pre_process_text(texts, tokenizer, tokenizer_config, text_tokens=False):
    """
    Use text_tokens = True if the attribute text is a list of strings. If is a list of tokens
    use text_tokens = False.
    :param texts:
    :param tokenizer:
    :param tokenizer_config:
    :param text_tokens:
    :return:
    """
    if text_tokens:
        text_tokens = [nltk.tokenize.word_tokenize(text) for text in texts]
    else:
        text_tokens = texts
    tokenized_inputs = tokenizer(text_tokens,
                                 truncation=tokenizer_config['truncation'],
                                 is_split_into_words=tokenizer_config['is_split_into_words'],
                                 padding=tokenizer_config['padding'],
                                 max_length=tokenizer_config['max_length'],
                                 )#.to(DEVICE) #return_tensors='pt'
    return tokenized_inputs, text_tokens


if __name__ == "__main__":
    pass
