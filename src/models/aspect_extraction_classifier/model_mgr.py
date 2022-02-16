import os
from src.models.aspect_extraction_classifier.data_parser import parse_data, data_reader_aspect_extraction, \
    dataset_length_filter
from src.models.common.utils_train import data_partitions
from src.models.aspect_extraction_classifier.pre_processing import token_tokenizer, build_enconde_dataset
from src.models.aspect_extraction_classifier.constants import AspectDatasetColumns, AspectDatasetFields
from src.models.aspect_extraction_classifier.eval_utils import compute_metrics
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from config.config import COMMON_MODELS, ASPECT_CLASSIFICATION_MODEL_PATH
from src.models.aspect_extraction_classifier.pre_processing import pre_process_text
import torch
import logging

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


class AspectClassifierManager:
    def __init__(self, model_path=None, production_mode=True):
        self.production_mode = production_mode
        self.model_path = ASPECT_CLASSIFICATION_MODEL_PATH if model_path is None else model_path
        self.model_checkpoint = "roberta-base"
        self.subword_token_threshold = 32
        self.model_dropout = 0.15
        self.max_length = 32
        self.tokenizer_config = {'max_length': 32,
                                 'truncation': True,
                                 'padding': 'max_length',
                                 'is_split_into_words': True,
                                 'add_prefix_space': True  # automatize add_prefix_space for roberta
                                 }
        self.model_status = self.get_model_status()
        self.args = TrainingArguments(
                    output_dir=COMMON_MODELS['temp'],
                    evaluation_strategy="steps",
                    learning_rate=7e-5,
                    per_device_train_batch_size=64,
                    per_device_eval_batch_size=64,
                    warmup_steps=400,
                    num_train_epochs=3,
                    weight_decay=0.08,
                    logging_steps=50,
                    lr_scheduler_type='cosine',
                    seed=42,
                    logging_dir=COMMON_MODELS['temp'],
                    load_best_model_at_end=True)

        self.model = self._get_model()
        self.tokenizer = self.get_tokenizer()

    def _get_model(self):
        if self.model_status:
            model_ref = self.model_path
        else:
            model_ref = self.model_checkpoint
        return self.get_model(model_ref)

    def get_model_status(self):
        if os.path.isdir(self.model_path):
            logger.info(f"Was found a trained model on {self.model_path}")
            return True
        else:
            logger.info(f"Was not found a trained model on {self.model_path}")
            return False

    def get_tokenizer(self):
        if self.model_status:
            model_ref = self.model_path
        else:
            model_ref = self.model_checkpoint
        return token_tokenizer(model_checkpoint=model_ref, add_prefix_space=self.tokenizer_config['add_prefix_space'])

    def get_model(self, model_ref):
         return AutoModelForTokenClassification.from_pretrained(model_ref,
                                                                num_labels=len(AspectDatasetFields.label_list),
                                                                attention_probs_dropout_prob=self.model_dropout,
                                                                hidden_dropout_prob=self.model_dropout).to(DEVICE)

    def _data_loader(self):
        df_aspects = data_reader_aspect_extraction(verbose=True)
        df_aspects = parse_data(df_aspects_raw=df_aspects, verbose=True)
        df_aspects = dataset_length_filter(df=df_aspects,
                                           subword_token_threshold=self.subword_token_threshold,
                                           tokenizer=self.tokenizer)
        return df_aspects

    def train(self):
        logger.info(f"Training model using {self.model_checkpoint}")
        if self.model_status:
            raise Exception(f"You have a trained model on {self.model_path}, if you want to train a new one"
                            f"change the model_path or delete the existing model file")
        self.model = self.get_model(model_ref=self.model_checkpoint)
        df_aspects = self._data_loader()

        stratify = df_aspects.aspect.apply(
            lambda x: AspectDatasetFields.aspect_flag_false if x == [AspectDatasetFields.empty_tokens] else
            AspectDatasetFields.aspect_flag_true).to_list()

        df_aspects_train, df_aspects_test, unique_labels = data_partitions(
            data=df_aspects,
            production_mode=self.production_mode,
            target_label_name=AspectDatasetColumns.label_column,
            stratify=stratify,
            token_level=True)

        df_test = build_enconde_dataset(df_partition=df_aspects_test,
                                        tokenizer=self.tokenizer,
                                        tokenizer_config=self.tokenizer_config)
        df_train = build_enconde_dataset(df_partition=df_aspects_train,
                                         tokenizer=self.tokenizer,
                                         tokenizer_config=self.tokenizer_config)
        trainer = Trainer(self.model,
                          self.args,
                          train_dataset=df_train,
                          eval_dataset=df_test,
                          compute_metrics=compute_metrics)
        trainer.train()
        trainer.evaluate()

        trainer.save_state()
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        self.model_status = self.get_model_status()
        logger.info(f"Model was trained and saved on {self.model_path}")

        return self.model

    @staticmethod
    def join_composed_aspects(found_aspects_word_ids, text_tokens):
        first_word_id = []
        if found_aspects_word_ids == []:
            found_aspects = ['None']
            first_word_id = [0]
        else:
            found_aspects = []
            last_aspect_word_id = 0
            text_tokens_ = text_tokens[0].copy()
            word_ids = []
            for relevant_label, word_id in found_aspects_word_ids:
                word_ids.append(word_id)
                aspect = text_tokens_[word_id]
                # if is a composed aspect
                if (word_id - last_aspect_word_id) == 1 and relevant_label == AspectDatasetFields.tags_encoder['I']:
                    found_aspects[-1] = found_aspects[-1] + f' {aspect}'
                else:
                    if word_id not in first_word_id:
                        first_word_id.append(word_id)
                        found_aspects.append(aspect)  # if word is repeated in other word_id it will appear more than 1
                last_aspect_word_id = word_id
        return found_aspects, first_word_id

    def predict(self, text):
        if self.model_status:
            tokens, text_tokens = pre_process_text(
                                  texts=text,
                                  tokenizer=self.tokenizer,
                                  tokenizer_config=self.tokenizer_config,
                                  text_tokens=True,
                                  )

            word_ids = tokens.word_ids()
            # raw_predictions = self.model(**tokens)
            raw_predictions = self.model(**tokens.convert_to_tensors(tensor_type='pt').to(DEVICE))
            predictions_probs = raw_predictions.logits.softmax(dim=2)
            labels_ = predictions_probs.argmax(dim=2).tolist()[0]
            # tokens_ = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'].tolist()[0])
            relevant_labels = [(labels_[word_idx], word_id) for word_idx, word_id in enumerate(word_ids)
                               if word_id is not None]  # --> list((label,word_id)) ->[(0, 0), (0, 1), (0, 2), (1, 3),..
            found_aspects_word_ids = [(relevant_label, word_id) for relevant_label, word_id in relevant_labels if
                                      AspectDatasetFields.tags_decoder[relevant_label]!="O"]
            found_aspects, first_word_ids = self.join_composed_aspects(found_aspects_word_ids, text_tokens)

        else:
            raise Exception(f"Not trained models available on {self.model_path}, please execute self.train() to build"
                            f"a model")

        return found_aspects, first_word_ids


if __name__ == "__main__":

    aspect_classifier_mgr = AspectClassifierManager(production_mode=True,
                                                    model_path=ASPECT_CLASSIFICATION_MODEL_PATH)
    #  model_path= 'temp/aspect_classifier'
    #  aspect_classifier_mgr.train()

    reviews = ['I like the mobile screen but I hate the buttons',
               'The food was amazing but the drink was very cool',
               'The soap smell good and the colour is nice',
               'The museum music was amazing, and so on the ticket price, it was so cheap!',
               'Wow amazing night. In love with the scenario lights and security staff',
               'The Peabody is top of the line in terms of accomodations and service Wonderful beds, linens, '
               'furnishings The daily Duck March is an absolute for the kids to see We would stay there any time!']

    for review in reviews:
        print(f'For sentence {review} we found the following aspects: '
              f'--{aspect_classifier_mgr.predict(text=[review])}--')
