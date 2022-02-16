import pandas as pd
import ast
import nltk
from collections import Counter
from src.constants import DataStructure
from config.config import ASPECT_DATA_SET_FILE_PATH
from src.text_processing import word_tokenizer_basic
from src.models.aspect_extraction_classifier.constants import AspectDatasetColumns, AspectDatasetFields
from src.models.aspect_extraction_classifier.pre_processing import pre_process_text

def impute_empty_annotations(df_aspect_dataset):
    """
    Fill in empty aspect annotations using ['NULL']
    :param df_aspect_dataset:
    :return:
    """
    df_aspect_dataset = df_aspect_dataset.copy()
    empty = []
    null = 'NULL'
    df_aspect_dataset[DataStructure.ASPECT] = df_aspect_dataset[DataStructure.ASPECT].apply(
        lambda x: [null] if x == empty else x)
    return df_aspect_dataset


def eval_implicit(aspect):
    return False if (len(aspect) > 1) and ('NULL' in aspect) else True


def set_bio_tags(sentence, aspects, verbose=True):
    """
    Implement a BIO tagging for aspect extraction.
    Scope: match with first entity occurrence, if entity appear more than once will take first
    occurrence, if aspect contain more than one word and first word match with another word is not an entity
    this sample will be discarded.

    outputs examples:
    >>['aspect_implicit']: mix of nulls with aspects
    >>[O, O, O, O, O, O]: non aspect
    >>[O, O, B ,O, O, O]: aspect match
    >>[O, O, B ,I, O, O]: aspect composed match
    >>['implicit_aspect']: implicit aspect in the sentence
    >>['corrupted']: aspect is not in the text
    >>['aspect_multi_occurrence']: aspects appear more than once and do not match the first occurrence

    :param sentence:
    :param aspects:
    :param verbose:
    :return:
    """

    empty_tokens = AspectDatasetFields.empty_tokens
    tags = AspectDatasetFields.tags
    word_tokens = word_tokenizer_basic(sentence)
    #  word_tokens = [basic_pre_processing(word_token) for word_token in word_tokens]
    #  [word_tokenizer_basic(basic_pre_processing(aspect)) for aspect in aspects]
    aspects_tokens_clean = [word_tokenizer_basic(aspect) for aspect in aspects]
    labels = [tags["non_aspect"]]*len(word_tokens)
    aspects_valid = aspects.copy()
    aspects_size = len(aspects_tokens_clean)

    # Eval corner cases of aspect tags, when Null and empty cases.
    if aspects_size:
        null_aspects = [True if aspect[0] in empty_tokens else False for aspect in aspects_tokens_clean]
        if any(null_aspects):
            if len(null_aspects) == 1:
                return labels
            else:
                return ['implicit_aspect']
        else:
            pass  # perform calculation above  if not nulls and not empty
    else:
        return labels

    # If review has aspects (without Nulls) we will annotate the BIO position
    # Rejecting some corner cases.
    for aspect in aspects_tokens_clean:
        aspect_size = len(aspect)
        if aspect_size == 1:
            try:
                idx = word_tokens.index(aspect[0])
                labels[idx] = tags["beginning_aspect"]
            except:
                aspects_valid.pop(0)
                if verbose:
                    print("aspect not found in sentence")
                return ['corrupted']
        else:  # aspect_size>1
            idxs = []
            for aspect_token in aspect:
                try:
                    idx_ = word_tokens.index(aspect_token)
                    idxs.append(idx_)
                except:
                    if verbose:
                        print("compose aspect not found sentence")
                    return ['corrupted']

                if len(idxs) == aspect_size:
                    idxs_expected = list(range(min(idxs), max(idxs) + 1))
                    if idxs_expected == idxs:
                        for idx in idxs:
                            last_tag_idx = idx - 1 if idx - 1 >= 0 else 0
                            last_tag = labels[last_tag_idx]
                            if last_tag in [tags['beginning_aspect'], tags['inside_aspect']]:
                                labels[idx] = tags['inside_aspect']
                            else:
                                labels[idx] = tags['beginning_aspect']
                    else:
                        if verbose:
                            print("aspect words are spread in the sentences")
                        return ['aspect_multi_occurrence']
    return labels


def data_reader_aspect_extraction(file_path=None, verbose=True):
    """
    Read the data set of aspect extraction and assign the corresponding BIO tags
    :return:
    """
    valid_columns = [DataStructure.REVIEW, DataStructure.ASPECT]
    file_path = ASPECT_DATA_SET_FILE_PATH if file_path is None else file_path
    df_aspect_dataset = pd.read_csv(file_path, sep=';')
    if verbose:
        print(f'Raw dataset size {df_aspect_dataset.__len__()}')
    df_aspect_dataset = df_aspect_dataset[valid_columns]
    df_aspect_dataset[DataStructure.ASPECT] = df_aspect_dataset[DataStructure.ASPECT].apply(lambda x:
                                                                                            ast.literal_eval(x))
    df_aspect_dataset = impute_empty_annotations(df_aspect_dataset=df_aspect_dataset)

    df_aspect_dataset = df_aspect_dataset.reset_index()
    del df_aspect_dataset['index']
    if verbose:
        print(f'Raw dataset size {df_aspect_dataset.__len__()}')
    df_aspect_dataset['annotations'] = df_aspect_dataset.apply(lambda row: set_bio_tags(sentence=row.review_text,
                                                                                        aspects=row.aspect,
                                                                                        verbose=verbose), axis=1)

    if verbose:
        annotations_occurrences = Counter([str(sample) for sample in df_aspect_dataset.annotations.to_list()])
        print("corrupted", annotations_occurrences["['corrupted']"])  # 267 -61
        print("aspect_multi_occurrence", annotations_occurrences["['aspect_multi_occurrence']"])  # 383 -399
        print("implicit_aspect", annotations_occurrences["['implicit_aspect']"])  # 0 - 274

    return df_aspect_dataset


def parse_data(df_aspects_raw, verbose=True):
    """
    Implement data filtering and transformation in the token structure according the needs of the token classification
    algorithms.
    :param df_aspects_raw:
    :param verbose:
    :return:
    """
    df_aspects = df_aspects_raw.copy()

    df_aspects = df_aspects[~df_aspects[AspectDatasetColumns.annotation_column].apply(
        lambda x: x in AspectDatasetFields.annotations_tricky)]

    if verbose:
        print(f'Deleted {len(df_aspects_raw)-len(df_aspects)} noisy samples')
        print(f'Dataset size: {len(df_aspects)}')

    df_aspects[AspectDatasetColumns.input_column] = df_aspects[AspectDatasetColumns.text_column].apply(
        lambda text_: nltk.tokenize.word_tokenize(text_))

    df_aspects[AspectDatasetColumns.label_column] = df_aspects[AspectDatasetColumns.annotation_column].apply(
        lambda annotations: [AspectDatasetFields.tags_encoder[annotation] for annotation in annotations])

    df_aspects = df_aspects.reset_index()

    del df_aspects['index']

    return df_aspects


def dataset_length_filter(df, subword_token_threshold, tokenizer):
    tokenized_inputs, _ = pre_process_text(texts=df[AspectDatasetColumns.input_column].to_list(),
                                           tokenizer=tokenizer,
                                           tokenizer_config={'max_length': 128, # If false GPU error compability
                                                             'truncation': False,
                                                             'padding': False,
                                                             'is_split_into_words': True
                                                             },
                                           text_tokens=False)
    subword_token_lenghts = [len(i) for i in tokenized_inputs['input_ids']]
    valid_samples = [True if subword_token_lenght <= subword_token_threshold else False for subword_token_lenght in
                     subword_token_lenghts]
    df = df[valid_samples]
    return df


if __name__ == "__main__":
    df_aspect_ = data_reader_aspect_extraction(verbose=False)
    df_aspect_ = parse_data(df_aspects_raw=df_aspect_, verbose=False)




















