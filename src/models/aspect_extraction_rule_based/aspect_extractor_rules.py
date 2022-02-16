import spacy
import nltk
from typing import List
from src.text_processing import basic_pre_processing

nlp = spacy.load('en_core_web_lg')

PRODUCT_PRONOUNS = ['it', 'this', 'they', 'these']

DEFAULT_TOKEN_TEXT = '999999'


class RuleID:
    R1 = 'r1'
    R2 = 'r2'
    R3 = 'r3'
    R4 = 'r4'
    R5 = 'r5'
    R6 = 'r5'
    R7 = 'r7'


ASPECT_FORMAT = {'aspect': None,
                 'modifier': None,
                 'rule': None,
                 'aspect_idx': None,
                 'aspect_i': None,
                 'aspect_sentence': None}


def word_id_to_character_id(text_review, idx):
    word_to_character = {}
    word_counter = 0
    for n, character in enumerate(text_review):
        word_to_character[n] = word_counter
        if character == ' ':
            word_counter += 1
    return word_to_character[idx]


def spacy_doc_parser(text_review):
    """
    To increase the application performance some methods re use the same spacy doc conversion
    :param text_review:
    :return:
    """
    if not isinstance(text_review, spacy.tokens.doc.Doc):
        doc = nlp(text_review)
    else:
        doc = text_review
    return doc


def word_id_sentence_getter(text_review, word_id, n, i_s):
    """
    Get aspect sentence using ngram approach
    :param text_review: text review string
    :param i: word index position
    :param n: ngram size of phrase
    :param aspects_counter:
    :param i_s:
    Get the phrase text_review[n-aspect_word_id:n+aspect_word_id] using a window size of 2 * n, where the aspect
    is the centran element in the ngram
    :return:
    """
    aspects_counter = len(i_s)
    i_s_ = i_s.copy()
    i_s_ = sorted(i_s_)
    current_aspect_i_s_index = i_s_.index(word_id)
    next_aspect_word_index = None if current_aspect_i_s_index==len(i_s)-1 else i_s_[current_aspect_i_s_index + 1]
    previous_aspect_word_index = None if current_aspect_i_s_index==0 else i_s_[current_aspect_i_s_index -1]

    text_review = spacy_doc_parser(text_review)

    if aspects_counter == 1:
        return text_review
    else:
        id_right = word_id+n if word_id+n < len(text_review) else len(text_review)
        if next_aspect_word_index is not None:  # if is not the last element, then i_s_ != [] to check aspect overlaps
            id_right = id_right if id_right < next_aspect_word_index else next_aspect_word_index-1
        else:
            id_right = len(text_review)

        id_left = word_id-n if word_id-n > 0 else 0
        if previous_aspect_word_index is not None: # check it if is the first
            id_left = id_left if id_left > previous_aspect_word_index else previous_aspect_word_index+1
        else:
            id_left = 0

        return text_review[id_left:id_right]


def entity_recognizer(text_review):

    doc = spacy_doc_parser(text_review)

    ner_heads = {ent.root.idx: ent for ent in doc.ents}

    return ner_heads


def entity_matcher(ner_heads: dict, aspect: str, idx: int):
    """
    If aspect is a NER entity it will be parsed.
    :param ner_heads:
    :param aspect:
    :param idx:
    :return:
    """
    return ner_heads.get(idx, aspect)


def sentence_aspect_extraction_pipeline(text_review, n=7):

    text_review = spacy_doc_parser(text_review)
    aspects_dicts = aspect_extraction(text_review)
    ner_heads = entity_recognizer(text_review)
    i_s = [aspect_dict['aspect_i'] for aspect_dict in aspects_dicts]  # list of aspect word index position

    if aspects_dicts:
        for aspect_dict in aspects_dicts:
            aspect_dict['aspect'] = entity_matcher(ner_heads=ner_heads,
                                                   aspect=aspect_dict['aspect'],
                                                   idx=aspect_dict['aspect_idx'])

            aspect_dict['aspect_sentence'] = word_id_sentence_getter(text_review,
                                                                     word_id=aspect_dict['aspect_i'],
                                                                     n=n,
                                                                     i_s=i_s)
    else:
        aspects_dict = ASPECT_FORMAT.copy()
        aspects_dict['aspect_sentence'] = text_review
        aspects_dicts = [aspects_dict]

    return aspects_dicts


def aspect_extraction(text_review: str) -> List[dict]:

    doc = spacy_doc_parser(text_review)

    ## FIRST RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect
    ## RULE = M is child of A with a relationshio of amod
    rule1_pairs = []

    for token in doc:
        A_i = None
        A_idx = None
        A = DEFAULT_TOKEN_TEXT
        M = DEFAULT_TOKEN_TEXT

        if token.dep_ == "amod" and not token.is_stop:
            M = token.text
            A = token.head.text
            A_idx = token.head.idx
            A_i = token.head.i

            # add adverbial modifier of adjective (e.g. 'most comfortable headphones')
            M_children = token.children
            for child_m in M_children:
                if child_m.dep_ == "advmod":
                    M_hash = child_m.text
                    M = M_hash + " " + M
                    break

            # negation in adjective, the "no" keyword is a 'det' of the noun (e.g. no interesting characters)
            A_children = token.head.children
            for child_a in A_children:
                if child_a.dep_ == "det" and child_a.text == 'no':
                    neg_prefix = 'not'
                    M = neg_prefix + " " + M
                    break

        if A != DEFAULT_TOKEN_TEXT and M != DEFAULT_TOKEN_TEXT:
            rule1_pairs.append((A, M, RuleID.R1, A_idx, A_i))

    ## SECOND RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect
    #Direct Object - A is a child of something with relationship of nsubj, while
    # M is a child of the same something with relationship of dobj
    #Assumption - A verb will have only one NSUBJ and DOBJ

    rule2_pairs = []
    for token in doc:
        children = token.children
        A = DEFAULT_TOKEN_TEXT
        M = DEFAULT_TOKEN_TEXT
        add_neg_pfx = False
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                A = child.text
                A_idx = child.idx
                A_i = child.i

            if (child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop:
                M = child.text

            if child.dep_ == "neg":
                neg_prefix = child.text
                add_neg_pfx = True

        if add_neg_pfx and M != DEFAULT_TOKEN_TEXT:
            M = neg_prefix + " " + M

        if A != DEFAULT_TOKEN_TEXT and M != DEFAULT_TOKEN_TEXT:
            rule2_pairs.append((A, M, RuleID.R2, A_idx, A_i))

    ## THIRD RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect
    ## Adjectival Complement - A is a child of something with relationship of nsubj, while
    ## M is a child of the same something with relationship of acomp
    ## Assumption - A verb will have only one NSUBJ and DOBJ
    ## "The sound of the speakers would be better. The sound of the speakers could be better" - handled using AUX dependency

    rule3_pairs = []

    for token in doc:

        children = token.children
        A = DEFAULT_TOKEN_TEXT
        M = DEFAULT_TOKEN_TEXT
        add_neg_pfx = False
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                A = child.text
                A_idx = child.idx
                A_i = child.i

            if child.dep_ == "acomp" and not child.is_stop:
                M = child.text

            # example - 'this could have been better' -> (this, not better)
            if child.dep_ == "aux" and child.tag_ == "MD":
                neg_prefix = "not"
                add_neg_pfx = True

            if child.dep_ == "neg":
                neg_prefix = child.text
                add_neg_pfx = True

        if add_neg_pfx and M != DEFAULT_TOKEN_TEXT:
            M = neg_prefix + " " + M

        if A != DEFAULT_TOKEN_TEXT and M != DEFAULT_TOKEN_TEXT:
            rule3_pairs.append((A, M, RuleID.R3, A_idx, A_i))

    ## FOURTH RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect

    #Adverbial modifier to a passive verb - A is a child of something with relationship of nsubjpass, while
    # M is a child of the same something with relationship of advmod

    #Assumption - A verb will have only one NSUBJ and DOBJ

    rule4_pairs = []
    for token in doc:

        children = token.children
        A = DEFAULT_TOKEN_TEXT
        M = DEFAULT_TOKEN_TEXT
        add_neg_pfx = False
        for child in children:
            if (child.dep_ == "nsubjpass" or child.dep_ == "nsubj") and not child.is_stop:
                A = child.text
                A_idx = child.idx
                A_i = child.i

            if child.dep_ == "advmod" and not child.is_stop:
                M = child.text
                M_children = child.children
                for child_m in M_children:
                    if child_m.dep_ == "advmod":
                        M_hash = child_m.text
                        M = M_hash + " " + child.text
                        break

            if child.dep_ == "neg":
                neg_prefix = child.text
                add_neg_pfx = True

        if add_neg_pfx and M != DEFAULT_TOKEN_TEXT:
            M = neg_prefix + " " + M

        if A != DEFAULT_TOKEN_TEXT and M != DEFAULT_TOKEN_TEXT:
            rule4_pairs.append((A, M, RuleID.R4, A_idx, A_i))

    ## FIFTH RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect

    #Complement of a copular verb - A is a child of M with relationship of nsubj, while
    # M has a child with relationship of cop

    #Assumption - A verb will have only one NSUBJ and DOBJ

    rule5_pairs = []
    for token in doc:
        children = token.children
        A = DEFAULT_TOKEN_TEXT
        buf_var = DEFAULT_TOKEN_TEXT
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                A = child.text
                A_idx = child.idx
                A_i = child.i

            if child.dep_ == "cop" and not child.is_stop:
                buf_var = child.text

        if A != DEFAULT_TOKEN_TEXT and buf_var != DEFAULT_TOKEN_TEXT:
            rule5_pairs.append((A, token.text, RuleID.R5, A_idx, A_i))

    ## SIXTH RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect
    ## Example - "It ok", "ok" is INTJ (interjections like bravo, great etc)

    rule6_pairs = []
    for token in doc:
        children = token.children
        A = DEFAULT_TOKEN_TEXT
        M = DEFAULT_TOKEN_TEXT
        if token.pos_ == "INTJ" and not token.is_stop:
            for child in children :
                if child.dep_ == "nsubj" and not child.is_stop:
                    A = child.text
                    A_idx = child.idx
                    A_i = child.i
                    M = token.text

        if A != DEFAULT_TOKEN_TEXT and M != DEFAULT_TOKEN_TEXT:
            rule6_pairs.append((A, M, RuleID.R6, A_idx, A_i))

    ## SEVENTH RULE OF DEPENDANCY PARSE -
    ## M - Sentiment modifier || A - Aspect
    ## ATTR - link between a verb like 'be/seem/appear' and its complement
    ## Example: 'this is garbage' -> (this, garbage)

    rule7_pairs = []
    for token in doc:
        children = token.children
        A = DEFAULT_TOKEN_TEXT
        M = DEFAULT_TOKEN_TEXT
        add_neg_pfx = False
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                A = child.text
                A_idx = child.idx
                A_i = child.i

            if (child.dep_ == "attr") and not child.is_stop:
                M = child.text

            if child.dep_ == "neg":
                neg_prefix = child.text
                add_neg_pfx = True

        if add_neg_pfx and M != DEFAULT_TOKEN_TEXT:
            M = neg_prefix + " " + M

        if A != DEFAULT_TOKEN_TEXT and M != DEFAULT_TOKEN_TEXT:
            rule7_pairs.append((A, M, RuleID.R7, A_idx, A_i))

    aspects_tuples = rule1_pairs + rule2_pairs + rule3_pairs + rule4_pairs + rule5_pairs + rule6_pairs + rule7_pairs

    # replace all instances of "it", "this" and "they" with "product"
    aspects_dict = [{'aspect': A, 'modifier': M, 'rule': r, 'aspect_idx': A_idx, 'aspect_i': A_i}
                    if A not in PRODUCT_PRONOUNS
                    else
                    {'aspect': 'product', 'modifier': M, 'rule': r, 'aspect_idx': A_idx, 'aspect_i': A_i}
                    for A, M, r, A_idx, A_i in aspects_tuples]

    return aspects_dict


class AspectExtractorMgr:

    def __init__(self, window):
        self.window = window
        self.aspect_format = {'aspect': None,
                              'modifier': None,
                              'rule': None,
                              'aspect_idx': None,
                              'aspect_i': None,
                              'aspect_sentence': None}

    @staticmethod
    def text_preprocessing(text):
        return basic_pre_processing(text)

    @staticmethod
    def sentence_tokenizer(text):
        return nltk.sent_tokenize(text)

    def get_aspects(self, text):
        return sentence_aspect_extraction_pipeline(text_review=text, n=self.window)

    def extract_aspect(self, text):
        sentences = self.sentence_tokenizer(text)
        aspects = []
        for sentence in sentences:
            sentence = self.text_preprocessing(sentence)
            aspect_dict = self.get_aspects(text=sentence)
            aspects.extend(aspect_dict)
        return aspects


if __name__ == "__main__":

    sentences_ = [
        'The food we had yesterday was delicious but the breakfast was delivered late',
        'The food we had yesterday was delicious. The breakfast was delivered late',
        'My time in Italy was very enjoyable',
        'I found the meal to be tasty',
        'The internet was slow.',
        'Our experience was suboptimal',
        'Air France is cool',
        'I think Gabriel García Márquez is not boring. The first book I read was funny',
        'They say Central African Republic is really great',
        'I like the hotel but hate those dirty bathroom',
        'The product price is so high',
        'The battery duration is not bad',
        'This Huawei Pro colour is beauty and the camera is awesome',
        'This is not exactly the best screen of Apple but it has a very powerful processor and a amazing memory',
        'I bought a HP Pavilion DV4-1222nr laptop and have had so many problems with the computer.'
                ]

    aspect_extractor = AspectExtractorMgr(window=5)
    aspects_list = []
    for sentence_ in sentences_:
        aspects_ = aspect_extractor.extract_aspect(text=sentence_)
        aspects_list.append(aspects_)
