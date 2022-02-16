"""
This script implement the methods to read amazon review files from
https://jmcauley.ucsd.edu/data/amazon/
"""
import os
import pandas as pd
import json
import pickle
from src.constants import DataStructure, DataSetNames, PolarityLabels
from src.text_processing import basic_pre_processing
from xml.etree.ElementTree import parse

data_path = 'data/raw'
output_path = 'data/processed'

config_paths = {DataSetNames.AMAZON: os.path.join(data_path, 'sorted_data'),
                DataSetNames.TRIP_ADVISOR:  os.path.join(data_path, 'json'),
                DataSetNames.IMDB_MOVIE: os.path.join(data_path, 'movie'),
                DataSetNames.GOOGLE_PLAY: os.path.join(data_path, 'archive'),
                DataSetNames.SEM_EVAL_2014: os.path.join(data_path, 'sem_eval_2014'),
                DataSetNames.MAMS_ATSA: os.path.join(data_path, 'MAMS-ATSA/raw'),
                DataSetNames.SEM_EVAL_2015: os.path.join(data_path, 'sem_eval_2015'),
                DataSetNames.SEM_EVAL_2016: os.path.join(data_path, 'sem_eval_2016')}




def parse_sentiment_polarity(max, min, value):
    if value is not None:
        if (value > min) and (value < max):
            return PolarityLabels.NEUTRAL
        elif value <= min:
            return PolarityLabels.NEGATIVE
        elif value >= max:
            return PolarityLabels.POSITIVE
        else:
            print('None value 1')
            return None
    else:
        print('None value 2')
        return None


def reader_multi_domain_amazon():
    """

    :return:
    """
    files = ['googleplaystore.csv', 'googleplaystore_user_reviews.csv']
    df_google_play = pd.read_csv(os.path.join(config_paths[DataSetNames.GOOGLE_PLAY],
                                 files[0]),
                                 sep=',')
    df_google_play_user_review = pd.read_csv(os.path.join(config_paths['google-play'],
                                 files[1]),
                                 sep=',')

    df_google_reviews = pd.merge(df_google_play, df_google_play_user_review,
                                 how='inner',
                                 on='App')

    df_google_reviews = df_google_reviews.rename(columns={'Translated_Review': DataStructure.REVIEW,
                                                          'Sentiment_Polarity': DataStructure.RATING,
                                                          'App': DataStructure.PRODUCT,
                                                          'Sentiment': DataStructure.POLARITY,
                                                          })

    df_google_reviews[DataStructure.SOURCE] = DataSetNames.GOOGLE_PLAY
    #@todo filter as is needed, check noisy
    return df_google_reviews[DataStructure.COLUMNS]


def reader_lara_tripadvisor():
    """

    :return:
    """
    files = os.listdir(os.path.join(config_paths[DataSetNames.TRIP_ADVISOR]))
    samples = []
    for file in files:
        file_path = os.path.join(config_paths[DataSetNames.TRIP_ADVISOR],file)
        with open(file_path, 'r') as file_:
            try:
                data = json.load(file_)
                product = data['HotelInfo']['HotelID']
                reviews = data['Reviews']
                for review in reviews:
                    rating = review['Ratings']['Overall']
                    text_review = review['Content']
                    samples.append({DataStructure.PRODUCT: product,
                                    DataStructure.REVIEW: text_review,
                                    DataStructure.RATING: rating})
            except:
                print(f'error')

    df_tripadvisor = pd.DataFrame(samples)
    df_tripadvisor[DataStructure.SOURCE] = DataSetNames.TRIP_ADVISOR
    df_tripadvisor[DataStructure.POLARITY] = df_tripadvisor[DataStructure.RATING].apply(lambda x:
                                                                                        parse_sentiment_polarity(max=3, min=2, value=float(x)))

    return df_tripadvisor[DataStructure.COLUMNS]


def reader_amazon_multi_domain():
    """

    :return:
    """

    def processing_xml_reviews_dict(ListReviews):
        """

        :param ListReviews:
        :return:
        """
        count = 0
        Review = []
        Reviews = {}
        for i in range(len(ListReviews)):
            if ListReviews[i] != '</review>\n':
                if ListReviews[i] == '<review>\n' and ListReviews[i + 1] == '<unique_id>\n':
                    # unique_id
                    Review.append('unique_id/' + ListReviews[i + 2])
                # if ListReviews[i] == '</unique_id>\n' and ListReviews[i+1] == '<unique_id>\n':
                # unique_idN
                # Review.append('unique_id/'+ListReviews[i+2])
                if ListReviews[i] == '<asin>\n':
                    # asin
                    Review.append('asin/' + ListReviews[i + 1])
                if ListReviews[i] == '<product_name>\n':
                    # productName
                    Review.append('product_name/' + ListReviews[i + 1])
                # if  ListReviews[i] == '</product_type>\n' and ListReviews[i+1] == '<product_type>\n' :
                # here we append the producttype
                # Review.append('product_type/'+ListReviews[i+2])
                if ListReviews[i] == '<helpful>\n':
                    # helpful
                    Review.append('helpful/' + ListReviews[i + 1])
                if ListReviews[i] == '<rating>\n':
                    Review.append('rating/' + ListReviews[i + 1])
                if ListReviews[i] == '<title>\n':
                    Review.append('title/' + ListReviews[i + 1])
                if ListReviews[i] == '<date>\n':
                    Review.append('date/' + ListReviews[i + 1])
                if ListReviews[i] == '<reviewer>\n':
                    Review.append('reviewer/' + ListReviews[i + 1])
                if ListReviews[i] == '<reviewer_location>\n':
                    Review.append('reviewer_location/' + ListReviews[i + 1])
                if ListReviews[i] == '<review_text>\n':
                    Review.append('review_text/' + ListReviews[i + 1])
            elif ListReviews[i] == '</review>\n':
                count = count + 1
                r = 'review' + str(count)
                Reviews[r] = Review
                # nfargou list
                Review = []
        return Reviews

    def dictionary_to_data_frame(Dict):
        """

        :param Dict:
        :return:
        """
        # on prepare notre dataframe pour les données
        df = pd.DataFrame(columns=['unique_id', 'asin', 'product_name', 'helpful', 'rating', 'title',
                                   'date', 'reviewer', 'reviewer_location', 'review_text'])
        count = 0
        for i, k in Dict.items():
            df.loc[count] = [k[0].split("/")[1].split("\n")[0], k[1].split("/")[1].split("\n")[0]
                , k[2].split("/")[1].split("\n")[0], k[3].split("/")[1].split("\n")[0]
                , k[4].split("/")[1].split("\n")[0], k[5].split("/")[1].split("\n")[0]
                , k[6].split("/")[1].split("\n")[0], k[7].split("/")[1].split("\n")[0]
                , k[8].split("/")[1].split("\n")[0], k[9].split("/")[1].split("\n")[0]
                             ]
            count = count + 1

        return df

    files = os.listdir(config_paths[DataSetNames.AMAZON])
    pos_df = []
    neg_df = []
    for file in files:
        file_path = os.path.join(config_paths[DataSetNames.AMAZON], file)
        if os.path.isdir(file_path):
            BooksPositRev = open(os.path.join(file_path, 'positive.review'), 'r', encoding="utf8",
                                 errors='ignore').readlines()  # read = r
            BooksNegatRev = open(os.path.join(file_path, 'negative.review'), 'r', encoding="utf8",
                                 errors='ignore').readlines()  # read = r
            NegRev_Dict = processing_xml_reviews_dict(BooksNegatRev)
            posRev_Dict = processing_xml_reviews_dict(BooksPositRev)
            BooksNeg = dictionary_to_data_frame(NegRev_Dict)
            BooksPos = dictionary_to_data_frame(posRev_Dict)
            pos_df.append(BooksPos)
            neg_df.append(BooksNeg)
    df_positive = pd.concat(pos_df)
    df_negative = pd.concat(neg_df)
    df_positive[DataStructure.POLARITY] = PolarityLabels.POSITIVE
    df_negative[DataStructure.POLARITY] = PolarityLabels.NEGATIVE
    df_amazon_review = pd.concat([df_positive, df_negative])
    df_amazon_review[DataStructure.SOURCE] = DataSetNames.AMAZON

    return df_amazon_review[DataStructure.COLUMNS]


def reader_sem_eval_2014():
    datasets = ['laptop', 'restaurants']
    partitions = ['trainset', 'testset']
    corpora = pickle.load(open(os.path.join(config_paths[DataSetNames.SEM_EVAL_2014], 'corpora.pkl'), 'rb'))
    samples = []
    for dataset in datasets:
        for partition in partitions:
            for corpus_ in corpora[dataset][partition]['corpus'].corpus:
                aspect_terms = [aspect_term.term for aspect_term in corpus_.aspect_terms]
                aspect_polarity = [aspect_term.polarity for aspect_term in corpus_.aspect_terms]
                samples.append({DataStructure.REVIEW: corpus_.text,
                                DataStructure.ASPECT: aspect_terms,
                                DataStructure.POLARITY: aspect_polarity,
                                DataStructure.SOURCE: DataSetNames.SEM_EVAL_2014,
                                DataStructure.DOMAIN: dataset,
                                DataStructure.PARTITION: partition,
                                DataStructure.POSITION: [(None, None)]})
    return pd.DataFrame(samples)


def reader_mams_atsa():
    """
    Data set for aspect based sentiment analysis that involves 500 annotations from reviews dataset.
    The dataset novelty involves challengin annotations with multiple aspect and multiple polarity.
    The stated of the art for this dataset using BERT is 79% f1-score.
    :return:
    """
    domain = 'restaurants'
    partitions = ['test', 'train', 'val']
    samples = []
    for partition in partitions:
        path = os.path.join(config_paths[DataSetNames.MAMS_ATSA], f'{partition}.xml')
        tree = parse(path)
        sentences = tree.getroot()
        for sentence in sentences:
            text = sentence.find('text')
            if text is None:
                continue
            text = text.text
            aspectTerms = sentence.find('aspectTerms')
            if aspectTerms is None:
                continue
            aspect_term_list = []
            aspect_polarity_list = []
            position_list = []
            for aspectTerm in aspectTerms:
                term = aspectTerm.get('term')
                polarity = aspectTerm.get('polarity')
                start = aspectTerm.get('from')
                end = aspectTerm.get('to')
                position_list.append((start, end))
                aspect_term_list.append(term)
                aspect_polarity_list.append(polarity)

            samples.append({DataStructure.REVIEW: text,
                            DataStructure.ASPECT: aspect_term_list,
                            DataStructure.POLARITY: aspect_polarity_list,
                            DataStructure.SOURCE: DataSetNames.MAMS_ATSA,
                            DataStructure.DOMAIN: domain,
                            DataStructure.PARTITION: partition,
                            DataStructure.POSITION: position_list})

    return pd.DataFrame(samples)


def reader_sem_eval_201x(year='2015'):
    """
    Each review is splited in sentences and each sentence can contain several opinions.
    This method accpets as year parameter 2015 and 2016
    :return:
    """
    if year == '2015':
        source = DataSetNames.SEM_EVAL_2015
    elif year == '2016':
        source = DataSetNames.SEM_EVAL_2016
    else:
        raise ValueError

    partitions = {'test': os.path.join(config_paths[source], 'Restaurants_Train_Final.xml'),
                  'train': os.path.join(config_paths[source], 'Restaurants_Train_Final.xml')}
    samples = []
    for partition, path in partitions.items():
        tree = parse(path)
        reviews = tree.getroot()
        for review in reviews:
            for sentences in review:
                for sentence in sentences:
                    sentences_texts_aspects = []
                    sentences_aspects_polarity = []
                    sentences_aspects_positions = []
                    text = sentence.find('text').text
                    opinions = sentence.find('Opinions')
                    if opinions is None:
                        print(f'text without opinion will be ignored:{text}')
                        break
                    for opinion in opinions:
                        aspect = opinion.get('target')
                        polarity = opinion.get('polarity')
                        position = (opinion.get('from'), opinion.get('to'))
                        sentences_texts_aspects.append(aspect)
                        sentences_aspects_polarity.append(polarity)
                        sentences_aspects_positions.append(position)

                    samples.append({
                                DataStructure.REVIEW: text,
                                DataStructure.ASPECT: sentences_texts_aspects,
                                DataStructure.POLARITY: sentences_aspects_polarity,
                                DataStructure.SOURCE: source,
                                DataStructure.DOMAIN: 'restaurants',
                                DataStructure.PARTITION: partition,
                                DataStructure.POSITION: sentences_aspects_positions
                                    })
    return pd.DataFrame(samples)


if __name__ == "__main__":

    # Building sentiment dataset:
    df_google_apps_ = reader_multi_domain_amazon()
    df_amazon_sample = reader_amazon_multi_domain()
    df_trip_advisor = reader_lara_tripadvisor()
    df_reviews = pd.concat([df_google_apps_, df_amazon_sample, df_trip_advisor])
    df_reviews[DataStructure.REVIEW] = df_reviews[DataStructure.REVIEW].apply(
        lambda x: basic_pre_processing(x))
    df_reviews.to_csv(os.path.join(output_path, 'reviews_dataset.csv'),  sep=';')

    # Building aspect dataset (for testing purpose):
    # pd.set_option('display.max_colwidth', None)
    df_aspect_sem_eval_2014 = reader_sem_eval_2014()
    df_aspect_sem_eval_2015 = reader_sem_eval_201x(year='2015')
    df_aspect_sem_eval_2016 = reader_sem_eval_201x(year='2016')
    df_aspect_mams_atsa = reader_mams_atsa()
    df_aspects = pd.concat([df_aspect_sem_eval_2014, df_aspect_sem_eval_2015, df_aspect_sem_eval_2016,
                            df_aspect_mams_atsa])
    df_aspects.to_csv(os.path.join(output_path, 'aspect_dataset.csv'), sep=';', index=False)


    """
    Notes todo data cleaning - pre procesing:
    0) Clean inconsistencies (count aspect/polarity and clean non equal size samples) 
    1) Filter Null aspects/sentiments: prepare non aspect cases 
    (When an entity E is only implicitly referred (e.g. through pronouns) or inferred in a sentence, then the OTE slot
     is assigned the value “NULL”)
    a. sem-eval-2015, include some NULL aspect with sentiments example:
    review_text    Chow fun was dry; pork shu mai was more than usually greasy and had to share a table with loud and 
    rude family. 
    aspect -> [Chow fun, pork shu mai, NULL]
    polarity -> [negative, negative, negative]    
    b. sem-eval-2014 include empty cases when non aspect in the sentence -> []                                                                                     
    2) Process each sentence such as NER BERT format
    3) Analise stats and clean short/long samples
    4) Balance dataset choose more "rich" Null examples
    
    """



