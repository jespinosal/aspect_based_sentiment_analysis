"""
Evaluation assumption: the testing dataset complexity is intermediate level, the used reviews for
training and calibration are not the same of testing.

For aspect evaluation we used 3404 text reviews, with zero or one aspects and less than 15 words. All the long
reviews with many aspects was excluded from the analysis. The model accuracy is 62.16%

For polarity evaluation we used 1811 samples (negative:655, neutral:284, positive:872), excluding all the
really short reviews (with poor context) with less than 5 words. The testing reviews without aspect and polarity
where ignored, because they has not a grand truth to compare the prediction results. The model accuracy
is 67% (high recall for positive 91%, intermediate precision for negative and poor results for neutral 10% f1_score)

The state of the art of aspect base sentiment analysis models evaluated with an unseen application domain is about 72%.
(For known application domain outperform 88%, but that involves train a test in same application domain, and huge amount
of training data of each application domain and it is not our situation).

We performed independent test for sentiment and polarity, because for both model we used different metrics. Another
reason, the aspect model we will consider review sentences without aspects as well, to analyze the model performance
when have to predict "None" aspect or have to predict any aspect. For sentiment we ignore the reviews without
aspect and sentiment in the testing dataset, also we ignore the polarity category "conflict", to evaluate just
the test with positive, negative and neutral labels. (The ML model is able to predict positive and negative labels,
neutral was added using a threshold of 70%, when positive/negative probabilities are under this threshold we asing
"neutral", with more neutral samples we could perform a tuning to set this parameter properly).

Please do not forget to include the testing dataset on <data/processed/aspect_dataset.csv> and
the polarity model on <data/models/sa_polarity>

:param df_test:
:return:
"""

from src.data_loader import reader_sem_eval_2014
from src.models.aspect_based_sentiment_analysis_mgr import AspectSentimentManager
from sklearn.metrics import classification_report


def aspect_dataset_parser(aspect_dataset):
    aspect_dataset = aspect_dataset.copy()
    aspect_dataset['aspect_sentiment'] = aspect_dataset.apply(lambda row: list(zip(row.aspects, row.polarity)),
                                                              axis=1)

    if 'review' in aspect_dataset.columns:
        return aspect_dataset[['id_review', 'review', 'aspect_sentiment']]
    else:
        return aspect_dataset[['text', 'aspect_sentiment', 'dataset']]


def evaluator():
    """
    Perforn a model analysis of the aspect based sentiment analysis module
    :return:
    """
    df_test = reader_sem_eval_2014()
    df_test = aspect_dataset_parser(aspect_dataset=df_test)

    aspect_sentiment_manager = AspectSentimentManager()

    df_predict = aspect_sentiment_manager.predict_batch(text_reviews=df_test.text.to_list(), flatten=True)
    df_predict_ = aspect_dataset_parser(aspect_dataset=df_predict)

    df_test['predicted_aspect_sentiment'] = df_predict_['aspect_sentiment'].to_list()

    df_test['review'] = df_predict_['review'].to_list()

    df_test['id_review'] = df_predict_['id_review'].to_list()

    df_test['predicted_aspects'] = df_predict['aspects'].to_list()

    return df_test


def use_case_polarity_eval(df_test):
    df_test = df_test.copy()

    df_test_single_review = df_test[df_test.aspect_sentiment.str.len() == 1]  # process reviews with one aspect

    df_test_single_review = df_test_single_review[[len(review.split()) >= 5 for review in
                                                   df_test_single_review.review.to_list()]]  # get text with unless 1 word

    df_test_single_review = df_test_single_review[df_test_single_review.aspect_sentiment.apply(lambda x:
                                                                                               x[0][1] not in
                                                                                               ['conflict'])]

    y_true_sentiment = df_test_single_review.aspect_sentiment.apply(lambda x: x[0][1].lower() if x[0][1] is not None
                                                                    else 'None')

    y_pre_sentiment = df_test_single_review.predicted_aspect_sentiment.apply(lambda x: x[0][1].lower()
                                                                             if x[0][1] is not None else 'None')

    #  y_pre_entity = df_test_single_review.apply(lambda row: any([str(predict_aspect) in row.aspect_sentiment[0][0]
    #                                                            for predict_aspect in row.predicted_aspects]),
    #                                           axis=1)
    eval = (y_pre_sentiment==y_true_sentiment).to_list()
    df_test_single_review['eval'] = eval

    print(f'Sentiment Polarity accuracy: {eval.count(True)/(eval.count(True)+eval.count(False))}')
    print(classification_report(y_true=y_true_sentiment, y_pred=y_pre_sentiment))

    return df_test_single_review


def use_case_aspect_eval(df_test):
    df_test = df_test.copy()
    min_word_size = 15
    df_test['aspect_sentiment'] = df_test.aspect_sentiment.apply(lambda x: [('None', 'Positive')]
                                                                 if not x else x)

    df_test_single_review = df_test[df_test.aspect_sentiment.str.len() == 1]  # process reviews with one aspect
    df_test_single_review = df_test_single_review[[len(review.split()) <= min_word_size for review in
                                                   df_test_single_review.review.to_list()]]  # get text with unless 1 word

    df_test_single_review = df_test_single_review[
        df_test_single_review.aspect_sentiment.apply(lambda x: x[0][1] not
                                                     in ['conflict'])]

    #  y_true_sentiment = df_test_single_review.aspect_sentiment.apply(
    #    lambda x: x[0][1].lower() if x[0][1] is not None
    #    else 'None')

    #  y_pre_sentiment = df_test_single_review.predicted_aspect_sentiment.apply(
    #  lambda x: x[0][1].lower() if x[0][1] is
    #  not None else 'None')

    y_pre_entity = df_test_single_review.apply(
        lambda row: any([str(predict_aspect) in row.aspect_sentiment[0][0]
                         for predict_aspect in row.predicted_aspects]), axis=1)

    df_test_single_review['eval'] = y_pre_entity

    acc = y_pre_entity.to_list().count(True) / ((y_pre_entity.to_list().count(True) +
                                                y_pre_entity.to_list().count(False)))

    print(f'Aspect extraction accuracy:{acc} test dataset size: {len(y_pre_entity)}')

    return df_test_single_review


if __name__ == "__main__":

    df_test_ = evaluator()
    df_test_single_review_polarity_test = use_case_polarity_eval(df_test=df_test_)
    df_test_single_review_entity_test = use_case_aspect_eval(df_test=df_test_)

    """
            Sentiment Polarity accuracy: 0.6697956929872998
                      precision    recall  f1-score   support
            negative       0.73      0.61      0.66       655
             neutral       0.21      0.07      0.10       284
            positive       0.68      0.91      0.78       872
            accuracy                           0.67      1811
           macro avg       0.54      0.53      0.51      1811
        weighted avg       0.62      0.67      0.63      1811
        
        
        Aspect extraction accuracy:0.6216216216216216 test dataset size: 3404
    """
