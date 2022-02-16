from src.models.aspect_based_sentiment_analysis_mgr import AspectSentimentManager

"""
Implementation notes:
Please do not forget to include the trained polarity model on <data/models/sa_polarity>.
If you want to train your all model, execute <src/notebooks/sentiment_model_analysis.ipynb> and do not forget to include
the training dataset <data/processed/reviews_dataset.csv>
"""

if __name__ == "__main__":

    review_batch = [
        'The food we had yesterday was delicious. And the breakfast was delivered terribly late',
        'It was a surprise, the screen is not bad'
    ]

    # initialize the module
    aspect_sentiment_manager = AspectSentimentManager()

    # use case 1: perform unique prediction with high level output
    df_aspect_sentiment = aspect_sentiment_manager.predict(text_review=review_batch[0], id_review=0, flatten=True)
    print(df_aspect_sentiment)
    """
     id_review                                review                    aspects             polarity  
        0          The food we had yesterday was delicious. The b... [food, breakfast]  [Positive, Negative]    
    """

    # use case 2: perform unique prediction with low level output (include many low level descriptors)
    df_aspect_sentiment = aspect_sentiment_manager.predict(text_review=review_batch[0], id_review=0, flatten=False)
    print(df_aspect_sentiment)
    """
              aspect       modifier rule  aspect_idx  aspect_i  \
    0       food      delicious   r3           4         1   
    1  breakfast  terribly late   r4           8         2   
                                         aspect_sentence  id_review  \
    0    (The, food, we, had, yesterday, was, delicious)          0   
    1  (And, the, breakfast, was, delivered, terribly...          0   
                                                  review sentiment  sentiment_prob  
    0  The food we had yesterday was delicious. And t...  Positive        0.997638  
    1  The food we had yesterday was delicious. And t...  Negative        0.973641  
    """

    # use case 3: perform batch prediction high level:
    df_aspects_sentiment_flatten = aspect_sentiment_manager.predict_batch(text_reviews=review_batch, flatten=True)
    print(df_aspect_sentiment)
    """
           id_review                                             review  \
    0          0  The food we had yesterday was delicious. And t...   
    0          1           It was a surprise, the screen is not bad   
                 aspects              polarity  
    0  [food, breakfast]  [Positive, Negative]  
    0           [screen]            [Positive]  
    """

    # use case 4: perform batch prediction low level:
    df_aspects_sentiment_flatten = aspect_sentiment_manager.predict_batch(text_reviews=review_batch, flatten=False)
    print(df_aspect_sentiment)
    """
              aspect       modifier rule  aspect_idx  aspect_i  \
    0       food      delicious   r3           4         1   
    1  breakfast  terribly late   r4           8         2   
    0     screen        not bad   r3          23         6   
                                         aspect_sentence  id_review  \
    0    (The, food, we, had, yesterday, was, delicious)          0   
    1  (And, the, breakfast, was, delivered, terribly...          0   
    0  (It, was, a, surprise, ,, the, screen, is, not...          1   
                                                  review sentiment  sentiment_prob  
    0  The food we had yesterday was delicious. And t...  Positive        0.997638  
    1  The food we had yesterday was delicious. And t...  Negative        0.973641  
    0           It was a surprise, the screen is not bad  Positive        0.949025  
    """
