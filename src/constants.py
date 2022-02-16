class DataStructure:
    PRODUCT = 'product_name'
    POLARITY = 'polarity'
    RATING = 'rating'
    REVIEW = 'review_text'
    SOURCE = 'source'
    DOMAIN = 'domain'
    ASPECT = 'aspect'
    PARTITION = 'partition'
    POSITION = 'position'
    COLUMNS = [SOURCE, PRODUCT, REVIEW, RATING, POLARITY]


class PolarityLabels:
    POSITIVE = 'Positive'
    NEGATIVE = 'Negative'
    NEUTRAL = 'Neutral'
    CONFLICTED = 'Conflicted'


class DataSetNames:
    AMAZON = 'multi-domain-amazon'
    GOOGLE_PLAY = 'google-play'
    TRIP_ADVISOR = 'lara-tripadvisor'
    IMDB_MOVIE = 'movie-imdb'
    SEM_EVAL_2014 = 'sem-eval-2014'
    SEM_EVAL_2015 = 'sem-eval-2015'
    SEM_EVAL_2016 = 'sem-eval-2016'
    MAMS_ATSA = 'mams_atsa'