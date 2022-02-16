# product-absa
This project implement an aspect based sentiment analysis to evaluate the polarity of product features in 
customer reviews of e-commerce platforms.

1. SYSTEM COMPONENTS

ASPECT MODULE:
The aspect extraction state of the art involves supervised machine learning methods, most of them offers a decent
performance, the problem, those models do not generalize well for new application domains (if train with movies,
games and phones, you prediction for TV and computers will be inaccurate). Because of this reason we follow
a rule based approach, that cover some generic rules that exhibit better generalization properties. The aspect
extraction module is available importing <from src.models.aspect_extraction.aspect_extractor_rules import 
AspectExtractorMgr>. 

The aspect module provide the list of found aspects, and the sentences tokens that include the text section
with the aspect (those sections are processed by the sentiment model to predict the aspect polarity).

POLARITY MODULE:
The module was trained combining 3 types of reviews datasets from Google App Store, Amazon and TripAdvisor, using 
1.783.166 samples, the training dataset is available on: data/processed/reviews_dataset.csv. The model was tested
in a 5 CV validation with high stability with a average performance over 91%, that is the normal result for transformer
models in the field of sentiment analysis. The training involve two categories Positive and Negative.
A Third category was added artificially in the inference module, to predict "Neutral" when the probability
is under 70%. 


ASPECT SENTIMENT MODULE:
Describes a inference pipeline to extract aspect and evaluate their polarity. The model performance is not the same for
the individual modules, regarding the end-to-end module, because of this reason we evaluate the integration performance
with some selected uses cases described on src/models/evaluation.py


2. HOW TO USE:
This project will be implemented in a batch of product text reviews to extract aspects and
evaluate polarity, like the following black box diagram explains(each review in the input list, will be represented
on a new column):

+-----------------------------------------------------------------------------------------+
| Input  [list[str]]:                                                                     |
| "The food we had yesterday was delicious. And the breakfast was delivered               | 
|  terribly late"                                                                         |
+ ----------------------------------------------------------------------------------------+                                       
                                  |
                                  |
                                  V
             + –––––––––––––––––––––––––––––––––––––––-+
             |  Aspect Based Sentiment Analysis Module |            
             + ––––––––––––––––––––––––––––––––––––––––+ 
                                  |
                                  |
                                  V
 + -------------------------------------------------------------------------------------+
 | Output [pandas-Dataframe]:                                                           |
 | id_review              review                    aspects             polarity        |
 |   0          The food we had yesterday was . [food, breakfast]  [Positive, Negative] |      
 + -------------------------------------------------------------------------------------+

To use the inference engine please follow the next instructions

1. Import the module : <from src.models.aspect_based_sentiment_analysis_mgr import AspectSentimentManager>
2. Create a instance : <aspect_sentiment_manager = AspectSentimentManager()>
3. Call the method   : <df_aspects_sentiment_flatten = aspect_sentiment_manager.predict_batch(text_reviews=review_batch, 
                        flatten=True)>

The "predict_batch" include a configuration method called "flatten", if true you will get a high level
output where each review is a new row in the output dataframe with follow columns:

id_review: int            : unique identifier of review in the "text_reviews" list.
review: str               : Text string given in the input
aspects: list[str]        : List of found aspects
polarity: list[str]       : List of polarities of each aspect on "aspects" column.

By another hand if flatten == False, then you will get a low level output where each aspect in the
input review will be show in a new row. The output dataframe contains the following columns:

aspect: str           : Found aspect
modifier rule: str    : Found aspect modifier
aspect_idx: int       : Character index position of the aspect in the text "review"
aspect_i: int         : Word index position of the aspect in the tokenized text in "review"
aspect_sentence: str  : Tokenized sentence that involves the word "aspect"
id_review: int        : Identifier number of review in the list of reviews "review_batch"
review: str           : Text string given in the input
sentiment: str        : Polarity of aspect
sentiment_prob: float : Probability of polarity predicted on column "sentiment".

Fell free to run the toy examples that describe the main uses cases on: src/playground.py

2. HOW TO INSTALL AND SETUP:
To execute this project you need to install the libraries, download some package and download the trained models as
follow:

1. Clone the repo available on "https://github.com/jespinosal/aspect_based_sentiment_analysis"
From ssh : <git clone git@github.com:jespinosal/aspect_based_sentiment_analysis.git>
From html: <git clone https://github.com/jespinosal/aspect_based_sentiment_analysis.git>

2. Setup your log and temp path. This is a optional step (In training loop the temporal models will be store on temp)
modifying the dict "COMMON_MODELS" on the file config/config.py

2. Create a Python environment (The project was built on python 3.9.5) and install the requirements of requirements.txt
<pip install requirements.txt>

3. Download the Spacy model "en_core_web_lg"
<python -m spacy download en_core_web_lg>

4. Download the data files and paste it the corresponding directories (available on the zip files I shared):
    4.1 Download the sentiment analysis model and paste this on product-absa/data/models/sa_polarity

    4.2 If you want to execute the evaluation, training, data loaders or the available notebooks you have to
    download all the .csv and .json datasets.

The data directory have to follow the next structure: 

data/
    models/
        sa_polarity/
            metrics.json
            tokenizer_config.json
            special_tokens_map.json
            config.json
            train_dataset.csv
            tokenizer.json
            merges.txt
            pytorch_model.bin
            labels_map.json
            vocab.json
    eval/
    processed/
        reviews_dataset.csv
        aspect_dataset.csv
    raw/
        Restaurants_Train_v2.xml
        Laptop_Train_v2.xml
        archive/
            .DS_Store
            googleplaystore_user_reviews.csv
            googleplaystore.csv
            license.txt
        sem_eval_2014/
            Restaurants_Train_v2.xml
            corpora.pkl.txt
            corpora.pkl
            restaurants-trial.xml
            laptops-trial.xml
            CLAS_A_Database_for_Cognitive_Load_Affect_and_Stress_Recognition.pdf
            Laptop_Train_v2.xml
        json/
            280518.json
            241011.json
            2514728.json
            ...
            2514479.json
            239472.json
            78587.json
        sorted_data/
            stopwords
            summary.txt
            music/
                unlabeled.review
                processed.review.balanced
                processed.review.random
                all.review
                processed.review
                negative.review
                positive.review
            video/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            health_&_personal_care/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            jewelry_&_watches/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            magazines/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            electronics/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            camera_&_photo/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            tools_&_hardware/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            office_products/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            books/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            automotive/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            dvd/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            grocery/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            software/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            beauty/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            cell_phones_&_service/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            gourmet_food/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            kitchen_&_housewares/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            toys_&_games/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            sports_&_outdoors/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            musical_instruments/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            outdoor_living/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            apparel/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            baby/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
            computer_&_video_games/
                unlabeled.review
                processed.review.balanced
                all.review
                processed.review
                negative.review
                positive.review
                
5. Test the project executing
<python src/playground.py>

3. SCOPE AND IMPROVEMENTS:
- Tuning parameter "window" of the aspect module: this parameter define the aspect sentence length that follows a ngram
approach, the window defines the number of words around the extracted aspect. This sentence-segment is used to predict 
the aspect polarity.
- Tuning Neutral threshold: The default value is 70%, once you have your own testing dataset, run the evaluation routine
with different threshold values.
- Build your own aspect dataset, train a sequence-to-sequence model and build a model ensemble with rule based module.
