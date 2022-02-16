from config.config import COMMON_MODELS, ASPECT_CLASSIFICATION_MODEL_PATH, COMMON_MODELS
from src.models.aspect_extraction_classifier.model_mgr import AspectClassifierManager

"""
To execute this script from shelll set the PYTHONPATH variable and run
>> python src/models/aspect_extraction_classifier/train.py
Use production_mode=True to use all the data for training and production_mode=False to train with a data partition.
"""
if __name__ == "__main__":

    aspect_classifier_mgr = AspectClassifierManager(production_mode=True,
                                                    model_path=ASPECT_CLASSIFICATION_MODEL_PATH)
    aspect_classifier_mgr.train()
