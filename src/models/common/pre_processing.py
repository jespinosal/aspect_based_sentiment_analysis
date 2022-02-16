from transformers import AutoTokenizer


def auto_tokenizer(pretrained_model_name_or_path):
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                         do_lower_case=True)

def token_tokenizer():
    pass