"""Get pre-trained models
"""
from transformers import (AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM,
                          BertJapaneseTokenizer, BertTokenizer, T5Tokenizer,
                          logging)

logging.set_verbosity_error()

TOKENIZER_CLASSES = {
    'gpt2': AutoTokenizer,
    'rinna/japanese-gpt2-medium': T5Tokenizer,
    'cl-tohoku/bert-base-japanese': BertJapaneseTokenizer,
    'bert-base-cased': BertTokenizer,
    'NlpHUST/vibert4news-base-cased': BertTokenizer,
}

MODEL_CLASSSES = {
    'gpt2': AutoModelForCausalLM,
    'cl-tohoku/bert-base-japanese': BertForMaskedLM,
    'bert-base-cased': BertForMaskedLM,
    'NlpHUST/vibert4news-base-cased': BertForMaskedLM,
}

def get_pretrained(model_name_or_path, device='cpu'):
    if model_name_or_path in TOKENIZER_CLASSES:
        tokenizer_class = TOKENIZER_CLASSES[model_name_or_path]
    else:
        tokenizer_class = AutoTokenizer
    
    if model_name_or_path in MODEL_CLASSSES:
        model_class = MODEL_CLASSSES[model_name_or_path]
    else:
        model_class = AutoModelForCausalLM

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    if model_name_or_path == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(model_name_or_path)
    model = model.to(device)
    
    return model, tokenizer

