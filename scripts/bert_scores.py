# coding=utf-8
"""Test MLM scoring with Japanese BERT
"""
from jmlm.models import get_pretrained
from jmlm.scorers import MLMScorer

if __name__ == "__main__":
    model, tokenizer = get_pretrained('cl-tohoku/bert-base-japanese', device='cpu')

    scorer = MLMScorer(model, tokenizer, device='cpu')

    sentences = [
        '日本語は面白いです。',
        '日本語は硬いです。',
        'せのもんはなんですか。',
        '専門はないです。',
        'せんもんはないです。'
    ]

    for text in sentences:
        score = scorer.score_sentence(text)
        print(score, text)

    model, tokenizer = get_pretrained('bert-base-cased', device='cpu')
    scorer = MLMScorer(model, tokenizer, device='cpu')
    for text in ["Hello world!", "I is a student", "I am a student"]:
        score = scorer.score_sentence(text)
        print(score, text)
