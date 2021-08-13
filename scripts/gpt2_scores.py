# coding=utf-8
"""Test scores for GPT-2 model
"""
from jmlm.models import get_pretrained
from jmlm.scorers import LMScorer

if __name__ == "__main__":
    model, tokenizer = get_pretrained('rinna/japanese-gpt2-medium', device='cpu')

    scorer = LMScorer(model, tokenizer, device='cpu')

    sentences = [
        '日本語は面白いです。',
        '日本語は硬いです。',
        'せのもんはなんですか。',
        '専門はないです。',
        'せんもんはないです。'
    ]

    for i,score in enumerate(scorer.score_sentences(sentences)):
        print(score, '\t', sentences[i])
