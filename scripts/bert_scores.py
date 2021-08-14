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
        '二本語は面白いです。',
        '日本語は硬いです。',
        'せのもんはなんですか。',
        '専門はないです。',
        'せんもんはないです。',
        '趣味は映画を見るのです',
        '趣味は映画を見ることです',
        '英語をわかる',
        '英語がわかる',
        '手紙を書きない',
        '手紙を書かない',
        '普通に学校はあまり日本語で喋らないです。',
        '普通は学校はあまり日本語で喋らないです。',
        'デザートを食べながら、チャンペンを飲みました。',
        'デザートを食べながら、シャンパンを飲みました。',
        'そのあと、スポットライトと言うクラフトの店を見に行きました。',
        'そのあと、スポットライトというクラフトの店を見に行きました。'
    ]
    
    print('cl-tohoku/bert-base-japanese')
    for text in sentences:
        score = scorer.score_sentence(text)
        print(score, text)
    print("-" * 80)

    model, tokenizer = get_pretrained('cl-tohoku/bert-base-japanese-char', device='cpu')
    scorer = MLMScorer(model, tokenizer, device='cpu')
    print('cl-tohoku/bert-base-japanese-char')
    for text in sentences:
        score = scorer.score_sentence(text)
        print(score, text)
    print("-" * 80)

    model, tokenizer = get_pretrained('cl-tohoku/bert-large-japanese', device='cuda:0')
    scorer = MLMScorer(model, tokenizer, device='cuda:0')
    print('cl-tohoku/bert-large-japanese')
    for text in sentences:
        score = scorer.score_sentence(text)
        print(score, text)
    print("-" * 80)

    model, tokenizer = get_pretrained('bert-base-cased', device='cpu')
    scorer = MLMScorer(model, tokenizer, device='cpu')
    print('bert-base-cased')
    sentences = [
        "Hello world!", 
        "I is a student", 
        "I am a student"
    ]
    
    for text in sentences:
        score = scorer.score_sentence(text)
        print(score, text)

    print("-" * 80)

    model, tokenizer = get_pretrained('NlpHUST/vibert4news-base-cased', device='cpu')
    scorer = MLMScorer(model, tokenizer, device='cpu')
    print('NlpHUST/vibert4news-base-cased')
    sentences = [
        'Tôi là sinh viên đại học.',
        'Tôi là học sinh đại học',
        'Tôi là sinh viê đại học.',
        'Tôi la sinh vien đại học',
    ]
    for text in sentences:
        score = scorer.score_sentence(text)
        print(score, text)
