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

    for i,score in enumerate(scorer.score_sentences(sentences)):
        print(score, '\t', sentences[i])
    
    print('-' * 80)

    english_sentences = [
        'Hello world!',
        'I am a student',
        'He am a student',
    ]
    model, tokenizer = get_pretrained('gpt2', device='cuda:0')
    scorer = LMScorer(model, tokenizer, device='cuda:0')
    for i,score in enumerate(scorer.score_sentences(english_sentences, normalize=True)):
        print(score, '\t', english_sentences[i])
    print('-' * 80)
