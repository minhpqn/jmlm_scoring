# coding=utf-8
"""Calculate scores based on GPT-NEO-XX models and fine-tuned models
"""
from jmlm.models import get_pretrained
from jmlm.scorers import LMScorer
from torch.nn.functional import normalize

if __name__ == '__main__':
    english_sentences = [
        'Hello world!',
        'I am a student',
        'He am a student',
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
        '根が新しモノ好きですから。こういう新しい製品を見て、遠巻きに見ているだけの人は、次世代のビジネスパーソンとしては向かない。',
        '根が新しモノ好きですから。ただし、製品自体は、各社ともに一長一短があります。'
    ]
    model_name_or_path = '/mnt/disk1/minhpham/projects/gpt3-neox-training/finetune-gpt2xl/finetuned_1GB_ja_text_1epoch'
    model, tokenizer = get_pretrained(model_name_or_path, device='cuda:0')
    scorer = LMScorer(model, tokenizer, device='cuda:0')
    for i,score in enumerate(scorer.score_sentences(english_sentences, normalize=True)):
        print(score, '\t', english_sentences[i])
    print('-' * 80)
