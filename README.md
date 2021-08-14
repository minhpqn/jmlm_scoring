# Japanese Masked Language Model Scoring

Copyright by Pham Quang Nhat Minh (C).

In this package, we implement [Masked Language Model Scoring](https://arxiv.org/abs/1910.14659) in Pytorch and add Japanese, Vietnamese support.

## Installation

Python 3.6+ is required. Clone this repository and install:
```bash
pip install -e .
```

## How to use the package

```python
from jmlm.models import get_pretrained
from jmlm.scorers import MLMScorer, LMScorer

# Masked Language Model scroing with cl-tohoku/bert-base-japanese
model, tokenizer = get_pretrained('cl-tohoku/bert-base-japanese', device='cpu')
scorer = MLMScorer(model, tokenizer, device='cpu')
print(scorer.score_sentences(['英語をわかる','英語がわかる']))
# >> [-27.141216278076172, -16.992878675460815]

# Take average score of tokens
print(scorer.score_sentences(['英語をわかる','英語がわかる'], normalize=True))
# >> [-9.047072092692057, -5.6642928918202715]

# Get scores for each token
print(scorer.score_sentences(['英語がわかる'], per_token=True))
# >> [[-6.728611469268799, -2.057461977005005, -8.206805229187012]]

# Masked Language Model scoring for Vietnamese
model, tokenizer = get_pretrained('NlpHUST/vibert4news-base-cased', device='cpu')
scorer = MLMScorer(model, tokenizer, device='cpu')
print(scorer.score_sentences(['Tôi là sinh viên đại học.','Tôi là học sinh đại học']))
## >> [-3.132660969684366, -7.91806149110198]

# Language Model Scoring with rinna/japanese-gpt2-medium
model, tokenizer = get_pretrained('rinna/japanese-gpt2-medium', device='cpu')
scorer = LMScorer(model, tokenizer, device='cpu')
print(scorer.score_sentences(['英語をわかる','英語がわかる']))
## >> [-21.05287978053093, -15.756769746541977]
```

