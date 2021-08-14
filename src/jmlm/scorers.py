# coding=utf-8
"""Maksed Language Model Scoring class
"""
from typing import List

import numpy as np
import torch
from parso.python.tokenize import tokenize
from torch.functional import norm
from torch.nn.functional import softmax


class LMScorer():

    def __init__(self, model, tokenizer, device='cpu', eos=None):
        self._model = model
        self._tokenizer = tokenizer
        self._eos = eos
        self._max_length = 1024
        self._device = device

        self._model.to(self._device)
    
    def score_sentences(self, lines: List[str], normalize: bool=False, per_token: bool=False):
        lines = [self._tokenizer.bos_token + line + self._tokenizer.eos_token for line in lines]

        tok_res = self._tokenizer.batch_encode_plus(lines, return_tensors='pt', add_special_tokens=False, 
                                                    padding=True)
        input_ids = tok_res['input_ids'].to(self._device)
        attention_mask = tok_res['attention_mask'].to(self._device)
        lines_len = torch.sum(tok_res['attention_mask'], dim=1)
        lines_len = lines_len.cpu().numpy()

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss, logits = outputs[:2]

        list_token_scores = []
        log_probs = []
        for line_ind in range(len(lines)):
            line_log_prob = 0.0
            token_scores = []
            for token_ind in range(0, lines_len[line_ind]-2):
                token_prob = softmax(logits[line_ind, token_ind], dim=0)
                token_id = input_ids[line_ind, token_ind + 1]
                token_score = torch.log(token_prob[token_id])
                token_score = token_score.cpu().numpy()
                token_scores.append(token_score)
                line_log_prob += token_score
            
            if normalize:
                line_log_prob = line_log_prob / (lines_len[line_ind]-2)
            log_probs.append(line_log_prob)
            list_token_scores.append(token_scores)

        if per_token:
            return list_token_scores
        else:
            return log_probs

    def score_sentence(self, text: str, normalize: bool=False, per_token: bool=False):
        """Slow version of scoring a sentence
        """
        tokenize_input = self._tokenizer.tokenize(text)
        #50256 is the token_id for <|endoftext|>
        tensor_input = torch.tensor([ [ self._tokenizer.bos_token_id ] + self._tokenizer.convert_tokens_to_ids(tokenize_input) + [ self._tokenizer.eos_token_id ]])
        with torch.no_grad():
            outputs = self._model(tensor_input, labels=tensor_input)
            loss, logits = outputs[:2]

        log_prob = 0.0
        token_scores = []
        for i in range(0, len(tokenize_input)):
            masked_index = i
            predicted_score = logits[0, masked_index]
            predicted_prob = softmax(predicted_score, dim=0)
            score = np.log(predicted_prob[self._tokenizer.convert_tokens_to_ids([tokenize_input[i]])[0]])
            score = score.cpu().numpy()
            token_scores.append(score.item())
            log_prob += score

        if normalize:
            log_prob /= len(tokenize_input)

        if per_token:
            return token_scores
        else:
            return log_prob


class MLMScorer():

    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.model.to(device)
        self.device = device

    def score_sentences(self, sentences: List[str], normalize=False, per_token=False):
        return [self.score_sentence(text, normalize=normalize, per_token=per_token) for text in sentences]

    def score_sentence(self, text: str, normalize=False, per_token=False):
        tokenized_text = self.tokenizer.tokenize(text)

        masked_indexes = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for i in range(len(tokenized_text)):
            masked_text = tokenized_text.copy()
            masked_text[i] = '[MASK]'
            masked_indexes.append(i)
            masked_input_ids = self.tokenizer.encode(masked_text)
            input_ids.append(masked_input_ids)
            attention_mask.append([1] * len(masked_input_ids))
            token_type_ids.append([0] * len(masked_input_ids))

        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        token_type_ids = torch.tensor(token_type_ids, device=self.device)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        token_scores = []
        log_prob_score = 0.0
        for i, mask_index in enumerate(masked_indexes):
            predictions = logits[i].squeeze(0)
            probs = softmax(predictions, dim=1)
            masked_token_id = self.tokenizer.convert_tokens_to_ids([tokenized_text[mask_index]])[0]
            log_prob = np.log(probs[mask_index+1, masked_token_id].cpu().numpy()).item()
            token_scores.append(log_prob)
            log_prob_score += log_prob

        if normalize:
            log_prob_score /= len(tokenized_text)
        if per_token:
            return token_scores
        else:
            return log_prob_score
