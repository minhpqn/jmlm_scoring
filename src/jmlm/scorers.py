# coding=utf-8
"""Maksed Language Model Scoring class
"""
from typing import List

import numpy as np
import torch
from torch.nn.functional import softmax


class LMScorer():

    def __init__(self, model, tokenizer, device='cpu', eos=None):
        self._model = model
        self._tokenizer = tokenizer
        self._eos = eos
        self._max_length = 1024
        self._device = device

        self._model.to(self._device)
    
    def score_sentences(self, lines: List[str], per_token: bool=False):
        lines = [self._tokenizer.bos_token + line + self._tokenizer.eos_token for line in lines]

        tok_res = self._tokenizer.batch_encode_plus(lines, return_tensors='pt', add_special_tokens=False, 
                                                    padding=True)
        input_ids = tok_res['input_ids'].to(self._device)
        attention_mask = tok_res['attention_mask'].to(self._device)
        lines_len = torch.sum(tok_res['attention_mask'], dim=1)

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
                token_score = torch.log(token_prob[token_id]).cpu().numpy().item()
                token_scores.append(token_score)
                line_log_prob += token_score
            log_probs.append(line_log_prob)
            list_token_scores.append(token_scores)

        if per_token:
            return list_token_scores
        else:
            return log_probs

    def score_sentence(self, text: str, per_token: bool=False):
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

    def score_sentences(self, sentences: List[str]):
        return 0.0

    def score_sentence(self, text: str):
        tokenized_text = self.tokenizer.tokenize(text)
        text_length = len(tokenized_text)
        input_ids = self.tokenizer.encode(text)
        # print(input_ids)
        masked_input_ids = []
        masked_indexes = []
        for i in range(1, len(input_ids)-1):
            masked_token_ids = input_ids.copy()
            masked_token_ids[i] = self.tokenizer.mask_token_id
            masked_input_ids.append(masked_token_ids)
            masked_indexes.append(i)
        
        # print(masked_input_ids)
        # encoded_inputs = self.tokenizer.batch_encode_plus(masked_input_ids)
        # print(encoded_inputs)

        masked_input_ids = torch.LongTensor(masked_input_ids)

        with torch.no_grad():
            outputs = self.model(masked_input_ids)
            logits = outputs.logits
            # print(logits)
        logits = softmax(logits, dim=1)
        token_scores = []
        log_prob_score = 0.0
        for i, mask_index in enumerate(masked_indexes):
            probs = logits[i]
            # probs = softmax(probs, dim=0)
            masked_token_id = input_ids[mask_index]
            print(mask_index, masked_token_id, self.tokenizer.convert_ids_to_tokens([masked_token_id])[0])
            log_prob = np.log(probs[mask_index, masked_token_id])
            print(probs[mask_index, masked_token_id], log_prob)
            token_scores.append(log_prob)
            log_prob_score += log_prob
            # print(probs.shape)

            #             with torch.no_grad():
            #     predictions = model(tokens_tensor)

            # predicted_index = torch.argmax(predictions[0, masked_index]).item()
            # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            # print(predicted_token)

        print(token_scores)
        return log_prob_score
