import torch
import torch.nn as nn
from .tokenizer import SimpleTokenizer
from transformers import AutoTokenizer
from typing import Optional, List


def _return_tokenizer(model_name):
    dict = {
        "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
        "BlueBERT-B": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "BlueBERT-L": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
    }
    return dict[model_name]


def _tokenization(
    texts: tuple,
    model: nn.Module,
    pretrained_model: Optional[str] = None
):
    if pretrained_model is None:
        _tokenizer = SimpleTokenizer()
        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] +
                      _tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(
            len(all_tokens), model.context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > model.context_length:
                tokens = tokens[:model.context_length]
                tokens[model.context_length - 1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)
        mask = None
    else:
        _tokenizer = AutoTokenizer.from_pretrained(
            _return_tokenizer(pretrained_model), clean_up_tokenization_spaces=True)
        encoded_inputs = _tokenizer(texts, padding="max_length", truncation=True,
                                    max_length=model.context_length, return_tensors="pt")
        result = encoded_inputs['input_ids']
        mask = encoded_inputs["attention_mask"]
    return result, mask


def preprocess_text(
    texts: List,
    model: nn.Module,
    pretrained_model: Optional[str] = None
):
    token_texts = []
    if pretrained_model is not None:
        masks = []
    else:
        masks = None
    for single_texts in texts:
        token_text, mask = _tokenization(
            texts=single_texts, model=model, pretrained_model=pretrained_model)
        token_texts.append(token_text)
        if masks is not None:
            masks.append(mask)
    token_texts = torch.stack(token_texts, dim=1)
    if masks is not None:
        masks = torch.stack(masks, dim=1)
    return token_texts, masks
