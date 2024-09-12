import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import List, Union
from tokenizer import SimpleTokenizer
from transformers import AutoTokenizer
from typing import Optional


def check_path(input_path: str):
    assert Path(input_path).exists(), f"{input_path} does not exist"
    assert Path(input_path).is_file(), f"{input_path} is not a file"


def read_text_data(
    text_path: str,
    id_col: str,
    text_col: Union[str, List[str]]
):
    check_path(text_path)
    csv_file = pd.read_csv(text_path)
    id = list(csv_file[id_col])
    text = list(csv_file[text_col])
    return id, text


def return_img_path(
    text_path: str,
    id_col: str,
    img_col: str
):
    check_path(text_path)
    pass


def return_tokenizer(model_name):
    dict = {
        "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
        "BlueBERT-B": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "BlueBERT-L": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
    }
    return dict[model_name]


def preprocess_text(
    texts: List,
    model: nn.Module,
    pretrained: Optional[str] = None
):
    if pretrained is None:
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
            return_tokenizer(pretrained), clean_up_tokenization_spaces=True)
        encoded_inputs = _tokenizer(texts, padding="max_length", truncation=True,
                                    max_length=model.context_length, return_tensors="pt")
        result = encoded_inputs['input_ids']
        mask = encoded_inputs["attention_mask"]
    return result, mask
