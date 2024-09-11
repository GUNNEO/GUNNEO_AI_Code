import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import List, Union
from tokenizer import SimpleTokenizer


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
    id = csv_file[id_col]
    text = csv_file[text_col]
    return id, text


def return_img_path(
    text_path: str,
    id_col: str,
    img_col: str
):
    check_path(text_path)
    pass


def preprocess_text(
    texts: List,
    model: nn.Module
):
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
    return result
