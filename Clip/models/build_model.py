import torch
import models_info
from pathlib import Path
import requests
from tqdm import tqdm
from typing import Optional, Union


_MODELS = {
    # pretrained_text_encoder
    "ClinicalBERT": "https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/pytorch_model.bin",
    "BlueBERT-B": "https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12/resolve/main/pytorch_model.bin",
    "BlueBERT-L": "https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16/resolve/main/pytorch_model.bin",

    # pretrained_vision_encoder
    "ViT-B/16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
    "ViT-B/32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    "ViT-L/16": "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
    "ViT-L/32": "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
}


_CONFIGS = {
    "ClinicalBERT": "https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/config.json",
    "BlueBERT-B": "https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12/resolve/main/config.json",
    "BlueBERT-L": "https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16/resolve/main/config.json"
}


def _download(key: str, url: str, root: str):
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    filename = Path(url).name
    if filename in ["pytorch_model.bin", "config.json"]:
        filename = key + "_" + filename
    download_target = root_path / filename

    if download_target.exists():
        if not download_target.is_file():
            raise RuntimeError(
                f"{download_target} exists and is not a regular file")
        else:
            return download_target
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB',
                            unit_scale=True, unit_divisor=1024)

        with open(download_target, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError(
                "WARNING: Downloaded size does not match expected size")
    else:
        raise RuntimeError(
            f"Failed to download file with status code: {response.status_code}")
    return download_target


def text_pretrained_default_config():
    return {
        "name": "ClinicalBERT",
        "download_root": Path.home() / ".cache" / "clip",
        "context_length": 128
    }


def vision_pretrained_default_config():
    return {
        "name": "ViT-B/16",
        "download_root": Path.home() / ".cache" / "clip",
        "num_channels": 3
    }


def clip_default_params():
    return {
        # params for clip model
        "embed_dim": 256,
        "vision_pretrained": {"pretrained": False, "model_name": None, "model_path": None},
        "image_size": 224,
        "patch_size": 16,
        "vision_channels": 1,
        "vision_layers": 12,
        "vision_hidden_dim": 768,
        "text_pretrained": {"pretrained": False, "model_name": None, "model_path": None, "config_path": None},
        "context_length": 77,
        "vocab_size": 49408,
        "text_layers": 12,
        "text_headers": 8,
        "text_hidden_dim": 768,
        "output_patches": 49,
        "conv_stem_configs": None
    }


def check_keys_type(dict1: dict, dict2: dict):
    for key in dict1:
        if type(dict1[key]) is not type(dict2[key]):
            raise KeyError(
                f"Type mismatch for key '{key}': Expected {type(dict1[key])}, but got {type(dict2[key])}")


def load_clip(
    num_img_modalities: int,

    # pretrained text model specific params
    text_pretrained: bool = False,
    text_pretrained_params: Optional[dict] = None,

    # pretrained vision models specific params
    vision_pretrained: bool = False,
    vision_pretrained_params: Optional[dict] = None,

    # clip model specific params
    clip_params: Optional[dict] = None,

    # device setting
    device: Union[str,
                  torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
):
    # load default set up of clip
    if clip_params is None:
        clip_params = clip_default_params()
    else:
        # check keys
        default_keys = set(clip_default_params().keys())
        input_keys = set(clip_params.keys())
        torch._assert(default_keys == input_keys,
                      f"params are not supported by clip, expected keys: {default_keys}, but got {input_keys}")
        check_keys_type(default_keys, input_keys)

    # load text pretrained model configuration if text_pretrained is set to True
    if text_pretrained:
        if text_pretrained_params is None:
            text_pretrained_params = text_pretrained_default_config()
        else:
            default_pretrained_keys = set(
                text_pretrained_default_config().keys())
            input_pretrained_keys = set(text_pretrained_params.keys())
            torch._assert(default_pretrained_keys == input_pretrained_keys,
                          f"text_pretrained_params not supported, expected keys: {default_pretrained_keys}, but got {input_pretrained_keys}")
            check_keys_type(default_pretrained_keys, input_pretrained_keys)

        # download pretrained model
        name = text_pretrained_params["name"]
        download_root = text_pretrained_params["download_root"]
        if name in _MODELS:
            model_path = _download(name, _MODELS[name], download_root)
        elif Path(name).isfile():
            model_path = name
        else:
            raise RuntimeError(
                f"Model {name} not found; available text models = {list(_MODELS.keys())[:3]}")
        if name in _CONFIGS:
            config_path = _download(name, _CONFIGS[name], download_root)
        elif Path(name).isfile():
            config_path = name
        else:
            raise RuntimeError(
                f"Model {name}'s configuration not found; available text models = {list(_MODELS.keys())[:3]}")
        clip_params["text_pretrained"]["pretrained"] = True
        clip_params["text_pretrained"]["model_name"] = name
        clip_params["text_pretrained"]["model_path"] = model_path
        clip_params["text_pretrained"]["config_path"] = config_path
        clip_params["context_length"] = text_pretrained_params["context_length"]

    # load vision pretrained model configuration if vision_pretrained is set to True
    if vision_pretrained:
        if vision_pretrained_params is None:
            vision_pretrained_params = vision_pretrained_default_config()
        else:
            default_pretrained_keys = set(
                vision_pretrained_default_config().keys())
            input_pretrained_keys = set(vision_pretrained_params.keys())
            torch._assert(default_pretrained_keys == input_pretrained_keys,
                          f"vision_pretrained_params not supported, expected keys: {default_pretrained_keys}, but got {input_pretrained_keys}")
            check_keys_type(default_pretrained_keys, input_pretrained_keys)

        # download pretrained model
        name = vision_pretrained_params["name"]
        download_root = vision_pretrained_params["download_root"]
        if name in _MODELS:
            model_path = _download(name, _MODELS[name], download_root)
        elif Path(name).isfile():
            model_path = name
        else:
            raise RuntimeError(
                f"Model {name} not found; available vision models = {list(_MODELS.keys())[3:]}")
        clip_params["vision_pretrained"]["pretrained"] = True
        clip_params["vision_pretrained"]["model_name"] = name
        clip_params["vision_pretrained"]["model_path"] = model_path
        clip_params["vision_channels"] = vision_pretrained_params["num_channels"]
    model = models_info.CLIP(
        num_img_modalities=num_img_modalities, **clip_params)
    model.to(device)
    if not text_pretrained:
        models_info.convert_weights(model)  # convert model's precision to fp16
    model.eval()
    return model
