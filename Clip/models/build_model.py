import torch
import models
from pathlib import Path
import urllib
from tqdm import tqdm
from typing import Optional, Union


_MODELS = {
    # pretrained_text_encoder

    # pretrained_vision_encoder
    "ViT-B/16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
    "ViT-B/32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    "ViT-L/16": "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
    "ViT-L/32": "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
}


def _download(url: str, root: str):
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    filename = Path(url).name
    download_target = root_path / filename

    if download_target.exists():
        if not download_target.is_file():
            raise RuntimeError(
                f"{download_target} exists and is not a regular file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, download_target.open("wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def pretrained_default_config():
    return {
        "name": "ViT-B/16",
        "download_root": Path.home() / ".cache" / "clip"
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
        "text_pretrained": {},
        "context_length": 77,
        "vocab_size": 49408,
        "text_layers": 12,
        "text_headers": 8,
        "text_hidden_dim": 768,
        "output_patches": 49
    }


def check_keys_type(dict1: dict, dict2: dict):
    for key in dict1:
        if type(dict1[key]) is not type(dict2[key]):
            raise KeyError(
                f"Type mismatch for key '{key}': Expected {type(dict1[key])}, but got {type(dict2[key])}")


def load_clip(
    num_img_modalities: int,

    # pretrained visual and text models specific params
    pretrained: bool = False,
    pretrained_params: Optional[dict] = None,

    # clip model specific params
    params: Optional[dict] = None,

    # device setting
    device: Union[str,
                  torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
):
    # load default set up of clip
    if params is None:
        params = clip_default_params()
    else:
        # check keys
        default_keys = set(clip_default_params().keys())
        input_keys = set(params.keys())
        torch._assert(default_keys == input_keys,
                      f"params are not supported by clip, expected keys: {default_keys}, but got {input_keys}")
        check_keys_type(default_keys, input_keys)

    if pretrained:
        if pretrained_params is None:
            pretrained_params = pretrained_default_config()
        else:
            default_pretrained_keys = set(pretrained_default_config().keys())
            input_pretrained_keys = set(pretrained_params.keys())
            torch._assert(default_pretrained_keys == input_pretrained_keys,
                          f"pretrained_params not supported, expected keys: {default_pretrained_keys}, but got {input_pretrained_keys}")
            check_keys_type(default_pretrained_keys, input_pretrained_keys)

        # download pretrained model
        name = pretrained_params["name"]
        download_root = pretrained_params["download_root"]
        if name in _MODELS:
            model_path = _download(_MODELS[name], download_root)
        elif Path(name).isfile():
            model_path = name
        else:
            raise RuntimeError(
                f"Model {name} not found; available models = {list(_MODELS.keys())}")
        params["vision_pretrained"]["pretrained"] = True
        params["vision_pretrained"]["model_name"] = name
        params["vision_pretrained"]["model_path"] = model_path
        params["vision_channels"] = 3
    model = models.CLIP(num_img_modalities=num_img_modalities, **params)
    model.to(device)
    models.convert_weights(model)
    model.eval()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = torch.randn(4, 20, 3, 224, 224).to(device)
text = torch.randint(0, 49408, (20, 77)).to(device)
model = load_clip(num_img_modalities=4, pretrained=True)
i, t = model(image, text)
print(i.shape, t.shape)
