import numpy as np
import torch
from torch.utils import data
import json
import nibabel
from typing import Optional
from pathlib import Path
from torchvision.transforms import Compose, Normalize, transforms


def _return_normalize_params(
    num_slices: int,
    modality: int
):
    means_modalities = [150.63895616116818, 124.80986430358928]
    stds_modalities = [91.48943914806172, 114.07748472505894]
    mean = [means_modalities[modality]] * num_slices
    std = [stds_modalities[modality]] * num_slices
    return Compose([
        transforms.ToTensor(),
        Normalize(mean=mean, std=std)
    ])


# build for vit2d model
class PCDataset(data.Dataset):
    def __init__(
        self,
        json_file_path: str,
        img_num_slices: int,
        transforms: Optional[callable] = None
    ):
        super().__init__()
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            self.data_dict = json.load(json_file)
        self.img_num_slices = img_num_slices
        self.transforms = transforms

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, id):
        id = str(id)
        img_paths = self.data_dict[id]["img_path"]
        images = []
        for index, img_path in enumerate(img_paths):
            image = nibabel.load(
                Path(img_path) / "new.nii.gz").get_fdata()
            # slice the image, image shape: (h, w, d)
            d = image.shape[2]
            indices = np.linspace(0, d - 1, num=self.img_num_slices, dtype=int)
            indices = np.clip(indices, 0, d - 1)
            image = image[:, :, indices]
            if self.transforms:
                transform = self.transforms(
                    num_slices=self.img_num_slices, modality=index)
                image = transform(image)
            else:
                image = torch.tensor(image)
            images.append(image)
        images = torch.stack(images)

        texts = self.data_dict[id]["report"]

        sample = {"images": images, "texts": texts}

        return sample


def load_data(
    json_file_path: str,
    batch_size: int = 4,
    img_resolution: Optional[int] = None,
    img_num_slices: int = 6,
    pretrained: bool = False,
    verbose: bool = False
):
    if torch.cuda.is_available():
        dev = "cuda"
        cuda_available = True
        print('using CUDA to load data')
    else:
        dev = "cpu"
        cuda_available = False
        print('Using cpu to load data')

    device = torch.device(dev)

    if cuda_available:
        torch.cuda.set_device(device)

    if pretrained:
        img_resolution = 224
    else:
        img_resolution = img_resolution if img_resolution is not None else 224
    img_transforms = _return_normalize_params

    torch_dset = PCDataset(json_file_path=json_file_path,
                           img_num_slices=img_num_slices, transforms=img_transforms)

    # show basic info of data
    if verbose:
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            print(i, sample['images'].size(), sample['texts'])
            if i == 3:
                break

    loader_params = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 0
    }

    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device
