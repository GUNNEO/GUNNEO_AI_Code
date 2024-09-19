import pandas as pd
from pathlib import Path
import torch
from torch.utils import data
import shutil
import json
import nibabel
from typing import List, Optional
from torchvision.transforms import Compose, Normalize, transforms
import libs.img_preprocessing as img_utils
import libs.text_preprocessing as text_utils


def check_file_path(
    input_path: str
):
    assert Path(input_path).exists(), f"{input_path} does not exist"
    assert Path(input_path).is_file(), f"{input_path} is not a file"


def check_dir_path(
    input_path: str
):
    assert Path(input_path).exists(), f"{input_path} does not exist"
    assert Path(input_path).is_dir(), f"{input_path} is not a directory"


def return_normalize_params(
    num_slices: int,
    modality: int
):
    means_modalities = [150.63895616116818, 124.80986430358928]
    stds_modalities = [91.48943914806172, 114.07748472505894]
    return Compose([
        Normalize((means_modalities[modality] for _ in range(
            num_slices)), (stds_modalities[modality] for _ in range(num_slices))),
        transforms.ToTensor()
    ])


class PCDataset(data.Dataset):
    def __init__(
        self,
        json_file_path: str,
        img_num_slices: int,
        tokenizer: str,
        transforms: Optional[str] = None
    ):
        super.__init__()
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            self.data_dict = json.load(json_file)
        self.img_num_slices = img_num_slices
        self.transforms = transforms

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, id):
        img_paths = self.data_dict[id]["img_path"]
        images = []
        for index, img_path in enumerate(img_paths):
            image = nibabel.load(
                Path(img_path) / "new.nii.gz").get_fdata()
            # slice the image, image shape: (h, w, d)
            d = image.shape[2]
            start = (d - self.img_num_slices) // 2
            end = start + self.img_num_slices
            image = image[:, :, start:end]
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


def load_data():
    pass


# config for csv data
def return_hnscc_config():
    return {
        "id_col": "影像号",
        "img_cols": [("T1CA", -3), ("T2A", -3)],
        "report_cols": ["mri diagnosis report"]
    }


def read_csv_data(
    text_path: str,
    id_col: str,
    img_cols: List[str],
    report_cols: List[str]
):
    check_file_path(text_path)
    csv_file = pd.read_csv(text_path)
    dict = {}

    for _, row in csv_file.iterrows():
        row_id = row[id_col]
        dict[row_id] = {
            "img_path": ["/".join(row[img_col[0]].split("/")[img_col[1]:]) for img_col in img_cols],
            "report": [row[report_col] for report_col in report_cols]
        }
    return dict


def mv_file(
    data_dict: dict,
    ori_path: str,
    new_path: str
):
    json_output_path = Path(__file__).parent.parent / "datasets/hnscc.json"
    for key, value in data_dict.items():
        img_paths = value["img_path"]
        for index, img_path in enumerate(img_paths):
            ori_img_path = Path(ori_path) / img_path
            check_dir_path(ori_img_path)
            new_img_path = Path(new_path) / img_path
            shutil.copytree(ori_img_path, new_img_path, dirs_exist_ok=True)
            # update new img_path here
            data_dict[key]["img_path"][index] = str(new_img_path)
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)


def process_img(
    json_file_path: str
):
    check_file_path(json_file_path)
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data_dict = json.load(json_file)
    for key, value in data_dict.items():
        img_paths = value["img_path"]
        for index, img_path in enumerate(img_paths):
            check_dir_path(img_path)
            img_utils.convert_dcm2nii(
                input_path=img_path, output_path=img_path + "/new.nii.gz")
