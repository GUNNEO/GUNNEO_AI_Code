from pathlib import Path
from typing import List
import json
import pandas as pd
import shutil
import libs.img_preprocessing as img_utils


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

# config for csv data


def return_hnscc_config():
    return {
        "img_cols": [("T1CA", -3), ("T2A", -3)],
        "report_cols": ["mri diagnosis report"]
    }


def read_csv_data(
    text_path: str,
    img_cols: List[str],
    report_cols: List[str]
):
    check_file_path(text_path)
    csv_file = pd.read_csv(text_path)
    dict = {}

    for index, row in csv_file.iterrows():
        row_id = str(index)
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
