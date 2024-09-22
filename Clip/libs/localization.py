from pathlib import Path
from typing import List
import json
import pandas as pd
import shutil
import libs.img_preprocessing as img_utils


def check_file_path(
    input_path: str
):
    if not Path(input_path).exists():
        return False
    if not Path(input_path).is_file():
        return False
    return True


def check_dir_path(
    input_path: str
):
    if not Path(input_path).exists():
        return False
    if not Path(input_path).is_dir():
        return False
    return True

# config for csv data


def _return_hnscc_config():
    return {
        "img_cols": [("T1CA", ["头颈单独MR淋巴结清扫（王钰）2023.5.5/"]), ("T2A", ["头颈单独MR淋巴结清扫（王钰）2023.5.5/"])],
        "report_cols": ["mri diagnosis report"]
    }


def _return_cc_config():
    return {
        "img_cols": [("T1CA", ["DICOM-ROI/", "DICOM-WEIHUA ROI/"]), ("T2A", ["DICOM-ROI/", "DICOM-WEIHUA ROI/"])],
        "report_cols": ["mri diagnosis report"]
    }


def read_csv_data(
    text_path: str,
    ori_path: str,
    img_cols: List[str],
    report_cols: List[str]
):
    if not check_file_path(text_path):
        raise RuntimeError("text path is invalid")
    csv_file = pd.read_csv(text_path)
    dict = {}

    dict_index = 0
    for _, row in csv_file.iterrows():
        img_path_ls = []
        store_flag = True
        for img_col in img_cols:
            for split_i, split_w in enumerate(img_col[1]):
                if split_w in row[img_col[0]]:
                    img_path = Path(ori_path) / \
                        row[img_col[0]].split(split_w, 1)[1]
                    if check_dir_path(img_path):
                        img_path_ls.append(
                            row[img_col[0]].split(split_w, 1)[1])
                    else:
                        store_flag = False
                    break
                elif split_i == len(img_col[1]) - 1:
                    store_flag = False
        if store_flag:
            dict[str(dict_index)] = {
                "img_path": img_path_ls,
                "report": [row[report_col] for report_col in report_cols]
            }
            dict_index += 1
    return dict


def mv_file(
    data_dict: dict,
    ori_path: str,
    new_path: str,
    saved_name: str
):
    json_output_path = Path(__file__).parent.parent / \
        f"datasets/{saved_name}.json"
    for key, value in data_dict.items():
        img_paths = value["img_path"]
        for index, img_path in enumerate(img_paths):
            ori_img_path = Path(ori_path) / img_path
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
