import pandas as pd
from pathlib import Path
import shutil
from typing import List


def check_path(input_path: str):
    assert Path(input_path).exists(), f"{input_path} does not exist"
    assert Path(input_path).is_file(), f"{input_path} is not a file"


def return_hnscc_config():
    return {
        "id_col": "影像号",
        "img_cols": [("T1CA", -3), ("T2A", -3)],
        "report_cols": ["mri diagnosis report"]
    }


def read_data(
    text_path: str,
    id_col: str,
    img_cols: List[str],
    report_cols: List[str]
):
    check_path(text_path)
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
    for id, data in data_dict.items():
        img_paths = data["img_path"]
        for index, img_path in enumerate(img_paths):
            ori_img_path = Path(ori_path) / img_path
            check_path(ori_img_path)
            new_img_path = Path(new_path) / img_path
            shutil.copy(ori_img_path, new_img_path)
            # update new img_path here
            data_dict[id]["img_path"][index] = new_img_path
    return data_dict
