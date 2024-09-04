import os
import csv
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil


def check_folder_name(path):
    folder_name = ["T1W_mDIXON_COR+C",
                   "T1W_mDIXON_TRA+C", "T2W_mDIXON_TSE_TRA"]
    invalid_name = []
    for name in os.listdir(path):
        path1 = os.path.join(path, name)
        for name1 in os.listdir(path1):
            if not any(name1.startswith(prefix) for prefix in folder_name):
                invalid_name.append(name)
    print(len(set(invalid_name)))
    return set(invalid_name)


def check_nii_folder_name(path):
    folder_name = ["T1+C COR.nii.gz",
                   "T1+C.nii.gz", "T2.nii.gz"]
    invalid_name = []
    for name in os.listdir(path):
        path1 = os.path.join(path, name)
        for name1 in os.listdir(path1):
            if not any(name1.startswith(prefix) for prefix in folder_name):
                invalid_name.append(name)
    print(len(set(invalid_name)))
    return set(invalid_name)


def change_folder_name(path):
    folder_name = ["T1W_mDIXON_All_TRA+C"]
    new_folder_name = ["T1W_mDIXON_TRA+C"]
    for name in os.listdir(path):
        path1 = os.path.join(path, name)
        for name1 in os.listdir(path1):
            path2 = os.path.join(path1, name1)
            if name1[0:20] == folder_name[0]:
                path3 = os.path.join(
                    path1, new_folder_name[0] + name[20:])
                os.rename(path2, path3)


def check_diff(path, csv_path):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        img_num = []
        for row in reader:
            img_num.append(row[3])
    img_num = img_num[1:]
    print(len(img_num))
    for name in os.listdir(path):
        if name[-15:-7] in img_num:
            img_num.remove(name[-15:-7])
    return img_num


def convert2nii(path, mask_path):
    folder_name = ["T1W_mDIXON_COR+C",
                   "T1W_mDIXON_TRA+C", "T2W_mDIXON_TSE_TRA"]
    for name in os.listdir(path):
        path1 = os.path.join(path, name)
        mask_path1 = os.path.join(mask_path, name)
        for name1 in os.listdir(path1):
            path2 = os.path.join(path1, name1)
            if name1.startswith(folder_name[0]):
                nii_path = os.path.join(path1, "T1+C COR.nii.gz")
                mask_path2 = os.path.join(mask_path1, "T1+C COR.nii.gz")
            elif name1.startswith(folder_name[1]):
                nii_path = os.path.join(path1, "T1+C.nii.gz")
                mask_path2 = os.path.join(mask_path1, "T1+C.nii.gz")
            else:
                nii_path = os.path.join(path1, "T2.nii.gz")
                mask_path2 = os.path.join(mask_path1, "T2.nii.gz")
            mask = nib.load(mask_path2).get_fdata()
            _, _, d = mask.shape
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path2)[:d]
            reader.SetFileNames(dicom_names)
            image2 = reader.Execute()

            image_array = sitk.GetArrayFromImage(image2)  # z, y, x
            origin = image2.GetOrigin()  # x, y, z
            spacing = image2.GetSpacing()  # x, y, z
            direction = image2.GetDirection()  # x, y, z

            image3 = sitk.GetImageFromArray(image_array)
            image3.SetSpacing(spacing)
            image3.SetDirection(direction)
            image3.SetOrigin(origin)
            sitk.WriteImage(image3, os.path.join(nii_path,))


def delete_nii(path):
    nii_name = ["T1+C COR.nii.gz", "T1+C.nii.gz", "T2.nii.gz"]
    for name in os.listdir(path):
        path1 = os.path.join(path, name)
        for file in os.listdir(path1):
            if file in nii_name:
                os.remove(os.path.join(path1, file))


def check_nii_num(path):
    invalid_nii = []
    for name in os.listdir(path):
        path1 = os.path.join(path, name)
        cnt = 0
        for _ in os.listdir(path1):
            cnt += 1
        if cnt < 3:
            invalid_nii.append(path1)
    return invalid_nii


def img_mean_std(path, type):
    means = []
    stds = []
    nii_name = ["T1+C COR.nii.gz", "T1+C.nii.gz", "T2.nii.gz"]
    for name in os.listdir(path):
        path1 = os.path.join(path, name)
        for file in os.listdir(path1):
            if file == nii_name[type]:
                img = nib.load(os.path.join(path1, file)).get_fdata()
                _, _, d = img.shape
                for i in range(d):
                    slice = img[:, :, i]
                    means.append(np.mean(slice))
                    stds.append(np.std(slice))
    overall_mean = np.mean(means)
    overall_std = np.std(stds)
    return [nii_name[type][:-7] + "_mean", overall_mean, nii_name[type][:-7] + "_std", overall_std]


def split_csv_data(csv_file, train_frac=0.7, valid_frac=0.15, test_frac=0.15, random_state=42):
    data = pd.read_csv(csv_file)
    train_data, temp_data = train_test_split(
        data, train_size=train_frac, random_state=random_state)
    valid_ratio = valid_frac / (valid_frac + test_frac)
    valid_data, test_data = train_test_split(
        temp_data, train_size=valid_ratio, random_state=random_state)
    return train_data, valid_data, test_data


def delete_dir(path):
    for file in os.listdir(path):
        path1 = os.path.join(path, file)
        for file1 in os.listdir(path1):
            item_path = os.path.join(path1, file1)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)


# main_img_path = "/Users/gunneo/Documents/Dataset/370/image-before/"
# main_csv_path = "/Users/gunneo/Documents/Dataset/370/头颈癌新辅治疗时序名单-孙逸仙-370.csv"
# main_roi_path = "/Users/gunneo/Documents/Dataset/370/ROI/"
# train, val, test = split_csv_data(main_csv_path)
# train.to_csv('/Users/gunneo/Documents/Dataset/370/hnc_train.csv', index=False)
# val.to_csv('/Users/gunneo/Documents/Dataset/370/hnc_valid.csv', index=False)
# test.to_csv('/Users/gunneo/Documents/Dataset/370/hnc_test.csv', index=False)
# f = open("invalid_name.txt", "w")
# print(check_folder_name(main_img_path), file=f)
# f.close()
# change_folder_name(main_img_path)
# f1 = open("diff.txt", "w")
# print(check_diff(main_img_path, main_csv_path), file=f1)
# f1.close()
# f2 = open("invalid_roi.txt", "w")
# print(check_diff(main_roi_path, main_csv_path), file=f2)
# f2.close()
# convert2nii(main_img_path, main_roi_path)
# delete_nii(main_img_path)
# f3 = open("means_stds.json", "w")
# data = {}
# for i in range(3):
#     res = img_mean_std(main_img_path, i)
#     data[res[0]] = res[1]
#     data[res[2]] = res[3]
# data = json.dumps(data)
# print(data, file=f3)
# f3.close()
copy_img = "/Users/gunneo/Documents/Dataset/370 copy/image-before/"
copy_roi = "/Users/gunneo/Documents/Dataset/370 copy/ROI/"
delete_dir(copy_img)
delete_dir(copy_roi)
