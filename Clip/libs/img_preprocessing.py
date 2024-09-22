import json
import numpy as np
import cv2
import nibabel
import pydicom
from pydicom.errors import InvalidDicomError
from pathlib import Path as p
import matplotlib.pyplot as plt
from pathlib import Path


'''the input of the img should be a np.ndarray'''


def _resample_func(
    img: np.ndarray,
    target_dim: int,
    method: str
):
    if not method:
        return img
    scale_factor = target_dim / min(img.shape)
    new_size = (int(img.shape[1] * scale_factor),
                int(img.shape[0] * scale_factor))
    if method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'cubic':
        interpolation = cv2.INTER_CUBIC
    elif method == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError('Unsupported resampling method')
    resample_img = cv2.resize(img, new_size, interpolation=interpolation)
    return resample_img


def _smart_crop(
    img: np.ndarray,
    target_dim: int
):
    h, w = img.shape
    target_h, target_w = target_dim, target_dim
    max_intensity_index = np.unravel_index(np.argmax(img), img.shape)
    start_y = max(0, min(max_intensity_index[0] - target_h // 2, h - target_h))
    start_x = max(0, min(max_intensity_index[1] - target_w // 2, w - target_w))

    cropped_img = img[start_y:start_y + target_h, start_x:start_x + target_w]
    return cropped_img


def _convert_uname2dicom(
    folder_path: str
):
    folder_path = Path(folder_path)
    for dicom_folder in folder_path.iterdir():
        if dicom_folder.is_dir():
            for file_path in dicom_folder.rglob('*'):
                if file_path.is_file() and file_path.suffix == '':
                    new_file_path = file_path.with_suffix('.dcm')
                    file_path.rename(new_file_path)


def _load_dicom_files(
    dicom_dir: str
):
    dicom_dir = p(dicom_dir)
    dicom_files = sorted(
        [f for f in dicom_dir.iterdir() if f.suffix == '.dcm'])
    try:
        slices = [pydicom.dcmread(str(dicom_file))
                  for dicom_file in dicom_files]
    except (InvalidDicomError, FileNotFoundError, IsADirectoryError):
        raise RuntimeError(f"can not load {dicom_dir}")
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def _dicom_to_numpy(
    input_path: str,
    dicom_slices: list,
):
    pixel_arrays = [s.pixel_array for s in dicom_slices]
    try:
        image_3d = np.stack(pixel_arrays, axis=-1)
    except (ValueError, TypeError):
        raise RuntimeError(f"error shape, path: {input_path}")

    intercept = dicom_slices[0].RescaleIntercept if 'RescaleIntercept' in dicom_slices[0] else 0
    slope = dicom_slices[0].RescaleSlope if 'RescaleSlope' in dicom_slices[0] else 1

    if slope != 1:
        image_3d = slope * image_3d.astype(np.float64)
        image_3d = image_3d.astype(np.int16)

    image_3d += np.int16(intercept)

    return image_3d


def _process_image_3d(
    image_3d: np.ndarray,
    target_dim: int,
    method: str
):
    processed_slices = []
    for index in range(image_3d.shape[2]):
        slice = image_3d[:, :, index]
        resampled_slice = _resample_func(slice, target_dim, method)
        cropped_slice = _smart_crop(resampled_slice, target_dim)
        processed_slices.append(cropped_slice)

    processed_image_3d = np.stack(processed_slices, axis=-1)
    return processed_image_3d


def _save_as_nii(
    image_3d: np.ndarray,
    output_path: str,
    dicom_slices: list
):
    affine = np.eye(4)
    nii_image = nibabel.Nifti1Image(image_3d, affine)
    nibabel.save(nii_image, output_path)


def convert_dcm2nii(
    input_path: str,
    output_path: str,
    target_dim: int = 224,
    method: str = "nearest"
):
    dicom_slices = _load_dicom_files(dicom_dir=input_path)
    image_3d = _dicom_to_numpy(
        input_path=input_path, dicom_slices=dicom_slices)
    processed_image_3d = _process_image_3d(
        image_3d=image_3d, target_dim=target_dim, method=method)
    _save_as_nii(image_3d=processed_image_3d, output_path=output_path,
                 dicom_slices=dicom_slices)


def convert_nii2nii(
    input_path: str,
    output_path: str,
    target_dim: int = 224,
    method: str = 'nearest'
):
    input_path = p(input_path)
    output_path = p(output_path)
    for file in input_path.iterdir():
        if not file.suffix(".nii.gz"):
            continue
        nii_path = input_path / file
        image_3d = nibabel.load(nii_path).get_fdata()
        new_image_3d = _process_image_3d(
            image_3d=image_3d, target_dim=target_dim, method=method)
        new_nii = nibabel.Nifti1Image(new_image_3d, affine=np.eye(4))
        # use dir name to specify unique id
        nibabel.save(new_nii, output_path)


def cal_mean_std(
    json_file_path: str,
    modality: int,
    output_path: str
):
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data_dict = json.load(json_file)
    means = []
    stds = []
    for key, value in data_dict.items():
        img_path = data_dict[key]["img_path"][modality]
        image = nibabel.load(p(img_path) / "new.nii.gz").get_fdata()
        means.append(np.mean(image))
        stds.append(np.std(image))
    global_mean = np.mean(means)
    global_std = np.std(stds)

    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(f"Global mean: {global_mean}\n")
        out_file.write(f"Global std: {global_std}\n")

    return global_mean, global_std


# calculate the img shape distribution
def cal_img_distribution(
    input_path: str
):
    input_path = p(input_path)
    dict = {}
    area_min = (10000, 10000)
    area_max = (0, 0)
    depth_min = 1000
    depth_max = 0
    for img_path in input_path.glob('*.nii.gz'):
        img = nibabel.load(img_path).get_fdata().transpose(2, 0, 1)
        area = img.shape[1] * img.shape[2]
        depth = img.shape[0]
        shape_key = str(img.shape)
        dict[shape_key] = 1 if shape_key in dict else dict[shape_key] + 1
        if area > area_max[0] * area_max[1]:
            area_max = (img.shape[1], img.shape[2])
        if area < area_min[0] * area_min[1]:
            area_min = (img.shape[1], img.shape[2])
        if depth > depth_max:
            depth_max = depth
        if depth < depth_min:
            depth_min = depth
    sorted_shapes = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    shapes, counts = zip(*sorted_shapes)
    fig, ax = plt.subplots()
    ax.bar(shapes, counts)
    ax.set_xlabel('Image Shape')
    ax.set_ylabel('Frequency')
    ax.set_title('Image Shape Distribution')
    plt.xticks(rotation=45)
    fig.savefig(p(__file__).parent.parent / 'datasets/shape_distribution.png')
    plt.close(fig)
    file = open(p(__file__).parent.parent / 'datasets/shape.txt', 'w')
    print("area_min: ", min, file=file)
    print("area_max: ", max, file=file)
    print("depth_min: ", depth_min, file=file)
    print("depth_max: ", depth_max, file=file)
    file.close()
