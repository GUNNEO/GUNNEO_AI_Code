import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import cv2
import nibabel
from pathlib import Path as p
import matplotlib.pyplot as plt


'''the input of the img should be a np.ndarray'''


def resample_func(
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


def smart_crop(
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


# calculate the img shape distribution
def cal_img_distribution(
    input_path: str,
    modality: str
):
    input_path = p(input_path)
    input_path = input_path / modality
    input_path.mkdir(parents=True, exist_ok=True)
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


def convert_nii2nii(
    input_path: str,
    output_path: str,
    modality: str,
    target_dim: int = 224,
    method: str = 'nearest'
):
    input_path = p(input_path)
    output_path = p(output_path)
    output_path = output_path / modality
    output_path.mkdir(parents=True, exist_ok=True)
    for dir in input_path.iterdir():
        if dir.is_dir():
            dir_name = dir.name
            file_name = modality + '.nii.gz'
            nii_path = dir / file_name
            img_arr = nibabel.load(nii_path).get_fdata()
            img_arr = img_arr.transpose(2, 0, 1)
            new_img_arr = np.zeros((img_arr.shape[0], target_dim, target_dim))
            for i in range(img_arr.shape[0]):
                resample_img = resample_func(
                    img_arr[i], target_dim=target_dim, method=method)
                cropped_img = smart_crop(resample_img, target_dim=target_dim)
                new_img_arr[i] = cropped_img
            new_nii = nibabel.Nifti1Image(new_img_arr, affine=np.eye(4))
            # use dir name to specify unique id
            saved_path = output_path / f"{modality}-{dir_name}.nii.gz"
            nibabel.save(new_nii, saved_path)


def cal_mean_std(
    input_path: str,
    batch_size: int = 64
):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=input_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

    channel_sum = torch.tensor([0.0, 0.0, 0.0])
    channel_squared_sum = torch.tensor([0.0, 0.0, 0.0])
    num_batches = 0

    for data, _ in dataloader:
        # data: [batch_size, C, H, W]
        channel_sum += data.sum([0, 2, 3])
        channel_squared_sum += (data ** 2).sum([0, 2, 3])
        num_batches += data.size(0)

    mean = channel_sum / \
        (num_batches * dataset[0][0].size(1) * dataset[0][0].size(2))
    std = (channel_squared_sum / (num_batches *
           dataset[0][0].size(1) * dataset[0][0].size(2)) - mean ** 2) ** 0.5

    return mean, std
