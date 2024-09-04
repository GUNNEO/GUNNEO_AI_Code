import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import cv2
from PIL import Image
import nibabel
from pathlib import Path as p
import matplotlib.pyplot as plt


'''the input of the img should be a np.ndarray'''


def resample_func(img, target_dim, method):
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


def smart_crop(img, target_dim):
    h, w = img.shape
    target_h, target_w = target_dim, target_dim
    max_intensity_index = np.unravel_index(np.argmax(img), img.shape)
    start_y = max(0, min(max_intensity_index[0] - target_h // 2, h - target_h))
    start_x = max(0, min(max_intensity_index[1] - target_w // 2, w - target_w))

    cropped_img = img[start_y:start_y+target_h, start_x:start_x+target_w]
    return cropped_img


def cal_distribution(input, modality):
    input = p(input)
    input = input / modality
    input.mkdir(parents=True, exist_ok=True)
    x = []
    y = []
    min = (10000, 10000)
    max = (0, 0)
    for img_path in input.glob('*.png'):
        img = np.array(Image.open(img_path), dtype=np.uint8)
        area = img.shape[0] * img.shape[1]
        x.append(img.shape[0])
        y.append(img.shape[1])
        if area > max[0] * max[0]:
            max = (img.shape[0], img.shape[1])
        if area < min[0] * min[0]:
            min = (img.shape[0], img.shape[1])
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=10)
    fig.savefig('/Users/gunneo/AI/Codes/Clip/datasets/shape_distribution.png')
    plt.close(fig)
    file = open('/Users/gunneo/AI/Codes/Clip/datasets/shape.txt', 'w')
    print("min: ", min, file=file)
    print("max: ", max, file=file)
    file.close()


def convert_nii_png(input, output, modality, target_dim=224, method='nearest'):
    input = p(input)
    output = p(output)
    output = output / modality
    output.mkdir(parents=True, exist_ok=True)
    for dir in input.iterdir():
        if dir.is_dir():
            dir_name = dir.name
            file_name = modality + '.nii.gz'
            nii_path = dir / file_name
            img_arr = nibabel.load(nii_path).get_fdata()
            img_arr = img_arr.transpose(2, 0, 1)
            for i in range(img_arr.shape[0]):
                resample_img = resample_func(
                    img_arr[i], target_dim=target_dim, method=method)
                cropped_img = smart_crop(resample_img, target_dim=target_dim)
                saved_path = output / p(modality + '-' +
                                        dir_name[-15:-7] + '-' + str(i) + '.png')
                cropped_img = Image.fromarray(cropped_img.astype(np.uint8))
                cropped_img.save(saved_path)


def cal_mean_std(input, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=input, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

    channel_sum = torch.tensor([0.0, 0.0, 0.0])
    channel_squared_sum = torch.tensor([0.0, 0.0, 0.0])
    num_batches = 0

    for data, _ in dataloader:
        # data: [batch_size, 3, H, W]
        channel_sum += data.sum([0, 2, 3])
        channel_squared_sum += (data ** 2).sum([0, 2, 3])
        num_batches += data.size(0)

    mean = channel_sum / \
        (num_batches * dataset[0][0].size(1) * dataset[0][0].size(2))
    std = (channel_squared_sum / (num_batches *
           dataset[0][0].size(1) * dataset[0][0].size(2)) - mean ** 2) ** 0.5

    return mean, std


input = "/Users/gunneo/AI/Dataset/370 copy/image-before/"
output_png = "/Users/gunneo/AI/Dataset/370 copy/image-before-png/"
output_nii = "/Users/gunneo/AI/Dataset/370 copy/image-before-nii/"
modality = "T2"
cal_mean_std(output_png)
# convert_nii_png(input, output_png, modality, target_dim=224, method='nearest')
