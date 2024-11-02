import re
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import numpy as np
from scipy import io
import os


def calculate_psnr(image1, image2, max_pixel_value):
    if image1.shape != image2.shape:
        raise ValueError("The sizes of the two pictures must be the same!")
    mse = F.mse_loss(image1, image2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def get_dataset(dataset_path, dataset_gt_path):
    img = io.loadmat(dataset_path)['indian_pines_corrected']
    gt = io.loadmat(dataset_gt_path)['indian_pines_gt']
    ignored_labels = [0]
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    data = np.asarray(img, dtype='float32')
    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    return data, gt, ignored_labels


img, gt, ignored_labels = get_dataset("./data/Indian_pines_corrected.mat", "./data/Indian_pines_gt.mat")

xmax = img.max()
list = []
patch_size=5
device = torch.device('cuda')
for root, dirs, files in os.walk("./only_grad", topdown=True):
    dirs[:] = [d for d in dirs if not os.path.isdir(os.path.join(root, d))]
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)
        if file_path.endswith(".pth"):
            # match = re.match(r"\((\d+),\s*(\d+),\s*(\d+)\)\.pth", file)
            match = re.match(r"\((\d+),\s*(\d+)\)\.pth", file)
            if match:
                num1 = int(match.group(1))
                num2 = int(match.group(2))
                x = img[num1:num1 + patch_size, num2:num2 + patch_size, :]
                x = torch.from_numpy(x).transpose(2, 0).transpose(2, 1).unsqueeze(0).to(device)
                psnr = calculate_psnr(torch.load(file_path), x, xmax)
                list.append(psnr)
ave = sum(list) / len(list)
print(ave)
