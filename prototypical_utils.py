import os
from osgeo import gdal
from predict import data_transfer, read_tif
import torch
import numpy as np

def load_support_set(support_set_dir):
    classes = os.listdir(support_set_dir)
    support_images = []
    support_labels = []
    class_to_idx = {}
    idx_to_class = {}
    for idx, cls in enumerate(classes):
        class_to_idx[cls] = idx
        idx_to_class[idx] = cls
        class_dir = os.path.join(support_set_dir, cls)
        images = os.listdir(class_dir)
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            img = read_tif(img_path)
            img = data_transfer(img)
            support_images.append(img)
            support_labels.append(idx)
    support_images = torch.stack(support_images)  # Shape: [N, 1, H, W]
    support_labels = torch.tensor(support_labels)  # Shape: [N]
    return support_images, support_labels, class_to_idx, idx_to_class
