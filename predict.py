import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import torch
from osgeo import gdal
from torch.utils import data
from models.fcn_resnet import fcn_resnet50
import torch.nn.functional as F
import rasterio

#1 DEm visualization color band
# -----------------------------------------------------
hsv_colors = [(0, 89 / 100, 73 / 100), (60 / 360, 25 / 100, 100 / 100), (109 / 360, 77 / 100, 57 / 100)]
rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsv_colors]
rgb_colors = rgb_colors[::-1]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", rgb_colors)
print('RGB for 0:', rgb_colors[0])
print('RGB for 0.5:', rgb_colors[1])
print('RGB for 1:', rgb_colors[2])

#2 label color band
colors = ['#73b273', '#fff564', '#f0c567', '#de9e66']
colors = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in colors]
colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
lmap = mcolors.ListedColormap(colors)
# -----------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_tif(path):
    #3 Read tif file using gdal library
    dataset = gdal.Open(path)
    cols = dataset.RasterXSize
    #print(cols)
    rows = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, cols, rows)
    del dataset
    return im_data


def data_transfer(img):
    # Remove invalid values
    img[img > 10000] = 9000
    img[img < -1000] = 9000

    # Normalize
    img = (img + 1000) / 10000.0

    # Convert to tensor
    img = torch.from_numpy(img).float()  # Shape: (H, W) or (C, H, W)

    if img.dim() == 2:
        # If single channel, replicate to create 3 channels
        img = img.unsqueeze(0)  # Shape: (1, H, W)
        img = img.repeat(3, 1, 1)  # Shape: (3, H, W)
    elif img.dim() == 3:
        # If already multiple channels
        if img.shape[0] == 1:
            # Single-channel image with channel dimension
            img = img.repeat(3, 1, 1)  # Shape: (3, H, W)
        elif img.shape[0] == 3:
            # Already has 3 channels
            pass  # Do nothing
        else:
            # Handle images with other channel counts if necessary
            pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # Return image without adding extra batch dimension
    return img  # Shape: (3, H, W)



def visualize(img, pred, output_path):
    '''
    :param img: numpy array with shape (1024, 1024)
    :param pred: numpy array with shape (5, 1024, 1024)
    '''

    fig, axs = plt.subplots(3, 2, figsize=(8, 12))
    img = img * 10000 - 1000
    im = axs[0, 0].imshow(img, cmap=cmap)
    cbar = fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)
    axs[0, 0].set_title('Original DEM')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    result = np.argmax(pred, axis=0)
    im = axs[0, 1].imshow(result, cmap=lmap, vmin=0, vmax=4)
    cbar = fig.colorbar(im, ax=axs[0, 1], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['0', '1', '2', '3'])
    axs[0, 1].set_title('Prediction')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    pred1 = pred[1]
    im = axs[1, 0].imshow(pred1, cmap='bwr', vmin=-15, vmax=15)
    cbar = fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)
    axs[1, 0].set_title('Prediction of Class 1')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    pred2 = pred[2]
    im = axs[1, 1].imshow(pred2, cmap='bwr', vmin=-15, vmax=15)
    cbar = fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
    axs[1, 1].set_title('Prediction of Class 2')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    pred3 = pred[3]
    im = axs[2, 0].imshow(pred3, cmap='bwr', vmin=-15, vmax=15)
    cbar = fig.colorbar(im, ax=axs[2, 0], fraction=0.046, pad=0.04)
    axs[2, 0].set_title('Prediction of Class 3')
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])

    pred4 = pred[4]
    im = axs[2, 1].imshow(pred4, cmap='bwr', vmin=-15, vmax=15)
    cbar = fig.colorbar(im, ax=axs[2, 1], fraction=0.046, pad=0.04)
    axs[2, 1].set_title('Prediction of Class 4')
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])

    plt.savefig(output_path, dpi=300)
    plt.show()


def load_support_set(support_set_dir):
    classes = os.listdir(support_set_dir)
    support_images = []
    support_labels = []
    class_to_idx = {}
    idx_to_class = {}
    for idx, cls in enumerate(sorted(classes)):
        class_to_idx[cls] = idx
        idx_to_class[idx] = cls
        class_dir = os.path.join(support_set_dir, cls)
        images = os.listdir(class_dir)
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            img = read_tif(img_path)
            img = data_transfer(img)  # Converts to tensor and normalizes
            support_images.append(img)
            support_labels.append(idx)
    support_labels = torch.tensor(support_labels)  # Shape: [N]
    return support_images, support_labels, class_to_idx, idx_to_class


def compute_prototypes(support_images, support_labels, model, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for img, label in zip(support_images, support_labels):
            img = img.to(device)  # Shape: (3, H, W)
            embedding = model(img.unsqueeze(0))  # Add batch dimension: (1, 3, H, W)
            embedding = embedding.squeeze(0)  # Remove batch dimension: (2048, H', W')
            # Flatten spatial dimensions
            embedding = embedding.view(embedding.shape[0], -1).permute(1, 0)  # Shape: (H'*W', 2048)
            embeddings.append(embedding)
            # Labels: all pixels have the same class label
            labels.append(torch.full((embedding.shape[0],), label, dtype=torch.long, device=device))
    embeddings = torch.cat(embeddings, dim=0)  # Shape: (N_pixels, 2048)
    labels = torch.cat(labels, dim=0)  # Shape: (N_pixels)
    prototypes = []
    classes = torch.unique(labels)
    for cls in classes:
        cls_embeddings = embeddings[labels == cls]
        prototype = cls_embeddings.mean(dim=0)
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)  # Shape: [num_classes, 2048]
    return prototypes


def visualize(img, pred, output_path):
    '''
    :param img: numpy array with shape (H, W) or (3, H, W)
    :param pred: numpy array with shape (H, W)
    '''
    # If img has 3 channels, take the first one
    if img.ndim == 3 and img.shape[0] == 3:
        img = img[0]  # Shape: (H, W)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    img = img * 10000 - 1000
    im = axs[0].imshow(img, cmap=cmap)
    cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title('Original DEM')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    im = axs[1].imshow(pred, cmap=lmap, vmin=0, vmax=4)
    cbar = fig.colorbar(im, ax=axs[1], ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['0', '1', '2', '3', '4'])
    axs[1].set_title('Prediction')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.savefig(output_path, dpi=300)
    plt.show()


def main():
    support_set_dir = 'examples/support_set'

    if not os.path.exists('output_figs'):
        os.mkdir('output_figs')

    # Load model
    model = fcn_resnet50()
    model.to(device)
    print(f"Model initialized")

    # Load support set
    support_images, support_labels, class_to_idx, idx_to_class = load_support_set(support_set_dir)
    support_images = torch.stack(support_images)  # Shape: [N, 3, H, W]

    # Compute prototypes
    prototypes = compute_prototypes(support_images, support_labels, model, device)

    # Load query images
    data_list = [f for f in os.listdir('examples') if f.endswith('.tif')]
    for img_name in data_list:
        path = os.path.join('examples', img_name)
        img_array = read_tif(path)
        img = data_transfer(img_array)  # Shape: (3, H, W)
        img = img.to(device)

        # Inference
        model.eval()
        with torch.no_grad():
            embedding = model(img.unsqueeze(0))  # Shape: (1, 2048, H', W')
            embedding = embedding.squeeze(0)  # Shape: (2048, H', W')
            h, w = embedding.shape[1], embedding.shape[2]
            embedding = embedding.view(embedding.shape[0], -1).permute(1, 0)  # Shape: (H'*W', 2048)
            # Compute distances to prototypes
            dists = torch.cdist(embedding, prototypes.to(device))  # Shape: (H'*W', num_classes)
            pred_idxs = torch.argmin(dists, dim=1)  # Shape: (H'*W')
            pred_idxs = pred_idxs.view(h, w)  # Shape: (H', W')

            # Upsample to original image size
            pred_idxs = pred_idxs.unsqueeze(0).unsqueeze(0).float()
            pred_idxs = F.interpolate(pred_idxs, size=img.shape[1:], mode='nearest')
            pred_idxs = pred_idxs.squeeze().cpu().numpy()  # Shape: (H, W)

        # Save prediction as .tif
        pred_tif_path = os.path.join('output_figs', img_name.replace('.tif', '_pred.tif'))
        with rasterio.open(
            pred_tif_path,
            'w',
            driver='GTiff',
            height=pred_idxs.shape[0],
            width=pred_idxs.shape[1],
            count=1,
            dtype='uint8'
        ) as dst:
            dst.write(pred_idxs.astype('uint8'), 1)

        # Visualization
        output_path = os.path.join('output_figs', 'pred_' + img_name.replace('.tif', '.png'))
        visualize(img.cpu().detach().numpy(), pred_idxs, output_path)

if __name__ == '__main__':
    main()
