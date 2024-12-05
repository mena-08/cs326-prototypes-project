import numpy as np
import rasterio

def read_tif(file_path):
    """
    Read a .tif file and return it as a NumPy array.
    
    :param file_path: Path to the .tif file.
    :return: 2D NumPy array of the raster data.
    """
    with rasterio.open(file_path) as src:
        return src.read(1)  # Read the first band

def compute_confusion_matrix(gt, pred, num_classes):
    """
    Compute confusion matrix for semantic segmentation.
    
    :param gt: Ground truth array (2D).
    :param pred: Prediction array (2D).
    :param num_classes: Number of classes.
    :return: Confusion matrix of shape (num_classes, num_classes).
    """
    mask = (gt >= 1) & (gt <= num_classes)  # Ignore "no-data" pixels
    gt = gt[mask] - 1  # Shift to 0-based indexing
    pred = pred[mask] - 1  # Shift to 0-based indexing
    print("THIS IS THE pred", pred)
    

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((gt == i) & (pred == j))
            
    print("THIS IS THE CONFUSION MATRIX", confusion_matrix)
    return confusion_matrix

def compute_metrics(confusion_matrix):
    """
    Compute PA and MIoU from the confusion matrix.
    
    :param confusion_matrix: Confusion matrix of shape (num_classes, num_classes).
    :return: Pixel Accuracy (PA) and Mean IoU (MIoU).
    """
    # Pixel Accuracy
    total_correct = np.sum(np.diag(confusion_matrix))
    total_pixels = np.sum(confusion_matrix)
    pixel_accuracy = total_correct / total_pixels
    

    # Mean Intersection over Union
    IoUs = []
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        denominator = tp + fp + fn
        IoU = tp / denominator if denominator > 0 else 0
        IoUs.append(IoU)
    mean_iou = np.mean(IoUs)
    
    return pixel_accuracy, mean_iou

def main(gt_path, pred_path, num_classes=4):
    """
    Main function to compute PA and MIoU from ground truth and prediction rasters.
    
    :param gt_path: Path to the ground truth .tif file.
    :param pred_path: Path to the prediction .tif file.
    :param num_classes: Number of classes (default: 4).
    """
    # Load ground truth and prediction
    ground_truth = read_tif(gt_path)
    prediction = read_tif(pred_path)

    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(ground_truth, prediction, num_classes)

    # Compute metrics
    pa, miou = compute_metrics(confusion_matrix)

    # Display results
    print("Confusion Matrix:")
    print(confusion_matrix)
    print(f"Pixel Accuracy (PA): {pa:.4f}")
    print(f"Mean Intersection over Union (MIoU): {miou:.4f}")

if __name__ == "__main__":
    # Example file paths
    gt_path = "examples/labeled/dem_t6_23_labeled.tif"  # Replace with your ground truth file path
    pred_path = "dem_t6_44_pred.tif"  # Replace with your prediction file path

    # Compute PA and MIoU for 4 classes
    main(gt_path, pred_path, num_classes=4)


# ((("dem_t1_25@1" - 4398) / (5646 - 4398)) * 100 >= 75) * 3 + 
# ((("dem_t1_25@1" - 4398) / (5646- 4398)) * 100 >= 50 AND 
#  (("dem_t1_25@1" - 4398) / (5646 - 4398)) * 100 < 75) * 2 + 
# ((("dem_t1_25@1" - 4398) / (5646 - 4398)) * 100 >= 25 AND 
#  (("dem_t1_25@1" - 4398) / (5646 - 4398)) * 100 < 50) * 1 + 
# ((("dem_t1_25@1" - 4398) / (5646 - 4398)) * 100 < 25) * 0