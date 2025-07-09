import torch
from tabulate import tabulate
from src.config import class_names
import torch.distributed as dist
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import cv2


def is_main_process():
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def print_validation_metrics(epoch_counter, total_train_loss, val_metrics):
    """
    Print validation metrics in a formatted table.
    """
    headers = ['Class', 'IoU', 'Accuracy']
    metrics_table = [[name, f"{iou:.4f}", f"{acc:.4f}"]
                    for name, iou, acc in zip(class_names,
                                            val_metrics['iou_per_class'],
                                            val_metrics['acc_per_class'])]
    if is_main_process():
        print(f"\nEpoch {epoch_counter}:")
        print(f"Total Train Loss: {total_train_loss:.4f}")
        print(f"Seg_Val_loss: {val_metrics['seg_val_loss']:.4f}")
        print(f"Edge_Val_loss: {val_metrics['edge_val_loss']:.4f}")
        print(f"Total Val Loss: {val_metrics['total_val_loss']:.4f}")
        print(f"Val mIoU: {val_metrics['mIoU']:.4f}")
        print(f"Val mAcc: {val_metrics['mAcc']:.4f}")
        print(f"Foreground mIoU: {val_metrics['Foreground_mIoU']:.4f}")
        print(f"Foreground Accuracy: {val_metrics['Foreground_Accuracy']:.4f}")
        print("\nPer-class metrics:")
        print(tabulate(metrics_table, headers=headers, tablefmt='grid'))

    return metrics_table



def print_test_metrics(test_metrics):
    """
    Print test metrics in a formatted table.
    """
    headers = ['Class', 'IoU', 'Accuracy']
    metrics_table = [[name, f"{iou:.4f}", f"{acc:.4f}"]
                    for name, iou, acc in zip(class_names,
                                            test_metrics['iou_per_class'],
                                            test_metrics['acc_per_class'])]
    if is_main_process():
        print("\nTest Results:")
        #print(f"Test Loss: {test_metrics['test_loss']:.4f}")
        print(f"Test mIoU: {test_metrics['mIoU']:.4f}")
        print(f"Test mAcc: {test_metrics['mAcc']:.4f}")
        print(f"Foreground mIoU: {test_metrics['Foreground_mIoU']:.4f}")
        print(f"Foreground mAcc: {test_metrics['Foreground_mAcc']:.4f}")
        print("\nPer-class metrics:")
        print(tabulate(metrics_table, headers=headers, tablefmt='grid'))

    return metrics_table


# Before saving, convert tensors to Python types
def tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    else:
        return obj


def overlay_segmentation(image, mask, class_idx=6, color=(255, 0, 128)):
    """
    Overlay the specified class mask on the original image.

    Args:
        image: Tensor [C, H, W] or ndarray [H, W, C]
        mask: Tensor or ndarray [H, W] with class indices
        class_idx: class index to highlight
        color: BGR tuple for overlay color

    Returns:
        overlay image as Tensor [3, H, W] with values in [0, 1]
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # [H, W, 3]

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # If grayscale, convert to RGB
    if image.shape[2] == 1 or len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = image.copy()
    binary_mask = (mask == class_idx).astype(np.uint8)

    # Colorize the predicted region
    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask == 1] = color

    # Blend original and mask
    overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)

    # Convert back to tensor [C, H, W], float32 [0, 1]
    overlay = np.transpose(overlay, (2, 0, 1))
    overlay = overlay.astype(np.float32) / 255.0
    return torch.tensor(overlay)