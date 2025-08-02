import torch
from tabulate import tabulate
from src.config import class_names
import torch.distributed as dist
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

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


def visualize_prediction(image, mask, pred, save_path):
    """
    Visualize the prediction overlapped on the input image.
    Args:
        image: tensor of shape [3, H, W]
        mask: tensor of shape [H, W]
        pred: tensor of shape [H, W]
        save_path: path to save the visualization
    """


    # Convert tensors to numpy arrays
    image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()

    # Normalize image for visualization
    image = (image - image.min()) / (image.max() - image.min())

    # Create color maps for mask and prediction
    colors = [
        [0, 0, 0],  # Background (black)
        [1, 0.2, 0.2],  # Glioblastoma (bright red)
        [0.4, 0.8, 0],  # Ganglioglioma (lime green)
        [0, 0.6, 1],  # Meningioma (sky blue)
        [1, 0.5, 0],  # Neuroblastoma (orange)
        [0.8, 0, 0.8],  # Oligodendroglioma (purple)
        [1, 0.8, 0],  # Pituitary (gold)
        [0, 0.8, 0.8],  # Schwannoma (teal)
    ]

    class_names = [
        'Background', 'Glioblastoma', 'Ganglioglioma', 'Meningioma', 'Neuroblastoma','Oligodendroglioma', 'Pituitary', 'Schwannoma'
    ]

    # Create RGB masks
    mask_rgb = np.zeros((*mask.shape, 3))
    pred_rgb = np.zeros((*pred.shape, 3))

    for i, color in enumerate(colors):
        mask_rgb[mask == i] = color
        pred_rgb[pred == i] = color
        #pred_name = class_names[pred[i]]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image')
    axes[0].axis('off')

    # Plot ground truth mask
    axes[1].imshow(mask_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Plot prediction
    axes[2].imshow(pred_rgb)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Add text labels
    unique_classes = np.unique(pred)
    for cls_id in unique_classes:
        y, x = np.where(pred == cls_id)
        if len(x) > 0 and len(y) > 0:
            axes[2].text(
                int(np.mean(x)), int(np.mean(y)),
                class_names[cls_id],
                color='white', fontsize=6, ha='left', va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
            )

    # Plot overlay
    pred_overlay = image.copy()
    pred_overlay = pred_overlay * 0.7 + pred_rgb * 0.3
    axes[3].imshow(pred_overlay)
    axes[3].set_title('pred_overlay')
    axes[3].axis('off')

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()