import torch

class Metrics(object):
    """
        Calculate pixel accuracy from confusion matrix.
        TP = true positive = true sample predicted as true
        FN = false negative = true sample predicted as false
        FP = false positive = false sample predicted as true
        TN = true negative = false sample predicted as false
        IOU = TP / (TP + FP + FN)
        """

    def __init__(self,num_classes,device,ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.conf_matrix = torch.zeros(num_classes, num_classes, device=device)

    def generate_matrix(self, pred, mask):
        conf_matrix = torch.bincount(self.num_classes * mask.flatten().long() + pred.flatten().long(),
                                     minlength=self.num_classes **2).reshape(self.num_classes, self.num_classes)
        self.conf_matrix += conf_matrix
        return self.conf_matrix

    def reset(self):
        """Reset the confusion matrix to zeros."""
        self.conf_matrix.zero_()

    def pixel_accuracy(self):
        # acc = TP / all pixels
        pixel_accuracy = self.conf_matrix.diag().sum() / self.conf_matrix.sum() # TP / all pixels, value / value
        return pixel_accuracy

    def acc_per_class(self):
        # TP / (TP + FN) = true prediction / all predictions ->for each class, 1D vector / 1D vector
        acc_per_class = self.conf_matrix.diag() / self.conf_matrix.sum(1)
        return acc_per_class # 1D vector

    def iou_per_class(self):
        """
        Calculate Intersection over Union (IoU) for each class.
        IoU = TP / (TP + FP + FN) = intersection / union
        """
        inter = self.conf_matrix.diag()
        union = self.conf_matrix.sum(1) + self.conf_matrix.sum(0) - inter  # union = total pred + total true - intersection
        # Handle division by zero
        iou_class = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        return iou_class # 1D vector

    def mean_iou(self,ignore_inde_index=None):
        """
        Calculate mean Intersection over Union (mIoU).
        mIoU = mean(IoU for each class)
        """
        iou_class = self.iou_per_class()
        if self.ignore_index is not None:
            iou_class = iou_class[self.ignore_index != torch.arange(self.num_classes)]
        elif ignore_inde_index is not None:
            iou_class = iou_class[ignore_inde_index != torch.arange(self.num_classes)]
        else:
            iou_class = iou_class
        
        mIoU = iou_class.mean()
        return mIoU

    def mean_accuracy(self,ignore_inde_index=None):
        """
        Calculate mean accuracy across all classes.
        Mean Accuracy = mean(TP / (TP + FN))
        """
        acc_per_class = self.acc_per_class()
        if self.ignore_index is not None:
            acc_per_class = acc_per_class[self.ignore_index != torch.arange(self.num_classes)]
        elif ignore_inde_index is not None:
            acc_per_class = acc_per_class[ignore_inde_index != torch.arange(self.num_classes)]
        else:
            acc_per_class = acc_per_class
        mAcc = acc_per_class.mean()
        return mAcc