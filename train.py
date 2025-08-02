import json

import torch
from tqdm import tqdm

from src import config
from src.metrics import Metrics
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from src.utils import *
from src.config import *
import torch.distributed as dist


def train_one_epoch(model,train_dataloader, optimizer, device, criterion,epoch_counter=0):

    model.train()
    total_train_loss = 0.0
    #total_edge_loss = 0.0
    # If using distributed training, wrap the dataloader with DistributedSampler
    if is_main_process():
        data_iter = tqdm(train_dataloader, desc=f"Epoch {epoch_counter}")  # Show epoch and LR in progress bar
    else:
        data_iter = train_dataloader
    for images, masks in data_iter:
        image = images.to(device)
        mask = masks.to(device)

        optimizer.zero_grad()
        seg_logit,log_var= model(image) # [B, C, H, W]

        loss_main = criterion(seg_logit, mask,log_var)
        loss_total = loss_main
        loss_total.backward()
        optimizer.step()

        total_train_loss += loss_total.item()
        #total_edge_loss += edge_loss.item()

    return total_train_loss / len(train_dataloader)

def evaluate(model, val_dataloader, num_classes, device, base_criterion,writer=None,epoch_counter=None):

    model.eval()

    total_val_loss = 0.0
    metrics = Metrics(num_classes, device)
    data_iter = tqdm(val_dataloader) if is_main_process() else val_dataloader

    with torch.no_grad():
        for batch_idx, (image, mask) in enumerate(data_iter):
            image, mask = image.to(device), mask.to(device)
            seg_logit,log_var= model(image) # [B, C, H, W]
            pred = torch.argmax(seg_logit, dim=1)  # [B, H, W]
            loss_main = base_criterion(seg_logit, mask)
            loss_total = loss_main
            total_val_loss += loss_total.item()  # sum up the loss
            metrics.generate_matrix(pred, mask)
            # visualize pituitary gland segmentation
        if batch_idx == 0 and writer is not None and is_main_process():
            sample_img = image[0]  # [C, H, W]
            sample_gt = mask[0]  # [H, W]
            sample_pred = pred[0]  # [H, W]

            # Create colored masks
            num_classes = seg_logit.size(1)
            colors = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 1, 0], [1, 0, 1], [0, 1, 1]], device=device)  # Add more colors if needed

            # Convert prediction to colored mask
            colored_pred = colors[sample_pred]  # [H, W, 3]
            colored_pred = colored_pred.permute(2, 0, 1)  # [3, H, W]

            # Create overlay by blending original image with colored prediction
            alpha = 0.5  # Adjust transparency
            overlay = alpha * sample_img + (1 - alpha) * colored_pred

            # Add images to TensorBoard
            writer.add_image('Validation/Original', sample_img, epoch_counter)
            writer.add_image('Validation/GT', sample_gt.unsqueeze(0).float(), epoch_counter)
            writer.add_image('Validation/Colored_Pred', colored_pred, epoch_counter)
            writer.add_image('Validation/Overlay', overlay, epoch_counter)


    total_val_loss = total_val_loss / len(val_dataloader)
    iou_per_class = metrics.iou_per_class()
    mIoU = metrics.mean_iou()
    acc_per_class = metrics.acc_per_class()
    mAcc= metrics.mean_accuracy()
    foreground_miou = metrics.mean_iou(ignore_inde_index=0)
    foreground_acc = metrics.mean_accuracy(ignore_inde_index=0)
    return {
        'total_val_loss': total_val_loss,
        'iou_per_class': iou_per_class,
        'mIoU': mIoU,
        'acc_per_class': acc_per_class,
        'mAcc': mAcc,
        'Foreground_mIoU': foreground_miou,
        'Foreground_Accuracy': foreground_acc
    }


def train_with_validation(model, train_loader, val_loader, device, num_classes = config.num_classes,
                          optimizer = config.optimizer, scheduler = config.scheduler, criterion=config.criterion, num_epochs=config.num_epochs,
                            val_frequency=config.val_frequency, early_stopping_patience=config.EARLY_STOPPING, log_dir=config.DataConfig.WORK_DIR):


    # Create a unique run name with timestamp
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, run_name))

    best_val_miou = 0
    patience_counter = 0
    training_history = []
    base_criterion = config.base_criterion

    for epoch in range(num_epochs):
        epoch_counter = epoch + 1
        total_train_loss= train_one_epoch(model, train_loader, optimizer, device, criterion,epoch_counter)
        writer.add_scalar('Loss/Total Loss', total_train_loss, epoch_counter)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch_counter)

        scheduler.step()
        if epoch_counter!= 0 and epoch_counter % val_frequency == 0:
            val_metrics = evaluate(model, val_loader, num_classes, device, base_criterion,writer=writer, epoch_counter=epoch_counter)

            # Log to TensorBoard
            writer.add_scalar('val/total_loss', val_metrics['total_val_loss'], epoch_counter)
            writer.add_scalar('Pituitary/IoU', val_metrics['iou_per_class'][6], epoch_counter)
            writer.add_scalar('Pituitary/Accuracy', val_metrics['acc_per_class'][6], epoch_counter)
            writer.add_scalar('Metrics/foreground_mIoU', val_metrics['Foreground_mIoU'], epoch_counter)
            writer.add_scalar('Metrics/foreground_Accuracy', val_metrics['Foreground_Accuracy'], epoch_counter)
            writer.add_scalar('Metrics/mIoU', val_metrics['mIoU'], epoch_counter)
            writer.add_scalar('Metrics/mAcc', val_metrics['mAcc'], epoch_counter)


            # Track metrics
            metrics = {
                'epoch': epoch_counter,
                'total_train_loss': total_train_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'total_val_loss': val_metrics['total_val_loss'],
                'val_iou_per_class': val_metrics['iou_per_class'],
                'val_acc_per_class': val_metrics['acc_per_class'],
                'val_mIou': val_metrics['mIoU'],
                'val_mAcc': val_metrics['mAcc'],
                'Foreground_mIoU': val_metrics['Foreground_mIoU'],
                'Foreground_Accuracy': val_metrics['Foreground_Accuracy']
            }
            training_history.append(metrics)
            print_validation_metrics(epoch_counter, total_train_loss, val_metrics)

            # Early stopping check
            if val_metrics['mIoU'] > best_val_miou:
                best_val_miou = val_metrics['mIoU']
                patience_counter = 0
                # Save best model
                save_path = os.path.join(log_dir, run_name, f'best_mIou_epoch_{epoch_counter}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                tqdm.write(f"Early stopping triggered after {epoch_counter} epochs")
                break
        else:
            metrics = {
                'epoch': epoch_counter,
                'train_loss': total_train_loss,
            }

            training_history.append(metrics)
            if is_main_process():
                tqdm.write(f"Epoch {epoch_counter:03d} | Train Loss: {total_train_loss:.4f}  | LR: {optimizer.param_groups[0]['lr']:.6f}")


    writer.close()




def main():
    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        #print(local_rank)
        config.device = torch.device('cuda', local_rank)
        model = config.model.to(config.device)
        #weights = config.weights.to(device)
        #weighted_criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

        # Create DistributedSampler and DataLoader after process group is initialized
        train_sampler = DistributedSampler(config.train_dataset)
        train_loader = DataLoader(
            config.train_dataset,
            batch_size=config.TrainConfig.BATCH_SIZE,
            sampler=train_sampler,
            shuffle=False,
            num_workers=config.TrainConfig.NUM_WORKERS
        )
    else:
        config.device = torch.device('cpu')
        ddp_model = config.model
        train_loader = DataLoader(
            config.train_dataset,
            batch_size=config.TrainConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=config.TrainConfig.NUM_WORKERS
        )

    val_loader = DataLoader(
        config.val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.TrainConfig.NUM_WORKERS
    )

    

    train_with_validation(ddp_model, train_loader, val_loader,config.device)
    # Save training history to a JSON file


if __name__ == '__main__':
    main()
