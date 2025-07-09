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


def train_one_epoch(model,train_dataloader, optimizer, device, criterion,criterion_edge, epoch_counter=0):
    """
    Train the model for one epoch.
    :param model: The model to train.
    :param train_dataloader: The dataloader for training data.
    :param optimizer: The optimizer to use.
    :param device: The device to use (CPU or GPU).
    :param criterion: The loss_main function.
    :return: The average loss_main for the epoch.
    """
    model.train()
    total_train_loss = 0.0
    seg_train_loss = 0.0
    edge_train_loss = 0.0
    # If using distributed training, wrap the dataloader with DistributedSampler
    if is_main_process():
        data_iter = tqdm(train_dataloader, desc=f"Epoch {epoch_counter}")  # Show epoch and LR in progress bar
    else:
        data_iter = train_dataloader
    for images, masks,edge_masks, in data_iter:
        image = images.to(device)
        mask = masks.to(device)

        edge_mask = edge_masks.to(device)

        optimizer.zero_grad()
        seg_logit,edge_logit = model(image) # [B, C, H, W]

        loss_main = criterion(seg_logit, mask)
        loss_edge = criterion_edge(edge_logit, edge_mask)
        #print(f"loss main:{loss_main}, loss edge {loss_edge}")
        loss_total = loss_main + 0.4*loss_edge
        loss_total.backward()
        optimizer.step()

        total_train_loss += loss_total.item()
        seg_train_loss += loss_main.item()
        edge_train_loss += loss_edge.item()

    return total_train_loss / len(train_dataloader), seg_train_loss / len(train_dataloader), edge_train_loss / len(train_dataloader)


def evaluate(model, val_dataloader, num_classes, device, criterion, criterion_edge,writer=None,epoch_counter=None):

    model.eval()

    total_val_loss = 0.0
    seg_val_loss = 0.0
    edge_val_loss = 0.0
    metrics = Metrics(num_classes, device)
    data_iter = tqdm(val_dataloader) if is_main_process() else val_dataloader

    with torch.no_grad():
        for batch_idx, (image, mask, edge_mask) in enumerate(data_iter):
            image, mask, edge_mask = image.to(device), mask.to(device), edge_mask.to(device)
            #print("Edge mask stats:", edge_mask.min().item(), edge_mask.max().item(), edge_mask.mean().item())
            seg_logit,edge_logit = model(image) # [B, C, H, W]
            pred = torch.argmax(seg_logit, dim=1)  # [B, H, W]
            # visualize pituitary gland segmentation
            if batch_idx == 0 and writer is not None and is_main_process():
                sample_img = image[0]
                sample_gt = mask[0]
                sample_pred = pred[0]

                pit_overlay_gt = overlay_segmentation(sample_img, sample_gt, class_idx=6)
                pit_overlay_pred = overlay_segmentation(sample_img, sample_pred, class_idx=6)

                writer.add_image('Image/GT', pit_overlay_gt, global_step=epoch_counter)
                writer.add_image('Image/Pred', pit_overlay_pred, global_step=epoch_counter)
                sample_img_vis = sample_img.clone().detach().cpu()
                if sample_img_vis.max() > 1.0:
                    sample_img_vis = sample_img_vis / 255.0  # Normalize if needed

                writer.add_image('Image/Original', sample_img_vis, global_step=epoch_counter)
                writer.add_image('Image/GT Mask', sample_gt.unsqueeze(0).float() / 7.0, global_step=epoch_counter)
                writer.add_image('Image/Pred ', sample_pred.unsqueeze(0).float() / 7.0, global_step=epoch_counter)

            loss_main = criterion(seg_logit, mask) # [B, C, H, W] vs [B, H, W],return a single value
            loss_edge = criterion_edge(edge_logit, edge_mask)
            loss_total = loss_main + 0.4*loss_edge
            total_val_loss += loss_total.item() # sum up the loss
            seg_val_loss += loss_main.item()
            edge_val_loss += loss_edge.item()

            metrics.generate_matrix(pred, mask)


    total_val_loss = total_val_loss / len(val_dataloader)
    seg_val_loss = seg_val_loss / len(val_dataloader)
    edge_val_loss = edge_val_loss / len(val_dataloader)
    iou_per_class = metrics.iou_per_class()
    mIoU = metrics.mean_iou()
    acc_per_class = metrics.acc_per_class()
    mAcc= metrics.mean_accuracy()
    foreground_miou = metrics.mean_iou(ignore_inde_index=0)
    foreground_acc = metrics.mean_accuracy(ignore_inde_index=0)
    return {
        'total_val_loss': total_val_loss,
        'seg_val_loss': seg_val_loss,
        'edge_val_loss': edge_val_loss,
        'iou_per_class': iou_per_class,
        'mIoU': mIoU,
        'acc_per_class': acc_per_class,
        'mAcc': mAcc,
        'Foreground_mIoU': foreground_miou,
        'Foreground_Accuracy': foreground_acc
    }


def train_with_validation(model, train_loader, val_loader, device, num_classes = config.num_classes,
                          optimizer = config.optimizer, scheduler = config.scheduler, criterion=config.criterion,criterion_edge=config.criterion_edge, num_epochs=config.num_epochs,
                            val_frequency=config.val_frequency, early_stopping_patience=config.EARLY_STOPPING, log_dir=config.DataConfig.WORK_DIR):


    # Create a unique run name with timestamp
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, run_name))

    best_val_miou = 0
    patience_counter = 0
    training_history = []

    for epoch in range(num_epochs):
        epoch_counter = epoch + 1
        # Training
        #if is_main_process():
        #    tqdm.write(f"Start training on epoch {epoch_counter}")
        #if is_main_process():
        #    tqdm.write(f"Lr: {optimizer.param_groups[0]['lr']:.6f}, Epoch: {epoch_counter}/{num_epochs}")
        total_train_loss,seg_train_loss,edge_train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion,criterion_edge,epoch_counter)
        writer.add_scalar('Loss/Total Loss', total_train_loss, epoch_counter)
        writer.add_scalar('Loss/Seg Loss', seg_train_loss, epoch_counter)
        writer.add_scalar('Loss/Edge Loss', edge_train_loss, epoch_counter)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch_counter)
        scheduler.step()
        # Validation only every val_frequency epochs
        if epoch_counter!= 0 and epoch_counter % val_frequency == 0:
            val_metrics = evaluate(model, val_loader, num_classes, device, criterion, criterion_edge,writer=writer, epoch_counter=epoch_counter)

            # Log to TensorBoard
            writer.add_scalar('val/total_loss', val_metrics['total_val_loss'], epoch_counter)
            writer.add_scalar('val/seg_loss', val_metrics['seg_val_loss'], epoch_counter)
            writer.add_scalar('val/edge_loss', val_metrics['edge_val_loss'], epoch_counter)
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
                'seg_loss': seg_train_loss,
                'edge_loss': edge_train_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'total_val_loss': val_metrics['total_val_loss'],
                'seg_val_loss': val_metrics['seg_val_loss'],
                'edge_val_loss': val_metrics['edge_val_loss'],
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
            # Track only training metrics when not validating
            metrics = {
                'epoch': epoch_counter,
                'train_loss': total_train_loss,
                'seg_loss': seg_train_loss,
                'edge_loss': edge_train_loss,
            }

            training_history.append(metrics)
            if is_main_process():
                tqdm.write(f"Epoch {epoch_counter:03d} | Train Loss: {total_train_loss:.4f} | Seg Loss: {seg_train_loss:.4f} | Edge Loss: {edge_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    writer.close()
    #history_path = os.path.join(log_dir, run_name, 'train_history.json')
    #with open(history_path, 'w') as f:
    #    json.dump(training_history, f, indent=4)



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
