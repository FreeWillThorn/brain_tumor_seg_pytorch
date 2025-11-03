import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from torchvision.transforms import *
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

from src.cocodataset import COCODataset
from src.loss import HybridLoss, HybridLoss_image_wise_uncertainty
#from src.model_all import ResnetBackbone, Model, Resnet_model_backbone_mse, Swin_backbone
from collections import Counter
import numpy as np
import torch.distributed as dist

from src.model import Model


# Dataset Configuration
class DataConfig:
    BASE_DIR = '/root/final_dataset_edge'
    WORK_DIR = '/root/hy-nas/auto-weights-resnet50'
    TEST_DIR = '/root/hy-data/auto-weights-resnet50-test'
    AL_DIR = '/root/hy-nas/AL_workdir'
    SPLITS = {
        'train': 'train',
        'valid': 'valid',
        'test': 'test'
    }
    CLASS_NAMES = [
        'Background','Glioblastoma','Ganglioglioma','Meningioma','Neuroblastoma','Oligodendroglioma','Pituitary','Schwannoma'
    ]
    NUM_CLASSES = len(CLASS_NAMES)

class ALConfig:
    METHOD = 'entropy'
    AL_DIR = f'/root/hy-nas/AL_workdir/{METHOD}'
    N_QUERY = 412  # samples per round to add
    N_ROUNDS = 8  # number of AL rounds

# Training Configuration
class TrainConfig:
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    VAL_FREQUENCY = 10
    NUM_WORKERS = 8

    # Optimizer settings
    LEARNING_RATE = 1e-4
    BETAS = (0.9, 0.999)
    EPSILON = 1e-8
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING = 3
    WARM_UP_EPOCHS = 10

    # Scheduler settings
    #SCHEDULER_STEP_SIZE = 20
    #SCHEDULER_GAMMA = 0.5



# Transform Configuration
class TransformConfig:
    TRAIN_TRANSFORM = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=0
        ),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])

    VAL_TRANSFORM = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    TEST_TRANSFORM = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

# Initialize datasets
train_dataset = COCODataset(
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], 'image'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], 'mask'),
    #os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], 'edge_masks'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], '_annotations.coco.json'),
    transform=TransformConfig.TRAIN_TRANSFORM
)

val_dataset = COCODataset(
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], 'image'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], 'mask'),
    #os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], 'edge_masks'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], '_annotations.coco.json'),
    transform=TransformConfig.VAL_TRANSFORM
)

test_dataset = COCODataset(
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], 'image'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], 'mask'),
    #os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], 'edge_masks'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], '_annotations.coco.json'),
    transform=TransformConfig.TEST_TRANSFORM
)

# Initialize data loaders
'''If using distributed training, use DistributedSampler to ensure each process gets a unique subset of the data.
train_sampler = DistributedSampler(train_dataset)
#train_sampler = None
train_loader = DataLoader(train_dataset,
                          batch_size=TrainConfig.BATCH_SIZE,
                          sampler=train_sampler,
                          shuffle=False,
                          num_workers=TrainConfig.NUM_WORKERS
                          )

val_loader = DataLoader(val_dataset,
                        batch_size=TrainConfig.BATCH_SIZE,
                        shuffle=False,
                        num_workers=TrainConfig.NUM_WORKERS
)
'''

# Model initialization
model = Model(num_classes=DataConfig.NUM_CLASSES)
#model = Swin_backbone(num_classes=DataConfig.NUM_CLASSES)

#weights = torch.tensor([0.1, 1, 1, 1, 1, 1, 1, 1]).to(device)
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
else:
    local_rank = 0
device = torch.device('cuda', local_rank)

class_weights = torch.tensor([0.1,
                              11.5696075,
                              20.9788815,
                              31.1377409,
                              40.7049918,
                              50.7300624,
                              61.9656335,
                              70.9126968
                              ], dtype=torch.float32).to(device)


criterion = HybridLoss_image_wise_uncertainty(ce_weight_tensor=class_weights)
#criterion =HybridLoss(ce_weight_tensor=class_weights)
base_criterion = HybridLoss(ce_weight_tensor=class_weights)
#edge_criterion = combo_loss()


optimizer = optim.AdamW(
    model.parameters(),
    lr=TrainConfig.LEARNING_RATE,
    betas=TrainConfig.BETAS,
    eps=TrainConfig.EPSILON,
    weight_decay=TrainConfig.WEIGHT_DECAY
)
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=TrainConfig.WARM_UP_EPOCHS)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainConfig.NUM_EPOCHS-TrainConfig.WARM_UP_EPOCHS, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[TrainConfig.WARM_UP_EPOCHS])

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=TrainConfig.SCHEDULER_STEP_SIZE, gamma=TrainConfig.SCHEDULER_GAMMA)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainConfig.NUM_EPOCHS, eta_min=0)


# Export variables
num_classes = DataConfig.NUM_CLASSES
num_epochs = TrainConfig.NUM_EPOCHS
val_frequency = TrainConfig.VAL_FREQUENCY
class_names = DataConfig.CLASS_NAMES
EARLY_STOPPING = TrainConfig.EARLY_STOPPING