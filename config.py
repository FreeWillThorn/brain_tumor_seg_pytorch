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
from src.loss import HybridLoss, DiceLoss, EdgeFocalLoss, MultiClassFocalLoss
from src.model import ResnetBackbone, Model
from collections import Counter
import numpy as np
import torch.distributed as dist


def compute_class_weights(dataset, num_classes=8):
    pixel_counts = Counter()
    for _, mask in dataset:
        unique, counts = np.unique(mask.numpy(), return_counts=True)
        pixel_counts.update(dict(zip(unique, counts)))

    # Ensure all classes have a count (even if 0)
    weights = []
    total_pixels = sum(pixel_counts.values())

    for i in range(num_classes):
        class_count = pixel_counts.get(i, 0)
        # Compute inverse frequency with smoothing factor
        weight = total_pixels / (class_count + 1e-6)
        weights.append(weight)

    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes

    return weights

# Dataset Configuration
class DataConfig:
    BASE_DIR = '/root/final_dataset_edge'
    WORK_DIR = '/root/hy-nas/auto-weights-resnet50'
    SPLITS = {
        'train': 'train',
        'valid': 'valid',
        'test': 'test'
    }
    CLASS_NAMES = [
        'Background','Glioblastoma','Ganglioglioma','Meningioma','Neuroblastoma','Oligodendroglioma','Pituitary','Schwannoma'
    ]
    NUM_CLASSES = len(CLASS_NAMES)


# Training Configuration
class TrainConfig:
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    VAL_FREQUENCY = 10
    NUM_WORKERS = 4

    # Optimizer settings
    LEARNING_RATE = 3e-5
    BETAS = (0.9, 0.999)
    EPSILON = 1e-8
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING = 2

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
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], 'edge_masks'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], '_annotations.coco.json'),
    transform=TransformConfig.TRAIN_TRANSFORM
)

val_dataset = COCODataset(
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], 'image'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], 'mask'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], 'edge_masks'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], '_annotations.coco.json'),
    transform=TransformConfig.VAL_TRANSFORM
)

test_dataset = COCODataset(
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], 'image'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], 'mask'),
    os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], 'edge_masks'),
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

# Training components
#weights = compute_class_weights(train_dataset).to(device)
#weights = torch.tensor([0.1, 1, 1, 1, 1, 1, 1, 1]).to(device)
local_rank = int(os.environ['LOCAL_RANK'])
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


criterion = HybridLoss(ce_weight_tensor=class_weights)
#criterion = MultiClassFocalLoss(alpha = class_weights, gamma=2.0, reduction='mean')
#criterion_edge = nn.BCEWithLogitsLoss()
criterion_edge = EdgeFocalLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=TrainConfig.LEARNING_RATE,
    betas=TrainConfig.BETAS,
    eps=TrainConfig.EPSILON,
    weight_decay=TrainConfig.WEIGHT_DECAY
)
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=5)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=TrainConfig.SCHEDULER_STEP_SIZE, gamma=TrainConfig.SCHEDULER_GAMMA)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainConfig.NUM_EPOCHS, eta_min=0)


# Export variables
num_classes = DataConfig.NUM_CLASSES
num_epochs = TrainConfig.NUM_EPOCHS
val_frequency = TrainConfig.VAL_FREQUENCY
class_names = DataConfig.CLASS_NAMES
EARLY_STOPPING = TrainConfig.EARLY_STOPPING