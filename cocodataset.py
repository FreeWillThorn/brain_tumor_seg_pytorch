import json
import os

import numpy as np
import torch
from PIL import Image as PImage
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.tv_tensors import Mask, Image



class COCODataset(Dataset):
    def __init__(self, img_dir, mask_dir,edge_mask_dir,json_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.edge_mask_dir = edge_mask_dir
        self.json_dir = json_dir
        self.transform = transform

        # Load json file
        with open(json_dir) as f:
            coco = json.load(f)

        # Build map between file name and image_id
        self.filename_to_id = {img['file_name'] : img['id'] for img in coco['images']}
        self.valid_filenames = [fname for fname in os.listdir(self.img_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.valid_filenames)

    def __getitem__(self, idx):
        fname = self.valid_filenames[idx]
        #img_path = os.path.join(self.img_dir, fname)
        #mask_name = os.path.splitext(fname)[0] + '_mask.png'
        #mask_path = os.path.join(self.mask_dir, fname).replace('.jpg', '_mask.png')

        image = PImage.open(os.path.join(self.img_dir, fname))
        mask = PImage.open(os.path.join(self.mask_dir, fname).replace('.jpg', '_mask.png'))#.convert('L')  # Convert mask to grayscale
        #edge_mask = PImage.open(os.path.join(self.edge_mask_dir, fname).replace('.jpg', '_mask_edge.png'))
        image = Image(torch.tensor(np.array(image)).permute(2, 0, 1))  # Wrap with tv_tensors.Image
        mask = Mask(torch.tensor(np.array(mask), dtype=torch.long))  # Wrap with Mask
        #edge_mask = Mask(torch.tensor(np.array(edge_mask), dtype=torch.float32))
        if self.transform:
            sample = self.transform({"image": image, "mask": mask})
            image, mask = sample["image"], sample["mask"]
        return image, mask




