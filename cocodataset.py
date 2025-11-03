import json
import os

import numpy as np
import torch
from PIL import Image as PImage
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.tv_tensors import Mask, Image



class COCODataset(Dataset):
    def __init__(self, img_dir, mask_dir, json_dir, transform=None, img_ids_list=None, img_names_list=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.json_dir = json_dir
        self.transform = transform

        # Load json file
        with open(json_dir) as f:
            coco = json.load(f)

        # Build map between file name and image_id
        self.id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        self.filename_to_id = {img['file_name'] : img['id'] for img in coco['images']}

        # 2 maps
        all_files = [fname for fname in os.listdir(self.img_dir) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_ids = [self.filename_to_id[f] for f in all_files if f in self.filename_to_id]

        if img_ids_list is not None:
            allow_ids = set(img_ids_list)
            self.valid_ids = [img_id for img_id in all_ids if img_id in allow_ids]
        elif img_names_list is not None:
            allow_names = set(img_names_list)
            self.valid_ids = [self.filename_to_id[f] for f in all_files if f in allow_names]
        else:
            self.valid_ids = all_ids


    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        fname  = self.id_to_filename[img_id]
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




