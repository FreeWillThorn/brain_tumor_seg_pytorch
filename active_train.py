import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from alipy.index import IndexCollection

from alipy.query_strategy import QueryInstanceUncertainty
from mmengine import Config
from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model
import torch
import os
from tqdm import tqdm
import numpy as np
from alipy import ToolBox
import subprocess

# ----------- Functions --------


def write_filenames(labeled_filenames, pool_filenames, method, round):
    labeled_path = f'/root/hy-nas/AL_workdir/{method}/{round + 2}0%_labeled_filenames.txt'
    pool_path = f'/root/hy-nas/AL_workdir/{method}/{round + 2}0%_pool_filenames.txt'

    with open(labeled_path, 'w') as f:
        for fname in labeled_filenames:
            f.write(f"{fname}\n")

    with open(pool_path, 'w') as f:
        for fname in pool_filenames:
            f.write(f"{fname}\n")
    return labeled_path, pool_path


def initial_split(json_path, method, round):
    # 1. Load COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 2. Build image_id to label & image_id to filename mappings
    image_id_to_label = {}
    image_id_to_filename = {}

    for img in coco_data['images']:
        image_id_to_filename[img['id']] = img['file_name']

    for ann in coco_data['annotations']:
        image_id_to_label[ann['image_id']] = ann['category_id']

    # 3. Prepare image_ids & labels
    image_ids = list(image_id_to_label.keys())
    labels = list(image_id_to_label.values())

    # 4. Stratified Split: 10% labeled, 90% unlabeled
    train_ids, pool_ids = train_test_split(
        image_ids, train_size=0.10, stratify=labels, random_state=42
    )

    print(f"####### Initial labeled (10%) set: {len(train_ids)}, Unlabeled pool: {len(pool_ids)} #######")

    # 5. Create output directory
    os.makedirs(f'/root/hy-nas/AL_workdir/{method}/train_{round + 2}0%', exist_ok=True)

    # 6. Save ID lists
    np.savetxt(f'/root/hy-nas/AL_workdir/{method}/train_ids.txt', train_ids, fmt='%d')
    np.savetxt(f'/root/hy-nas/AL_workdir/{method}/pool_ids.txt', pool_ids, fmt='%d')

    # 7. Map IDs to filenames
    labeled_filenames = []
    pool_filenames = []

    for img_id in train_ids:
        labeled_filenames.append(image_id_to_filename[img_id])

    for img_id in pool_ids:
        pool_filenames.append(image_id_to_filename[img_id])

    labeled_path, pool_path = write_filenames(labeled_filenames, pool_filenames, method, -1)

    print("####### Saved both ID lists and filename lists successfully! #######")

    return IndexCollection(labeled_filenames), IndexCollection(pool_filenames), labeled_path, pool_path


def pred_model(cfg, pool_filename_path,round):
    with open(pool_filename_path, 'r') as f:
        pool_filenames = [line.strip() for line in f.readlines()]
    print(f"####### Loaded {len(pool_filenames)} pool images.#######")
    checkpoint_path = f'/root/hy-nas/AL_workdir/entropy/train_{round + 1}0%/iter_15000.pth'
    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    all_probs = []

    for img_file in tqdm(pool_filenames):
        img_path = os.path.join('/root/mmsegmentation/data/final_dataset/train/image/', img_file)

        with torch.no_grad():
            result = inference_model(model, img_path)

        logits = result.seg_logits.data.unsqueeze(0)  # [1, num_classes, H, W]
        probs = torch.softmax(logits, dim=1)  # [1, C, H, W]

        avg_probs = torch.mean(probs, dim=[2, 3]).squeeze(0)  # [C]
        all_probs.append(avg_probs.cpu().numpy())

    return np.array(all_probs)  # [N_images, num_classes]

'''
### --- Knowledge Distillation
def training_process(cfg, labeled_path, method, round):
    train_work_path = f'/root/hy-nas/AL_workdir/{method}/train_{round + 2}0%'
    cfg = cfg
    cfg.load_from = f'/root/hy-nas/AL_workdir/entropy/train_{round + 1}0%/iter_15000.pth'
    cfg.train_dataloader.dataset.img_names_file = labeled_path
    cfg.work_dir = train_work_path
    runner = Runner.from_cfg(cfg)
    runner.train()
'''

def training_process(cfg_path, labeled_path,method,round):
    train_work_path = f'/root/hy-nas/AL_workdir/{method}/train_{round + 2}0%'
    cmd = [
        'bash',
        '/root/mmsegmentation/tools/dist_train.sh',
        cfg_path,
        '2',
        '--work-dir',
        train_work_path,
        '--cfg-options',
        f'train_dataloader.dataset.img_names_file={labeled_path}'
    ]
    subprocess.run(cmd, check=True)


def test_process(cfg_path, method, round):
    test_work_path = f'/root/hy-nas/AL_workdir/{method}/test_{round + 2}0%'
    checkpoint_path = f'/root/hy-nas/AL_workdir/{method}/train_{round + 2}0%/iter_15000.pth'
    cmd = [
        'python',
        '/root/mmsegmentation/tools/test.py',
        cfg_path,
        checkpoint_path,
        '--work-dir',
        test_work_path,
    ]
    subprocess.run(cmd, check=True)


# ----initial settings----
json_path = '/root/mmsegmentation/data/final_dataset/train/train.json'
cfg_path = '/root/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py'
# ----initial settings----


N_QUERY = 412  # Number of samples to query each round
N_ROUNDS = 8
method = 'entropy'

# labeled_filenames, pool_filenames, labeled_filename_path, pool_filename_path = initial_split(json_path, method, -1)
# training_process(cfg, labeled_filename_path, method, -1)
# test_process(cfg, checkpoint_path,method, -1)

# ---- temporary setting to avoid redo the initial part -----
pool_filename_path = '/root/hy-nas/AL_workdir/entropy/10%_pool_filenames.txt'
labeled_filename_path = '/root/hy-nas/AL_workdir/entropy/10%_labeled_filenames.txt'

# Load labeled filenames
with open(labeled_filename_path, 'r') as f:
    labeled_list = [line.strip() for line in f.readlines()]

# Load pool filenames
with open(pool_filename_path, 'r') as f:
    pool_list = [line.strip() for line in f.readlines()]

labeled_filenames = IndexCollection(labeled_list)
pool_filenames = IndexCollection(pool_list)

# ---- temporary setting to avoid redo the initial part -----

for round in range(N_ROUNDS):
    print(f"\n ####### {round + 2}0% labeled training #######")

    probs_array = pred_model(cfg_path, pool_filename_path,round)

    with open(pool_filename_path, 'r') as f:
        pool_filenames_temp = [line.strip() for line in f.readlines()]
    unlabeled_indices = np.arange(len(pool_filenames_temp))

    query_method = QueryInstanceUncertainty(measure='entropy')
    selected_list = query_method.select_by_prediction_mat(unlabeled_indices, predict=probs_array, batch_size=N_QUERY)
    select_filenames = [pool_filenames[i] for i in selected_list]

    labeled_filenames.update(select_filenames)
    pool_filenames.difference_update(select_filenames)

    labeled_filename_path, pool_filename_path = write_filenames(labeled_filenames, pool_filenames, method, round)

    training_process(cfg_path, labeled_filename_path, method, round)
    test_process(cfg_path, method, round)