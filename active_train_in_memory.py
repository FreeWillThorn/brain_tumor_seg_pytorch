# --- Python 3.10+ compatibility for libraries that still use collections.Iterable ---
import collections, collections.abc
from src import config
for name in ("Iterable", "Mapping", "MutableMapping", "Sequence"):
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))
# -------------------------------------------------------------------------------
from alipy.query_strategy import QueryInstanceUncertainty
from tqdm import tqdm
import src.config
from src.cocodataset import *
from src.config import*
from src.model import Model
from sklearn.model_selection import train_test_split
from src.train import train_with_validation
from src.test import test as test_fn
import json,os,time
import numpy as np
from torch.utils.data import DataLoader
import subprocess, glob
import gc

# ---------- config ----------
METHOD = config.ALConfig.METHOD  # 'entropy', 'least_confidence', 'margin_sampling'
N_QUERY = config.ALConfig.N_QUERY  # samples per round to add
N_ROUNDS = config.ALConfig.N_ROUNDS  # number of AL rounds
WORK_BASE = config.ALConfig.AL_DIR  # base dir to save AL rounds
os.makedirs(WORK_BASE, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_IMG_DIR = os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], 'image')
TRAIN_MASK_DIR = os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], 'mask')
TRAIN_JSON     = os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['train'], '_annotations.coco.json')

VAL_IMG_DIR = os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], 'image')
VAL_MASK_DIR = os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], 'mask')
VAL_JSON     = os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['valid'], '_annotations.coco.json')

TEST_IMG_DIR = os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], 'image')
TEST_MASK_DIR= os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'], 'mask')
TEST_JSON    = os.path.join(DataConfig.BASE_DIR, DataConfig.SPLITS['test'],  '_annotations.coco.json')

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def build_dataset(img_dir, mask_dir, json_path, transform, img_ids_list=None):
    return COCODataset(img_dir, mask_dir, json_path,
                       transform=transform, img_ids_list=img_ids_list)

def build_loader(dataset, batch_size, shuffle, workers=TrainConfig.NUM_WORKERS):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)

def save_initial_split(save_dir, labeled_ids, pool_ids, seed=42, ratio=0.10, extra_meta=None):
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")  # timestamp for uniqueness
    payload = {
        "meta": {
            "seed": seed,
            "ratio": ratio,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(extra_meta or {})
        },
        "labeled_ids": list(labeled_ids),
        "pool_ids": list(pool_ids)
    }
    path = os.path.join(save_dir, f"initial_split_seed{seed}_{int(ratio*100)}_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[split] saved initial split to {path}")
    return path

def load_initial_split(path):
    with open(path) as f:
        payload = json.load(f)
    labeled_ids = payload["labeled_ids"]
    pool_ids = payload["pool_ids"]
    meta = payload.get("meta", {})
    print(f"[split] loaded {path} | seed={meta.get('seed')} ratio={meta.get('ratio')}")
    return labeled_ids, pool_ids, meta

def initial_split(train_json, train_img_dir,seed = 42,initial_ratio=0.1):

    with open(train_json) as f:
        coco = json.load(f)

    id2file = {img['id']: img['file_name'] for img in coco['images']}
    id2cat= {}

    for ann in coco['annotations']:
        if ann['image_id'] not in id2cat: # `ann['image_id']` it's already a value
            id2cat[ann['image_id']] = ann['category_id']

    # Some images were deleted from the directory due to annotation issues,so here we filter out the files that do not exist
    exist = set(os.listdir(train_img_dir))
    image_ids_exist = [i for i in id2cat.keys() if id2file[i] in exist]
    cat_ids_exist = [id2cat[i] for i in image_ids_exist]

    labeled_ids, pool_ids = train_test_split(image_ids_exist, train_size=initial_ratio, stratify=cat_ids_exist, random_state=seed)

    _ = save_initial_split(
        save_dir=os.path.join(WORK_BASE, "splits"),
        labeled_ids=labeled_ids,
        pool_ids=pool_ids,
        seed=42,
        ratio=0.10,
        extra_meta={"method": "stratified_random"}
    )

    return labeled_ids, pool_ids

def predict_pool_probs(ckpt_path,pool_ids,batch_size = 4):

    model = Model(num_classes=DataConfig.NUM_CLASSES)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    # Remove 'module.' prefix if present, duno why there are prefixes.
    state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # pool is all unlabeled data (untrained), so the img_ids_list here is unused imgs
    pool_dataset = build_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_JSON,transform=TransformConfig.VAL_TRANSFORM,img_ids_list=pool_ids)
    pool_loader = build_loader(pool_dataset, batch_size=batch_size, shuffle=False, workers=TrainConfig.NUM_WORKERS)
    print(f"####### Loaded {len(pool_dataset)} pool images.#######")

    all_probs = []
    #Input: images → [4, 3, 512, 512]  # 4 images, 3 channels, 512x512 pixels
    #Model output: seg_logits → [4, 10, 512, 512]  # 4 images, 10 classes, 512x512
    #After softmax: probs → [4, 10, 512, 512]  # Now values are probabilities (0-1)
    #After mean: avg → [4, 10]  # Average probability for each class in each image
    with torch.no_grad():
        for images, _ in tqdm(pool_loader, desc='Scoreing pool samples'):
            images = images.to(DEVICE)
            seg_logits,_ = model(images) # [B, C, H, W]
            probs = torch.softmax(seg_logits, dim=1)  # [B, C, H, W]
            avg = probs.mean(dim=[2, 3])  # [B, C]
            all_probs.append(avg.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)  # [N, C]

    ###### empty gpu memory ffffffk ######
    del model, images, seg_logits # probs is now a NumPy copy; safe to delete tensor refs

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ###### empty gpu memory ######
    return probs

def run_train_round_cpu(labeled_ids,round_idx):
    work_dir=ensure_dir(os.path.join(WORK_BASE, f'{METHOD}_train_{(round_idx+2)*10}%'))

    model = Model(num_classes=DataConfig.NUM_CLASSES).to(DEVICE)
    #cant use optimizer_new = src.config.optimizer, cause each time it will be a reference of the first round's optimizer
    optimizer_new = torch.optim.AdamW(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        betas=TrainConfig.BETAS,
        eps=TrainConfig.EPSILON,
        weight_decay=TrainConfig.WEIGHT_DECAY
    )
    warmup_new = torch.optim.lr_scheduler.LinearLR(optimizer_new, start_factor=0.2, total_iters=TrainConfig.WARM_UP_EPOCHS)
    cosine_new = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_new,
                                                        T_max=TrainConfig.NUM_EPOCHS - TrainConfig.WARM_UP_EPOCHS,
                                                        eta_min=1e-6)
    scheduler_new = torch.optim.lr_scheduler.SequentialLR(optimizer_new, schedulers=[warmup_new, cosine_new],
                                                      milestones=[TrainConfig.WARM_UP_EPOCHS])

    # train_ds is only labeled data, so img_ids_list is labeled_ids
    train_ds = build_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_JSON,transform=TransformConfig.TRAIN_TRANSFORM, img_ids_list=labeled_ids)

    val_ds = build_dataset(VAL_IMG_DIR, VAL_MASK_DIR, VAL_JSON,transform=TransformConfig.VAL_TRANSFORM,img_ids_list=None) # full VAL set
    # shuffle must be True, bcause we want to mix the labeled data each epoch
    train_loader = build_loader(train_ds,batch_size=TrainConfig.BATCH_SIZE,shuffle=True)
    val_loader = build_loader(val_ds,batch_size=1,shuffle=False)

    best_ckpt = train_with_validation(
        model,train_loader,val_loader,DEVICE,optimizer=optimizer_new,scheduler=scheduler_new,criterion=src.config.criterion,log_dir=work_dir
    )

    print(f"####### Training round {(round_idx+2)*10}% completed. Best ckpt: {best_ckpt} #######")
    return best_ckpt

def run_train_round(labeled_ids, round_idx):
    work_dir = ensure_dir(os.path.join(WORK_BASE, f'{METHOD}_train_{(round_idx + 2) * 10}%'))

    # Serialize labeled IDs for all ranks
    env = os.environ.copy()
    env['AL_IMG_IDS_JSON'] = json.dumps(sorted(list(labeled_ids)))  # stable order
    env['AL_WORKDIR'] = work_dir
    # Optional: if you want to override config.DataConfig.WORK_DIR implicitly, you can set another env var
    # but since train.py takes log_dir from function call, we'll just let it write into its default;
    # or patch train.py to read AL_WORKDIR and pass to train_with_validation.

    # Launch DDP: adjust nproc_per_node to your GPU count
    cmd = [
        'torchrun', '--nproc_per_node', '2', 'src/train_AL.py'
    ]
    subprocess.run(cmd, check=True, env=env)

    # After training, find the latest timestamp dir and the best ckpt
    # If your train.py writes to config.DataConfig.WORK_DIR/<timestamp>/best_mIou_epoch_*.pth
    base_dir = work_dir  # or 'work_dir' if you patched train.py to use AL_WORKDIR
    # pick the most recent timestamp directory
    run_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    assert run_dirs, f'No run directories found in {base_dir}'
    last_run = os.path.join(base_dir, run_dirs[-1])
    ckpts = sorted(glob.glob(os.path.join(last_run, 'best_mIou_epoch_*.pth')))
    assert ckpts, f'No best checkpoint found under {last_run}'
    best_ckpt = ckpts[-1]

    print(f"####### Training round {(round_idx + 2) * 10}% completed. Best ckpt: {best_ckpt} #######")
    return best_ckpt



def run_test_round(ckpt_path,round_idx):
    work_dir = ensure_dir(os.path.join(WORK_BASE,f'{METHOD}_test_{(round_idx + 2) * 10}%'))
    print(f"####### Starting testing round {(round_idx+2)*10}%, out_dir: {work_dir} #######")
    model = Model(num_classes=DataConfig.NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state)

    test_ds = build_dataset(TEST_IMG_DIR, TEST_MASK_DIR, TEST_JSON,transform=TransformConfig.TEST_TRANSFORM,img_ids_list=None) # full TEST set
    test_loader = build_loader(test_ds, batch_size=1, shuffle=False, workers=2)
    results = test_fn(model, test_loader, DataConfig.NUM_CLASSES, DEVICE, test_dir=work_dir)

    print(f"####### Testing round {(round_idx+2)*10}% completed. mIou: {results['Foreground_mIoU']} #######")
    with open(os.path.join(work_dir, f'{METHOD}_{(round_idx+2)*10}%_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
        print(f"####### Test results saved to {os.path.join(work_dir, f'{METHOD}_{(round_idx+2)*10}%_test_results.json')}")
    return results

def main():
    # initial 10% split, and all methods share the same initial split
    split_path = False
    ckpt_path = None
    if split_path and os.path.exists(split_path):
        labeled, pool, _ = load_initial_split(split_path)
        labeled_ic, pool_ic = set(labeled), set(pool)
        ckpt_prev = ckpt_path
    else:
        labeled, pool = initial_split(TRAIN_JSON, TRAIN_IMG_DIR, initial_ratio=0.1)
        labeled_ic = set(labeled)
        pool_ic = set(pool)
        print(f'####### 10% initial training: {len(labeled_ic)} labeled, {len(pool_ic)} in pool #######')
        # First(initial) round, 10% labeled, 90% unlabeled
        ckpt_prev = run_train_round(labeled_ic, -1)
        _ = run_test_round(ckpt_prev, -1)

    # AL rounds
    for r in range(N_ROUNDS):
        print(f'\n####### Method:{METHOD} {(r+2)*10}% labeled training, {len(labeled_ic)} labeled, {len(pool_ic)} in pool #######')
        # 1. Score pool
        pool_list = list(pool_ic) # set type for updating, but list type for indexing
        probs = predict_pool_probs(ckpt_prev, pool_list,batch_size=4)  # list of arrays, each array is [num_classes]
        ###### empty gpu memory ######
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ###### empty gpu memory ######
        # 2. Query
        query_method = QueryInstanceUncertainty(measure=METHOD)
        unlabeled_indices = np.arange(len(pool_list))
        selected = query_method.select_by_prediction_mat(unlabeled_indices, predict=probs,batch_size=N_QUERY)
        selected_ids = [pool_list[i] for i in selected]  # map back to img_ids
        print(f"####### Selected {len(selected_ids)} samples from pool #######")
        # 3. Update labeled and pool sets
        labeled_ic.update(selected_ids)
        pool_ic.difference_update(selected_ids)
        print(f"####### Added {len(selected_ids)} -> labeled = {len(labeled_ic)} | pool={len(pool_ic)}  #######")
        # 4. Train & Test
        ckpt_prev = run_train_round(labeled_ic,r)
        _ = run_test_round(ckpt_prev,r)

if __name__ == '__main__':
    main()
