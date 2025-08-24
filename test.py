import json, datetime
import torch
from torch.utils.data import DataLoader
from src.utils import *
from tqdm import tqdm
import os
from src import config
from src.metrics import Metrics
from src.model_all import Model,Swin_backbone  # Adjust if your model class is named differently
from src.config import DataConfig  # Adjust import as needed
from src.cocodataset import COCODataset  # Replace with your dataset class
from src.utils import print_test_metrics, tensor_to_python
from collections import OrderedDict


def test(model, test_loader, num_classes, device, test_dir, save_path=None):
    model.eval()
    test_metrics = Metrics(num_classes, device)

    #total_loss = 0.0


    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            index_pic = 0
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)[0]
            preds = torch.argmax(outputs, dim=1)
            test_metrics.generate_matrix(preds, masks)
            #vis_path = os.path.join(save_path, f'test.png')
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_path = os.path.join(config.DataConfig.TEST_DIR, 'predictions', str(date) + "_" + str(index_pic) + ".png")
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            if batch_idx% 10 == 0:
                visualize_prediction(
                    images[0],
                    masks[0],
                    preds[0],
                    vis_path
                )
                index_pic += 1

    results = {
        #'test_loss': total_loss / len(test_loader) if criterion is not None else None,
        'mIoU': test_metrics.mean_iou(),
        'mAcc': test_metrics.mean_accuracy(),
        'Foreground_mIoU': test_metrics.mean_iou(ignore_inde_index=0),
        'Foreground_mAcc': test_metrics.mean_accuracy(ignore_inde_index=0),
        'iou_per_class': test_metrics.iou_per_class(),
        'acc_per_class': test_metrics.acc_per_class()

    }
    print_test_metrics(results)
    results = tensor_to_python(results)

    if save_path:
        save_path = os.path.join(test_dir, str(date)  + ".json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Test results saved to {save_path}")

    return results

def main():

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank)
    num_classes = DataConfig.NUM_CLASSES  # Adjust as needed
    # Load model
    model = Model(num_classes=num_classes)  # No pretrained weights
    #model.load_state_dict(torch.load('/root/code/runs/20250619_011121/best_mIou_epoch_90.pth', map_location=device))
    state_dict = torch.load('/root/hy-nas/auto-weights-resnet50/20250823_005609/best_mIou_epoch_80.pth', map_location=device)
    # Remove 'module.' prefix if present

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove 'module.' if needed
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

    # Prepare test dataset and loader
    test_loader = DataLoader(
        config.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2

    ) # Adjust as neede

    results = test(model, test_loader, num_classes, device,test_dir=config.DataConfig.TEST_DIR)

    print(json.dumps(results, indent=4))



if __name__ == "__main__":
    main()