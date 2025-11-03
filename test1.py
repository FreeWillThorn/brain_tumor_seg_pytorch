import argparse
import torch
from sympy.solvers.diophantine.diophantine import reconstruct
from torch.utils.data import DataLoader
from src.model_all import Model
from src.dataset import get_test_dataset
#reconstruct the code if i got time
def main():
    parser = argparse.ArgumentParser(description="Test a trained model on the test set.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained .pth model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--save-metrics', type=str, default=None, help='Path to save test metrics JSON')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Model(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    test_dataset = get_test_dataset(args.test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    results = test(model, test_loader, args.num_classes, device, save_path=args.save_metrics)
    print("Test Results:", results)
