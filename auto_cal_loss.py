import torch
import json

from config import compute_class_weights, train_dataset

weights = compute_class_weights(train_dataset, num_classes=8)

# Optionally downscale background weight
weights[0] *= 0.1
weights = weights / weights.sum() * 8

# Save as JSON
weights_path = 'class_weights.json'
with open(weights_path, 'w') as f:
    json.dump(weights.tolist(), f)