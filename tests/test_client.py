# test_client.py

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.components.model.model import MLPClassifier
from src.components.client.client import FLIDSClient


def make_loader():
    x = torch.randn(64, 78)
    y = torch.randint(0, 2, (64,))
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)


train_loader = make_loader()
val_loader = make_loader()

model = MLPClassifier(78, [64, 32, 16], 2)

client = FLIDSClient(
    cid="0",
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    config={
        "local_epochs": 1,
        "lr": 1e-3,
        "device": "cpu",
    },
)

params = client.get_parameters({})

new_params, num_examples, train_metrics = client.fit(params, {})
print("Train examples:", num_examples)
print("Train metrics:", train_metrics)

loss, val_examples, val_metrics = client.evaluate(new_params, {})
print("Val examples:", val_examples)
print("Val loss:", loss)
print("Val metrics:", val_metrics)

print("FLIDSClient works")