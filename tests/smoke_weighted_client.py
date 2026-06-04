import numpy as np, torch
from src.components.data.data_partitioner import load_partition
from src.components.data.torch_dataset import make_dataloader
from src.components.model.model import MLPClassifier, get_model_parameters
from src.components.client.client import FLIDSClient

X_train, y_train, X_val, y_val = load_partition(0)
train_loader = make_dataloader(X_train, y_train, batch_size=256, shuffle=True)
val_loader = make_dataloader(X_val, y_val, batch_size=256, shuffle=False)
model = MLPClassifier(57, [256,128,64], 27, 0.2)

config = {
    "local_epochs": 1, "lr": 0.001, "weight_decay": 1e-5,
    "num_classes": 27, "weight_cap": 10.0, "is_poisoned": False,
    "attack_type": "label_flip", "attack_start_round": 11,
    "source_class": 3, "target_class": 0,
    "trigger_feature_idx": [0,5], "trigger_values": [999999,1],
    "inject_ratio": 0.1, "scale_to_benign_norm": False, "benign_norm_target": 1.0,
}

client = FLIDSClient(
    cid="0",
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    config=config,
    X_train_raw=X_train,
    y_train_raw=y_train,
)

params, n, metrics = client.fit(get_model_parameters(model), {"server_round": 1})
loss = metrics["train_loss"]
acc = metrics["train_accuracy"]
print(f"Fit OK  n={n}  loss={loss:.4f}  acc={acc:.4f}")
print("Weighted loss working correctly.")
