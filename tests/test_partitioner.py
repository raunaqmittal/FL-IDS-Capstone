import numpy as np

from src.components.data.data_partitioner import (
    partition_non_iid,
    save_partitions,
    load_partition,
    load_partition_dataloaders,
)

X = np.random.randn(1000, 78).astype(np.float32)
y = np.random.randint(0, 2, size=1000).astype(np.int64)

partitions = partition_non_iid(X, y, num_clients=3, alpha=0.5)
save_partitions(partitions, output_dir=None)

X_train, y_train, X_val, y_val = load_partition(0)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

train_loader, val_loader = load_partition_dataloaders(0, batch_size=8)

xb, yb = next(iter(train_loader))
print("Batch X:", xb.shape)
print("Batch y:", yb.shape)

print("Partitioner works")