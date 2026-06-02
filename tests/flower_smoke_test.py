import flwr as fl

from src.components.model.model import MLPClassifier
from src.components.client.client import FLIDSClient
from src.components.data.data_partitioner import load_partition_dataloaders
from src.configs.config import CONFIG

# Dimensions read from CONFIG to match real preprocessed data.
# Requires data_pipeline to have been run first (artifacts/data/*.npz must exist).
NUM_CLIENTS  = 3
INPUT_DIM    = CONFIG["model"]["input_dim"]    # 57 after variance + correlation filter
NUM_CLASSES  = CONFIG["model"]["num_classes"]  # 27 fine-grained CIC-IDS2017 classes
BATCH_SIZE   = 8


def client_fn(cid: str):
    cid_int = int(cid)

    train_loader, val_loader = load_partition_dataloaders(
        client_id=cid_int,
        batch_size=BATCH_SIZE,
    )

    model = MLPClassifier(
        input_dim=INPUT_DIM,
        hidden_dims=CONFIG["model"]["hidden_dims"],  # [256, 128, 64]
        num_classes=NUM_CLASSES,
    )

    client = FLIDSClient(
        cid=cid,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config={
            "local_epochs": 1,
            "lr": 1e-3,
            "device": "cpu",
        },
    )

    return client.to_client()


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)