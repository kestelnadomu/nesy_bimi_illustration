"""
TODO
"""
import yaml, os, sys

from typing import Dict, Callable
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tch_loss import loss_experiment
from tch_metrics import metrics_dict
from tch_data import create_dataloaders, load_and_transform_data
from tch_callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(512, 256, 128), output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train(model: nn.Module,
          loss_fn: nn.Module,
          optimizer: optim.Optimizer,
          dataloader: DataLoader,
          device=torch.device("cpu")):
    model.train()
    total_loss = 0.0
    for X_batch, A_batch, y_batch in tqdm(dataloader):
        X_batch = X_batch.to(device)
        A_batch = A_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_batch, y_pred, A_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model: nn.Module,
             loss_fn: nn.Module,
             dataloader: DataLoader,
             metrics: Dict[str, Callable],
             decision_threshold = 0.5,
             device=torch.device("cpu")) -> dict:
    model.eval()
    ys = []
    yps = []
    As = []
    with torch.no_grad():
        for X_batch, A_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            A_batch = A_batch.to(device)
            y_pred = model(X_batch)
            ys.append(y_batch.cpu())
            yps.append(y_pred.cpu())
            As.append(A_batch.cpu())
    y = torch.cat(ys)
    yp = torch.cat(yps)
    a = torch.cat(As)

    # Loss as first metric
    out = {"loss": loss_fn(y, yp, a).item()}

    # To numpy
    y = y.detach().numpy()
    yp = (yp.detach().numpy() >= decision_threshold)
    a = a.detach().numpy()

    # Calculate metrics
    for metric_key, metric_function in metrics.items():
        out[metric_key] = metric_function(y, yp, a)
    return out


def main(config_path: os.PathLike ="config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Data ---
    # Load data from FairGround
    d = load_and_transform_data(cfg["data"]["dataset_id"])

    # Split data
    train_df, val_df, test_df = d["dataset"].train_test_val_split(d["dataframe"], test_size=cfg["data"]["test_size"], val_size=cfg["data"]["val_size"]) 

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        list(d["feature_columns"]), d["sensitive_columns"], d["target_column"],
        cfg["train"].get("batch_size", 1024)
    )

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        input_dim = len(d["feature_columns"]),
        hidden_layer_sizes = tuple(cfg["model"].get("hidden_layer_sizes", (512,256,128)))
    ).to(device)

    # Loss and metrics
    loss_fn = loss_experiment[cfg["train"]["loss"]]
    metrics = {m: metrics_dict[m] for m in cfg["eval"]["metrics"]}

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr = cfg["train"].get("lr", 1e-3),
        weight_decay = cfg["train"].get("weight_decay", 1e-4)
    )


    # --- Callbacks ---
    patience = cfg["train"]["callbacks"].get("patience", 15)
    min_lr = cfg["train"]["callbacks"].get("min_lr", 1e-6)
    checkpoint_path = cfg["train"]["callbacks"].get("checkpoint_path", "best_model.pt")
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(optimizer, monitor="val_loss", factor=0.5, patience=3, min_lr=min_lr, verbose=1)


    # --- Training Loop ---
    epochs = cfg["train"].get("epochs", 10)
    for epoch in range(epochs):
        train_loss = train(model, loss_fn, optimizer, train_loader, device=device)
        val_metrics = evaluate(model, loss_fn, val_loader, metrics, device=device)
        val_loss = val_metrics.get("loss", 0.0)  # or use another metric if desired
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val: {val_metrics}")

        # Callbacks
        early_stopping.step(val_loss, model, epoch)
        model_checkpoint.step(val_loss, model)
        reduce_lr.step(val_loss)
        if early_stopping.stop_training:
            break

    # --- Evaluation ---
    test_metrics = evaluate(model, loss_fn, test_loader, metrics, device=device)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main(sys.argv[1])