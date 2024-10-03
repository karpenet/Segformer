import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Optimizer
from Segformer import Segformer
import torch
from typing import Optional
import yaml
from dataset import Datasets


def train_batch(model, images, targets, criterion, optimizer: Optimizer, device):
    images = images.to(device)
    targets = targets.to(device)

    pred = model(images)
    loss = criterion(pred, targets)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss

def validate_model(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            pred = model(images)
            miou = model.miou(pred, targets)

    return miou


def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


def train(
    model: nn.Module,
    optimizer: Optimizer,
    config: wandb.config,
    criterion,
    device,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
):
    batch_ct = 0
    example_ct = 0

    for epoch in tqdm(range(config.epochs)):
        for _, (images, targets) in enumerate(train_loader):
            loss = train_batch(model, images, targets, criterion, optimizer, device)

            example_ct += len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        if val_loader:
            miou = validate_model(model, val_loader, device)
            wandb.log({"mIoU": miou}, step=example_ct)
            print(f"mIoU after {str(epoch).zfill(2)} epoch: {miou:.3f}")



def model_pipeline(hyperparameters):
    with wandb.init(project="Segformer-B0", config=hyperparameters):
        config = wandb.config

        model, train_loader, val_loader, criterion, optimizer, device = make(config)
        print(model)

        train(model, optimizer, config, criterion, device, train_loader, val_loader)

    return model


def make(config):
    train_loader = DataLoader(Datasets[config.dataset].train_dataset(), batch_size=config.batch_size)
    val_loader = DataLoader(Datasets[config.dataset].val_dataset(), batch_size=config.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Segformer().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate))

    return model, train_loader, val_loader, criterion, optimizer, device


if __name__ == "__main__":
    with open("train_config.yaml", "r") as file:
        config = yaml.safe_load(file)['Segformer-B0']

    model_pipeline(hyperparameters=config)
