"""
Trainer class for PyTorch models
Supports modular training on any dataset/model combo with W&B logging (optional).
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device='cpu', scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x).squeeze()
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            epoch_loss += loss.item() * x.size(0)

        return epoch_loss / len(dataloader.dataset)

    def train(self, train_dataset, batch_size=32, epochs=10):
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch}: Train Loss = {loss:.6f}")
            wandb.log({"train_loss": loss, "epoch": epoch})