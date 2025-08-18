# lstm_engine.py

"""
This module defines the full training pipeline as a reusable function using the provided Dataset class,
model architecture, and Trainer for training a volatility forecasting LSTM model.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.datasets import LSTMDataset
from src.models import CNNBiLSTMAttention
from src.training import Trainer

def run_lstm_training(batch_size=64, epochs=50, lr=0.001, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Preprocessed Data ---
    X_train = torch.from_numpy(np.load('../data/model_input/X_train.npy')).float()
    X_test = torch.from_numpy(np.load('../data/model_input/X_test.npy')).float()
    y_train = torch.from_numpy(np.load('../data/model_input/y_train.npy')).float()
    y_test = torch.from_numpy(np.load('../data/model_input/y_test.npy')).float()

    # --- Create Datasets and DataLoaders ---
    train_dataset = LSTMDataset(X_train, y_train)
    test_dataset = LSTMDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Define Model ---
    model = CNNBiLSTMAttention(
        input_dim=X_train.shape[2],
        hidden_dim=64,
        lstm_layers=2,
        attention=True,
        cnn=True,
        bidirectional=True
    )

    # --- Loss Function and Optimizer ---
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Initialize Trainer ---
    trainer = Trainer(model, loss_fn, optimizer, device=device)

    # --- Train the Model ---
    trainer.train(train_dataset, val_dataset=test_dataset, epochs=epochs, batch_size=batch_size, log_wandb=True)

    # Optionally save the trained model
    torch.save(model.state_dict(), '../outputs/volatility_lstm.pth')

    return model