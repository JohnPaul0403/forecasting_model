"""
Tester class for evaluating any PyTorch model on test data.
Computes RMSE, MAE, and QLIKE metrics.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Tester:
    def __init__(self, model, loss_fn, device='cpu'):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device

    def test(self, test_dataset, batch_size=32):
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        preds = []
        true_vals = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                output = self.model(x).squeeze()

                preds.append(output.cpu().numpy())
                true_vals.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        true_vals = np.concatenate(true_vals)

        rmse = np.sqrt(mean_squared_error(true_vals, preds))
        mae = mean_absolute_error(true_vals, preds)
        qlike = np.mean(np.log(preds ** 2 + 1e-8) + (true_vals ** 2) / (preds ** 2 + 1e-8))

        print(f"\nTest Evaluation:")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE : {mae:.6f}")
        print(f"QLIKE: {qlike:.6f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'qlike': qlike,
            'predictions': preds,
            'true': true_vals
        }