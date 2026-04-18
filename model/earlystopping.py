import numpy as np
import torch
import logging

class little_EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 80
            verbose (bool): If True, prints a message for each loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = None
        self.delta = delta

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
            return

        if self.loss_min is None:
            self.loss_min = loss
            self.save_checkpoint(loss, model)
        elif loss > self.loss_min - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.loss_min = loss
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        """
        Saves model when loss decreases.
        """
        torch.save(model.state_dict(), 'checkpoint.pt')