import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, model_path, patience=7, verbose=False, delta=0, counter=0, val_loss_min=np.Inf, best_score=None):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path
        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, scheduler, epoch, args):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch, args)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch, args)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch, args):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint_save_path = os.path.join(self.checkpoint_path, args.data_mode + '_' + args.model_mode + '_' + args.context + str(args.max_span_len) + 'es.pth')
        # model_save_path = os.path.join(self.model_path, args.data_mode + '_' + args.model_mode + '_' + args.context  + str(args.max_span_len) + 'es.pkl')
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                'scheduler': scheduler.state_dict(), 'val_loss_min': self.val_loss_min, 
                'best_score': self.best_score, 'counter': self.counter, 'delta': self.delta}
        torch.save(state, checkpoint_save_path)	# save the best model
        # torch.save(model, model_save_path)
        self.val_loss_min = val_loss

