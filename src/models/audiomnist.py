import torch.nn as nn
import torch
import numpy as np

def conv1d_outputsize(input_shape, kernel_size, stride, padding):
    h_in = input_shape
    h_out = ((h_in + 2*padding - (kernel_size - 1) - 1) // stride) + 1
    return h_out

class conv2dblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv2dblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        return x

class AudioNet(nn.Module):
    def __init__(self, input_shape, num_classes, return_probs = False):
        super(AudioNet, self).__init__()
        kernel_size = (1,3)
        stride = 1
        padding = (0,1)
        channels = [input_shape[0], 100, 64, 128, 128, 128, 128]
        cnn_layers = []
        input_len = input_shape[1]

        for i in range(len(channels)-1):
            cnn_layers.append(conv2dblock(channels[i], channels[i+1], kernel_size, stride, padding))
            input_len = conv1d_outputsize(input_len, kernel_size[1], stride, padding[1])
            input_len = conv1d_outputsize(input_len, 2, 2, 0)
        self.cnn = nn.Sequential(*cnn_layers)
        self.return_probs = return_probs

        self.classifier = nn.Sequential(
            nn.Linear(128*input_len, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x, only_feats = False, feats_and_class = False):
        if len(x.shape) == 2:
            x = x.transpose(0,1).unsqueeze(1).unsqueeze(1)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif x.shape[1] > x.shape[2]:
            x = x.transpose(1,2).unsqueeze(1)
        feats = self.cnn(x)
        x = feats.view(feats.size(0), -1)
        if only_feats or feats_and_class:
            for l in self.classifier[:-1]:
                x = l(x)
            if feats_and_class:
                return x, self.classifier[-1](x)
            return x
        x = self.classifier(x)
        if self.return_probs:
            x = nn.functional.softmax(x, dim = 1)
        return x
    
    def test_model(self, test_loader, label):
        correct = 0
        total = 0
        all_labels = []
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                inputs, digit, gender = data
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                if label == 'gender':
                    labels = gender
                else:
                    labels = digit
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.append(labels)
                predictions.append(predicted)
        all_labels = torch.stack(all_labels)
        predictions = torch.stack(predictions)
        accuracy = 100 * correct / total
        print(f'Test Accuracy for {label}: {accuracy}%')
        return accuracy, all_labels, predictions


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    From https://github.com/Bjarten/early-stopping-pytorch """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_extra_score = -np.inf

    def __call__(self, val_loss, model, extra_score = None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, extra_score=extra_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, extra_score=extra_score)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, extra_score = None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_extra_score = extra_score