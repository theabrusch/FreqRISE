from torch import optim, nn, Tensor, utils
from torch.fft import irfft, rfft
import lightning as L
import torch

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_layers = 1, hidden_size = 2560, output_size = 1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, only_feats = False):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        if only_feats:
            return self.mlp[:-1](x)
        return self.mlp(x)
    
class ConvModel(nn.Module):
    def __init__(self, output_size, hidden_layers = 1, hidden_size = 64, kernel_size = 11):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2))
        for _ in range(hidden_layers):
            layers.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
        layers.append(nn.Conv1d(hidden_size, output_size, kernel_size=kernel_size, padding=kernel_size//2))
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
    
    def forward(self, x, only_feats = False):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        if only_feats:
            return self.cnn[:-3](x)
        return self.cnn(x)

class LightningModel(L.LightningModule):
    def __init__(self, model, loss, learning_rate, weight_decay = 0.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if loss == 'mse':
            self.loss = nn.MSELoss()
        elif loss == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Loss not supported")
        
    def forward(self, x, only_feats = False):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # compute accuracy
        _, predicted = torch.max(y_hat.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        self.log_dict({"val/accuracy": correct / total, "val/loss": loss}, on_epoch=True)
        return loss
    
    def test_model(self, testloader, device = 'cpu'):
        correct = 0
        total = 0
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for data in testloader:
                signals, labels = data
                outputs = self(signals.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.detach().cpu() == labels).sum().item()
        self.model.to('cpu')
        return correct / total
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)
