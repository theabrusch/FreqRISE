import torch
import torch.nn.functional as F
from src.freqrise import FreqRISE
from src.freqrise.masking import mask_generator


def mask_and_predict(model, test_loader, importance, quantile, domain = 'fft', device = 'cpu', stft_params = None):
    model.eval().to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        ce_loss = 0
        mean_true_class_prob = 0
        for i, batch in enumerate(test_loader):
            data, true_label = batch
            if domain == 'fft':
                data = torch.fft.rfft(data, dim=-1).to(device)
            elif domain == 'stft':
                data_shape = data.shape
                data = torch.stft(data.squeeze().to(device), **stft_params, return_complex = True)
            else:
                data = data.to(device)
            if importance == 'random':
                imp = torch.rand_like(data).float()
            elif importance == 'amplitude':
                imp = torch.abs(data)
            else:
                if domain in ['fft', 'time']:
                    imp = importance[i].reshape(-1, 1, 1, data.shape[-1])
                else:
                    imp = importance[i].squeeze()
            q = torch.quantile(imp, quantile).unsqueeze(-1)
            data[imp > q] = 0
            if domain == 'fft':
                data = torch.fft.irfft(data, dim=-1)
            elif domain == 'stft':
                data = torch.istft(data, length = data_shape[-1], **stft_params, return_complex = False)
                data = data.view(data_shape)
            output = model(data.float()).detach().cpu()
            _, predicted = torch.max(output, 1)
            total += true_label.size(0)
            correct += (predicted == true_label).sum().item()
            # one hot encode true label
            mean_true_class_prob += torch.take_along_dim(F.softmax(output, dim=1), true_label.unsqueeze(1), dim = 1).sum().item()
            ce_loss += F.cross_entropy(output, true_label).item()/len(batch)
    return correct / total, mean_true_class_prob / total, ce_loss