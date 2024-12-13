import numpy as np
import torch
import torch.nn.functional as F
from quantus.metrics import Complexity
from src.explainability.metrics import relevance_rank_accuracy
from src.data.generators import powerset
from src.lrp import dft_lrp
from src.lrp import lrp_utils

def deletion_curves(model, test_loader, importance, quantiles, domain = 'fft', device = 'cpu', stft_params = None, cutoff = None):
    deletion = {
        'mean_prob': [],
        'quantiles': quantiles
    }
    for quantile in quantiles:
        mean_true_class_prob = mask_and_predict(model, test_loader, importance, quantile, domain = domain, device = device, stft_params = stft_params, cutoff = cutoff)
        deletion['mean_prob'].append(mean_true_class_prob)
    auc = np.trapz(deletion['mean_prob'], deletion['quantiles'])
    deletion['AUC'] = auc
    return deletion

def complexity_scores(attributions, cutoff = None, only_pos = False):
    comp_scores = []
    comp = Complexity()
    for i, sample in enumerate(attributions):
        if only_pos:
            sample[sample < 0] = 0
        sample = (sample - np.min(sample))/(np.max(sample) - np.min(sample))
        if cutoff is not None:
            cut = np.quantile(sample, cutoff)
            sample[sample < cut] = 0
        # flatten sample
        sample = sample.flatten()
        comp_scores.append(comp.evaluate_instance(sample, sample))
    return comp_scores

def localization_scores(attributions, labels, cutoff = None, only_pos = False):
    loc_scores = []
    ks = [5, 16, 32, 53]
    classes_ = powerset(ks)
    for i, sample in enumerate(attributions):
        if labels[i]== 0:
            continue
        if only_pos:
            sample[sample < 0] = 0
        sample = (sample - np.min(sample))/(np.max(sample) - np.min(sample))
        if cutoff is not None:
            cut = np.quantile(sample, cutoff)
            sample[sample < cut] = 0
        # flatten sample
        sample = sample.flatten()
        loc_scores.append(relevance_rank_accuracy(sample, np.array(classes_[labels[i]])))
    return loc_scores


def mask_and_predict(model, test_loader, importance, quantile, domain = 'fft', device = 'cpu', stft_params = None, cutoff = None):
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
            shape = data.shape
            if importance == 'random':
                imp = torch.rand_like(data).float()
            elif importance == 'amplitude':
                imp = torch.abs(data)
            else:
                if domain in ['fft', 'time']:
                    imp = importance[i].reshape(-1, 1, 1, data.shape[-1])
                else:
                    imp = importance[i].squeeze()
                imp[imp < 0] = 0
            if cutoff is not None:
                if cutoff == 'mean':
                    flattened_imp = imp.reshape(shape[0], -1)
                    cutoff = flattened_imp.mean(-1, keepdim=True)
                    # set all values below mean to 0
                    flattened_imp[flattened_imp < cutoff] = 0
                    imp = flattened_imp.view(shape)
                else:
                    flattened_imp = imp.reshape(shape[0], -1)
                    cut = flattened_imp.quantile(cutoff, dim=-1, keepdim=True)
                    # set all values below mean to 0
                    flattened_imp[flattened_imp < cut] = 0
                    imp = flattened_imp.view(shape)

            # take q percent largest values 
            flattened_imp = imp.reshape(shape[0], -1)
            k = int(quantile * flattened_imp.size(1))
            # Find top 10% (T * D * 10%)
            topk_values, topk_indices = torch.topk(flattened_imp, k=k, dim=1)
            mask = torch.zeros_like(flattened_imp, dtype=torch.bool)
            # Set the positions of the top-k elements to True
            mask.scatter_(1, topk_indices, True)
            mask = mask.view(shape).to(device)
            data = data * (~mask)

            if domain == 'fft':
                data = torch.fft.irfft(data, dim=-1)
            elif domain == 'stft':
                data = torch.istft(data, length = data_shape[-1], **stft_params, return_complex = False)
                data = data.view(data_shape)
            output = model(data.float()).detach().cpu()
            total += true_label.size(0)
            # one hot encode true label
            mean_true_class_prob += torch.take_along_dim(F.softmax(output, dim=1), true_label.unsqueeze(1), dim = 1).sum().item()

    return mean_true_class_prob / total

def compute_gradient_scores(model, testloader, attr_method, domain = 'fft', stft_params = None):
    lrp_scores = []
    for sample, target in testloader:
        if domain == 'stft':
            cuda = False
        else:
            cuda = torch.cuda.is_available()
        if cuda:
            sample = sample.cuda()
            model = model.cuda()
        relevance_time = lrp_utils.zennit_relevance(sample.float(), model, target=target, attribution_method=attr_method, cuda=cuda)
        if domain == 'fft':
            dftlrp = dft_lrp.DFTLRP(sample.shape[-1], 
                                    leverage_symmetry=True, 
                                    precision=32,
                                    cuda = cuda,
                                    create_stdft=False,
                                    create_inverse=False
                                    )
            signal_freq, relevance_freq = dftlrp.dft_lrp(relevance_time, sample.float(), real=False, short_time=False)
            lrp_scores.append(torch.tensor(relevance_freq))
        elif domain == 'stft':
            relevance_freq = lrp_stft(relevance_time, sample, stft_params['n_fft'], cuda)
            lrp_scores.append(torch.tensor(relevance_freq))
        else:
            lrp_scores.append(torch.tensor(relevance_time))
    return lrp_scores


def lrp_stft(relevance_time, sample, window_length, cuda):
    dftlrp = dft_lrp.DFTLRP(window_length, 
                            leverage_symmetry=True, 
                            precision=32,
                            cuda = cuda,
                            create_stdft=False,
                            create_inverse=False
                            )
    freq_relevance = np.zeros((sample.shape[0], window_length//2+1, sample.shape[-1]//window_length))

    for i in range(sample.shape[-1]//window_length):
        signal_freq, relevance_freq = dftlrp.dft_lrp(relevance_time[...,i*window_length:(i+1)*window_length], sample[...,i*window_length:(i+1)*window_length].float(), real=False, short_time=False)
        freq_relevance[...,i] = relevance_freq[:,0,0,:]
    return freq_relevance
