import torch
import numpy as np
from src.lrp import dft_lrp
from src.lrp import lrp_utils

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