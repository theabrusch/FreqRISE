import torch
from src.freqrise import FreqRISE
from src.freqrise.masking import mask_generator

def compute_freqrise_scores(model, 
                         testloader, 
                         exp_label = None, 
                         n_samples = 3000,  
                         num_cells = 100, 
                         probability_of_drop = 0.2, 
                         domain = 'fft',
                         stft_params = None,
                         use_softmax = False,
                         device = 'cpu'):
    relax_scores = []
    i = 0
    if domain == 'stft':
        num_spatial_dims = 2
    else:
        num_spatial_dims = 1
    for data, target in testloader:
        batch_scores = []
        print("Computing batch", i+1, "/", len(testloader))
        i+=1
        for sample, y in zip(data, target):
            # RELAX
            m_generator = mask_generator
            rise_time = FreqRISE(sample.float().squeeze(0), model, batch_size=500, num_batches=n_samples//500, device=device, domain=domain, stft_params=stft_params, use_softmax=use_softmax)
            with torch.no_grad(): rise_time.forward(mask_generator = m_generator, num_spatial_dims = num_spatial_dims, num_cells = num_cells, probablity_of_drop = probability_of_drop)
            if not exp_label is None:
                if y == exp_label:
                    importance = rise_time.importance.cpu().squeeze()[:,exp_label]/probability_of_drop
                else:
                    importance = 1-rise_time.importance.cpu().squeeze()[:,exp_label]/probability_of_drop
            else:
                importance = rise_time.importance.cpu().squeeze()[...,y]/probability_of_drop
            # min max normalize
            importance = (importance - importance.min()) / (importance.max() - importance.min())
            batch_scores.append(importance)
        relax_scores.append(torch.stack(batch_scores))
    return relax_scores

