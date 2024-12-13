#@title RELAX
from torch.fft import rfft as tfft
from torch.fft import irfft as tifft
import torch
import torch.nn as nn
from src.explainability.masking import mask_generator

class FreqRISE(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 batch_size: int = 10,
                 num_batches: int = 300,
                 device: str = 'cpu',
                 domain = 'fft',
                 use_softmax = False,
                 stft_params = None,
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.domain = domain
        self.stft_params = stft_params

        self.num_batches = num_batches
        self.use_softmax = use_softmax
        self.encoder = encoder.eval().to(self.device)

    def forward(self, input_data, mask_generator, **kwargs) -> None:
        i = 0 
        p = []
        input_data = input_data.unsqueeze(0).to(self.device)
        # cast data to domain of interest
        if self.domain == 'fft':
            input_fft = tfft(input_data)
        elif self.domain == 'stft':
            input_fft = torch.stft(input_data.squeeze(), return_complex=True, **self.stft_params)
        else:
            input_fft = input_data
        shape = input_fft.shape
        original_shape = input_fft.shape

        mask_type = torch.complex64 if self.domain in ['fft', 'stft'] else torch.float32

        for _ in range(self.num_batches):
            for masks in mask_generator(self.batch_size, shape, self.device, dtype = mask_type, **kwargs):
                if len(masks) == 2:
                    x_mask, masks = masks
                else:
                    x_mask = input_fft*masks
                    if self.domain == 'fft':
                        x_mask = tifft(x_mask, dim=-1)
                    elif self.domain == 'stft':
                        x_mask = torch.istft(x_mask, length = original_shape[-1], return_complex = False, **self.stft_params)
                        x_mask = x_mask.reshape((self.batch_size, *original_shape))

                with torch.no_grad():
                    predictions = self.encoder(x_mask.float().to(self.device), only_feats = False).detach()
                if self.device == 'mps':
                    predictions = predictions.cpu()
                if self.use_softmax:
                    predictions = torch.nn.functional.softmax(predictions, dim=1)
                sal = torch.matmul(predictions.transpose(0,1).float(), masks.view(self.batch_size, -1).abs().float()).transpose(0,1).unsqueeze(0).cpu()
                if self.domain == 'stft':
                    sal = sal.view(1, *input_fft.shape, -1)
                p.append(sal)
                i += 1
        importance = torch.cat(p, dim=0).sum(dim=0)/(self.num_batches*self.batch_size)
        return importance
    
    def forward_dataloader(self, dataloader, num_cells, probability_of_drop):
        freqrise_scores = []
        i = 0
        if self.domain == 'stft':
            num_spatial_dims = 2
        else:
            num_spatial_dims = 1
        for data, target in dataloader:
            batch_scores = []
            print("Computing batch", i+1, "/", len(dataloader))
            i+=1
            for sample, y in zip(data, target):
                m_generator = mask_generator
                with torch.no_grad(): 
                    importance = self.forward(sample.float().squeeze(0), mask_generator = m_generator, num_spatial_dims = num_spatial_dims, num_cells = num_cells, probablity_of_drop = probability_of_drop)
                importance = importance.cpu().squeeze()[...,y]/probability_of_drop
                # min max normalize
                importance = (importance - importance.min()) / (importance.max() - importance.min())
                batch_scores.append(importance)
            freqrise_scores.append(torch.stack(batch_scores))
        return freqrise_scores