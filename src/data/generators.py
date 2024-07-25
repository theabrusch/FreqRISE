import numpy as np
from scipy.fft import rfftfreq
from statsmodels.tsa.arima_process import arma_generate_sample
import torch.nn as nn
import torch
from itertools import chain, combinations

class TSGenerator():
    def __init__(self, length, fs, num_variables = 1, n_freqs = 1):
        assert num_variables == 1, "Only one variable is supported at the moment"
        self.num_variables = num_variables
        self.length = length
        self.fs = fs
        self.n_freqs = n_freqs
        target_weights = []
        ranges_ = [(-2, -0.5), (0.5, 3)]
        for i in range(n_freqs):
            r = ranges_[i]
            target_weights.append(np.random.uniform(*r, (1, 1)))
        self.target_weights = np.concatenate(target_weights, axis = 1)
        self.target_bias = np.random.normal(0, 1, (1, 1))
    
    def generate(self, 
                 n_samples = 100, 
                 type='sine', 
                 phase=0, 
                 noiselevel=None):
        freqs = []
        max_freq = self.fs/2
        for i in range(self.n_freqs):
            freqs.append(np.random.uniform(0.1, max_freq, (n_samples, 1)))
        
        standardized_freqs = np.concatenate(freqs, axis = 1)
        target = standardized_freqs@self.target_weights.T + self.target_bias + np.random.normal(0, 0.1, (n_samples, 1))
        amp = []
        for _ in range(self.n_freqs):
            amp.append(np.random.uniform(0.8, 1, (n_samples, 1)))

        if type == 'sine':
            t = np.repeat(np.expand_dims(np.arange(0, self.length, 1/self.fs), axis = 0), n_samples, axis = 0)
            data = np.zeros(t.shape)
            for i in range(self.n_freqs):
                freq = freqs[i]
                data += amp[i] * np.sin(2*np.pi*freq*t + phase)
            if noiselevel is not None:
                data += np.random.normal(0, noiselevel, (n_samples, data.shape[1]))
                
            return data, freqs, target
        else:
            raise NotImplementedError("Only sine is supported at the moment")

class variable_freq_generator():
    def __init__(self, length, fs):
        self.length = length
        self.fs = fs

    def generate(self, 
                 n_samples = 100, 
                 max_frequencies = 5, 
                 noiselevel=None):
        freqs = []
        max_freq = self.fs/2
        t = np.arange(0, self.length, 1/self.fs)
        collect_freqs = []

        data = np.zeros((n_samples, len(t)))
        for i in range(n_samples):
            if np.random.rand() > 0.1:
                n_freqs = np.random.randint(1, max_frequencies+1)
                freqs = np.random.uniform(0.1, max_freq, n_freqs)
                amps = np.random.uniform(0.2, 1, n_freqs)
                collect_freqs.append(freqs)
                for amp, freq in zip(amps, freqs):
                    data[i] += amp*np.sin(2*np.pi*freq*t)
            else:
                collect_freqs.append([0])
        
        if noiselevel is not None:
            data += np.random.normal(0, noiselevel, (n_samples, data.shape[1]))
        return data, collect_freqs

class TStestGenerator():
    def __init__(self, length, fs, num_variables = 1):
        assert num_variables == 1, "Only one variable is supported at the moment"
        self.num_variables = num_variables
        self.length = length
        self.fs = fs
    
    def generate(self, 
                 n_samples = 100, 
                 type='sine', 
                 n_freqs = 2,
                 fix_frequency = True,
                 phase=0, 
                 use_digital_freqs = False,
                 noiselevel=None,
                 target_weights = None,
                 target_bias = None):
        if n_freqs > 1 and fix_frequency:
            if use_digital_freqs:
                true_freqs = rfftfreq(int(self.length/2)*self.fs, 1/self.fs)
                freq_1 = np.repeat(np.random.choice(true_freqs, 1), n_samples)[:, np.newaxis]
            else:
                freq_1 = np.expand_dims(np.repeat(np.array(np.random.uniform(0.1, self.fs/2)), n_samples), 1)
        else:
            if use_digital_freqs:
                true_freqs = rfftfreq(int(self.length/2)*self.fs, 1/self.fs)
                freq_1 = np.random.choice(true_freqs, n_samples)[:, np.newaxis]
            else:
                freq_1 = np.random.uniform(0.1, self.fs/2, (n_samples, 1))
        freqs = [freq_1]
        for _ in range(n_freqs-1):
            if use_digital_freqs:
                freq_2 = np.random.choice(true_freqs, n_samples)[:, np.newaxis]
            else:
                freq_2 = np.random.uniform(0.1, self.fs/2, (n_samples, 1))
            freqs.append(freq_2)
        amp = []
        for _ in range(n_freqs):
            amp.append(1)
            
        if target_weights is not None:
            standardized_freqs = np.concatenate(freqs, axis = 1)
            target = standardized_freqs@target_weights.T + target_bias + np.random.normal(0, 0.1, (n_samples, 1))
            # normalize target
            target = (target - np.min(target))/(np.max(target) - np.min(target))
        else:
            target = None

        if type == 'sine':
            t = np.repeat(np.expand_dims(np.arange(0, self.length, 1/self.fs), axis = 0), n_samples, axis = 0)
            data = np.zeros(t.shape)
            for i in range(n_freqs):
                freq = freqs[i]
                data += amp[i] * np.sin(2*np.pi*freq*t + phase)
            if noiselevel is not None:
                data += np.random.normal(0, noiselevel, (n_samples, data.shape[1]))
                
            return data, freqs, target
        else:
            raise NotImplementedError("Only sine is supported at the moment")

class RareModel(nn.Module):
    def __init__(self, t, f):
        self.t = t
        self.f = f
    def forward(self, x):
        out = torch.zeros(x.shape[0], x.shape[-1])
        out = torch.sum(x[:, self.f, self.t]**2, axis = 1)
        return out
        
def RareGenerator(saliency = 'time'): 
    data = arma_generate_sample(ar=[2, 0.5, 0.2, 0.1], ma = [2.], scale = 1.0, nsample=(50,50), axis = 1)
    target = np.zeros(50)
    if saliency == 'time':
        t = np.random.randint(0, 45)
        target[t:t+5] = np.sum(data[12:38, t:t+5]**2, axis = 0)
    elif saliency == 'feature':
        f = np.random.choice(np.arange(50), size = 5, replace = False)
        target[12:38] = np.sum(data[f, 12:38]**2, axis = 0)
    return data, target

def powerset(iterable):
    s = list(iterable)  # Convert the input iterable to a list.
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

def frequency_lrp_dataset(samples, length =  2560, noiselevel = 0.01, M_min=None, M_max = None, integer_freqs = True, return_ks = False, seed = 42):
    ks = np.array([5, 16, 32, 53])
    if not integer_freqs:
        # set seed
        np.random.seed(seed)
        ks = ks + np.random.uniform(0, 1, ks.shape)
        # remove seed
        np.random.seed(None)
    classes_ = powerset(ks)
    all_freqs = np.linspace(1, 60, 60, dtype = np.int32)
    for k in ks:
        idx = np.where(all_freqs == k)[0]
        all_freqs = np.delete(all_freqs, idx)
    data = np.zeros((samples, length))
    labels = []
    for i in range(samples):
        class_ = np.random.randint(0, len(classes_))
        freqs = np.array(classes_[class_])
        data[i] += np.random.normal(0, noiselevel, length)
        #data[i] += np.sum([np.sin(2*np.pi*freq/length*np.arange(0, length) + np.random.uniform(0, 2*np.pi)) for freq in freqs], axis = 0)
        # if M is a number then we add M random frequencies
        if M_min is not None:
            M = np.random.randint(M_min, M_max)
            if integer_freqs:
                # append to freqs
                freqs = np.append(freqs, np.random.choice(all_freqs, M-len(freqs), replace = False))
            else:
                # sample uniformly, but exclude a range of 1 Hz around frequencies in ks
                while len(freqs) < M:
                    f = np.random.uniform(1, 60)
                    if np.all(np.abs(ks - f) > 1):
                        freqs = np.append(freqs, f)

            data[i] += np.sum([np.sin(2*np.pi*freq/length*np.arange(0, length) + np.random.uniform(0, 2*np.pi)) for freq in freqs], axis = 0)
        labels.append(class_)
    if return_ks:
        return data, labels, ks
    return data, labels
