import numpy as np
from itertools import chain, combinations

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
