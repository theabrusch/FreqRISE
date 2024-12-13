import numpy as np
import torch

def relevance_rank_accuracy(relevance, true_label):
    if isinstance(relevance, torch.Tensor):
        relevance = relevance.numpy()
    if isinstance(true_label, torch.Tensor):
        true_label = true_label.numpy()
    true_label = true_label
    K = len(true_label)
    top_K = np.argsort(relevance)[-K:]
    accuracy = len(set(top_K).intersection(set(true_label)))/K
    return accuracy