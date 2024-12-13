from src.data.generators import frequency_lrp_dataset
from torch.utils.data import DataLoader, TensorDataset
from src.data.dataloader import AudioNetDataset
import torch

def load_data(args):
    if args.dataset == 'synthetic':
        test_data, test_labels = frequency_lrp_dataset(args.n_samples, length = 2560, noiselevel = 0., M_min = 10, M_max = 50)
        test_data, test_labels = torch.tensor(test_data).float(), torch.tensor(test_labels).long()
        test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=256)
    elif args.dataset == 'AudioMNIST':
        test_dset = AudioNetDataset(args.data_path, True, 'test', splits = [0], labeltype = args.labeltype, subsample = args.n_samples, seed = 0, add_noise=False)                     
        test_loader = DataLoader(test_dset, batch_size=10, shuffle=False)
    
    return test_loader
