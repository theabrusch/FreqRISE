import torch
import argparse
from src.explainability.evaluation import deletion_curves, complexity_scores, localization_scores
from src.data.load_data import load_data
from src.models.load_model import load_model
import pickle
import os
import numpy as np


def main(args):
    test_loader = load_data(args)
    model = load_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'synthetic':
        attributions_path = f'{args.output_path}/{args.dataset}_{args.noise_level}_attributions_{args.explanation_domain}_{args.n_samples}.pkl'
        output_path = f'{args.output_path}/{args.dataset}_{args.noise_level}_evaluation_{args.explanation_domain}_{args.n_samples}.pkl'
    else:
        attributions_path = f'{args.output_path}/{args.dataset}_{args.labeltype}_attributions_{args.explanation_domain}_{args.n_samples}.pkl'
        output_path = f'{args.output_path}/{args.dataset}_{args.labeltype}_evaluation_{args.explanation_domain}_{args.n_samples}.pkl'
    
    if os.path.exists(attributions_path):
        with open(attributions_path, 'rb') as f:
            attributions = pickle.load(f)
    else:
        # raise error
        raise FileNotFoundError(f'Attributions not found at {attributions_path}')
    
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            evaluation = pickle.load(f)
    else:
        evaluation = {}
    
    if not 'deletion curves' in evaluation:
        evaluation['deletion curves'] = {}
    if args.compute_deletion_scores:
        lrp_stft_args = {'n_fft': args.lrp_window, 'hop_length': args.lrp_hop, 'center': False}
        freqrise_stft_params = {'n_fft': 455, 'hop_length': 455-420, 'window': torch.hann_window(455, periodic = False).to(device)}
        quantiles = np.arange(0, 1, 0.05)
        for key, value in attributions.items():
            if key == 'predictions' or key == 'labels':
                continue
            if not key in evaluation['deletion curves']:
                if key == 'freqrise':
                    stft_params = freqrise_stft_params
                    cutoff = args.freqrise_cutoff
                else:
                    stft_params = lrp_stft_args
                    cutoff = None
                evaluation['deletion curves'][key] = deletion_curves(model, test_loader, value, quantiles, domain=args.explanation_domain, stft_params=stft_params, device=device, cutoff = cutoff)

        if not 'random' in evaluation['deletion curves']:
            # compute random deletion scores
            evaluation['deletion curves']['random'] = deletion_curves(model, test_loader, 'random', quantiles, domain=args.explanation_domain, stft_params=freqrise_stft_params, device = device)
            # get amplitude mask
            evaluation['deletion curves']['amplitude'] = deletion_curves(model, test_loader, 'amplitude', quantiles, domain=args.explanation_domain, stft_params=freqrise_stft_params, device = device)
    
    if not 'complexity scores' in evaluation:
        evaluation['complexity scores'] = {}
    if args.compute_complexity_scores:
        for key, value in attributions.items():
            if key in ['predictions', 'labels']:
                continue
            value = torch.cat(value).numpy()
            if key == 'freqrise':
                cutoff = args.freqrise_cutoff
                only_pos = False
            else:
                cutoff = None
                only_pos = True
            evaluation['complexity scores'][key] = np.mean(complexity_scores(value, cutoff = cutoff, only_pos = only_pos))

    if not 'localization scores' in evaluation and args.dataset == 'synthetic':
        evaluation['localization scores'] = {}
    if args.compute_localization_scores and args.dataset == 'synthetic':
        for key, value in attributions.items():
            if key in ['predictions', 'labels']:
                continue
            value = torch.cat(value).numpy()
            if key == 'freqrise':
                cutoff = args.freqrise_cutoff
                only_pos = False
            else:
                cutoff = None
                only_pos = True
            evaluation['localization scores'][key] = np.mean(localization_scores(value, attributions['labels'], cutoff = cutoff, only_pos = only_pos))
    
    with open(output_path, 'wb') as f:
        pickle.dump(evaluation, f)
    return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models', help='Path to model')
    parser.add_argument('--data_path', type = str, default = '/Users/theb/Desktop/data/AudioMNIST/', help='Path to AudioMNIST data')

    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', help='Dataset to use')
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use for AudioMNIST')
    parser.add_argument('--noise_level', type = int, default = 0.01, help='Noise level for synthetic data')
    
    parser.add_argument('--freqrise_cutoff', type = eval, default = None, help='Cutoff percent for FreqRISE during evaluation')
    
    parser.add_argument('--explanation_domain', type = str, default = 'fft', help='Domain of explanation')
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to compute attributions for')
    parser.add_argument('--lrp_window', type = int, default = 800, help='Window size for LRP')
    parser.add_argument('--lrp_hop', type = int, default = 800, help='Hop size for LRP')

    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to save output')
    parser.add_argument('--compute_deletion_scores', type = eval, default = True, help='Compute deletion scores')
    parser.add_argument('--compute_localization_scores', type = eval, default = False, help='Compute localization scores. NB only for synthetic data.')
    parser.add_argument('--compute_complexity_scores', type = eval, default = True, help='Compute complexity scores')
    args = parser.parse_args()
    main(args)