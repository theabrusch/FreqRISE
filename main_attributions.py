import torch
import argparse
from src.explainability.freqrise import FreqRISE
from src.explainability.evaluation import compute_gradient_scores
from src.data.load_data import load_data
from src.models.load_model import load_model
import pickle
import os


def main(args):
    test_loader = load_data(args)
    model = load_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'synthetic':
        output_path = f'{args.output_path}/{args.dataset}_{args.noise_level}_attributions_{args.explanation_domain}_{args.n_samples}.pkl'
    elif args.dataset == 'AudioMNIST':
        output_path = f'{args.output_path}/{args.dataset}_{args.labeltype}_attributions_{args.explanation_domain}_{args.n_samples}.pkl'

    # check if attributions are already computed
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            attributions = pickle.load(f)
    else:
        attributions = {}

    ## Compute all attributions
    lrp_stft_args = {'n_fft': args.lrp_window, 'hop_length': args.lrp_hop, 'center': False}
    if not 'saliency' in attributions:
        # compute saliency
        attributions['saliency'] = compute_gradient_scores(model, test_loader, attr_method = 'gxi', domain = args.explanation_domain, stft_params = lrp_stft_args)
    if not 'lrp' in attributions:
        # compute LRP
        attributions['lrp'] = compute_gradient_scores(model, test_loader, attr_method = 'lrp', domain = args.explanation_domain, stft_params = lrp_stft_args)
    if not 'IG' in attributions:
        # compute integrated gradients
        attributions['IG'] = compute_gradient_scores(model, test_loader, attr_method = 'ig', domain = args.explanation_domain, stft_params = lrp_stft_args)
    model.to(device)

    freqrise_stft_params = {'n_fft': 455, 'hop_length': 455-420, 'window': torch.hann_window(455, periodic = False).to(device)}
    if not f'freqrise_{args.num_cells}_{args.freqrise_samples}_dropprob_{args.probability_of_drop}' in attributions:
        # compute FreqRISE
        freqrise = FreqRISE(model, batch_size=500, num_batches=args.freqrise_samples//500, device=device, domain=args.explanation_domain, stft_params=freqrise_stft_params, use_softmax=False)
        attributions[f'freqrise_{args.num_cells}_{args.freqrise_samples}_dropprob_{args.probability_of_drop}'] = freqrise.forward_dataloader(test_loader, args.num_cells, args.probability_of_drop)

    if not 'predictions' in attributions:
        # get predictions and labels
        predictions = []
        labels = []
        for data, target in test_loader:
            data = data.to(device)
            output = model(data.float())
            predictions.append(output.detach().cpu())
            labels.append(target)
        attributions['predictions'] = torch.cat(predictions, dim=0)
        attributions['labels'] = torch.cat(labels, dim=0)

    with open(output_path, 'wb') as f:
        pickle.dump(attributions, f)
    
    return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = '/Users/theb/Desktop/data/AudioMNIST/', help='Path to AudioMNIST data')
    parser.add_argument('--model_path', type = str, default = 'models', help='Path to models folder')

    parser.add_argument('--dataset', type = str, default = 'AudioMNIST', help='Dataset to use')
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use for AudioMNIST')
    parser.add_argument('--noise_level', type = float, default = 0.01, help='Noise level for synthetic dataset. Either 0.8 or 0.01.')
    
    parser.add_argument('--explanation_domain', type = str, default = 'fft', help='Domain of explanation')
    parser.add_argument('--num_cells', type = int, default = 200, help='Number of cells in mask')
    parser.add_argument('--probability_of_drop', type = float, default = 0.5, help='Probability of dropping')
    parser.add_argument('--n_samples', type = int, default = 10, help='Number of samples to compute attributions for')
    parser.add_argument('--freqrise_samples', type = int, default = 3000, help='Number of samples to use to compute FreqRISE')
    parser.add_argument('--lrp_window', type = int, default = 800, help='Window size for LRP')
    parser.add_argument('--lrp_hop', type = int, default = 800, help='Hop size for LRP')

    parser.add_argument('--output_path', type = str, default = 'outputs', help='Path to save output')
    args = parser.parse_args()
    main(args)