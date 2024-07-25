from torch.utils.data import DataLoader
import torch
from src.models.audiomnist import AudioNet
import numpy as np
from src.evaluation import mask_and_predict
from src.freqrise import compute_freqrise_scores
from src.lrp import compute_gradient_scores
from src.data.dataloader import AudioNetDataset
import pickle
import os
import argparse

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dset = AudioNetDataset(args.data_path, True, 'test', splits = [0], labeltype = args.labeltype, subsample = args.n_samples, seed = 0)                     
    test_dloader = DataLoader(test_dset, batch_size=10, shuffle=False)

    if args.labeltype == 'gender':
        model = AudioNet(input_shape=(1, 8000), num_classes=2).eval()
        model.load_state_dict(torch.load('models/AudioNet_gender.pt'))
    else:
        model = AudioNet(input_shape=(1, 8000), num_classes=10).eval()
        model.load_state_dict(torch.load('models/AudioNet_digit.pt'))

    # get LRP explanations
    # compute saliency
    lrp_stft_args = {'n_fft': args.lrp_window, 'hop_length': args.lrp_hop, 'center': False}
    if args.compute_grad_scores:
        grad_path = f'outputs/gradient_scores_{args.explanation_domain}_{args.labeltype}_{args.lrp_window}_{args.lrp_hop}.pkl'
        # check if path exists
        if os.path.exists(grad_path):
            with open(grad_path, 'rb') as f:
                grad_scores = pickle.load(f)
            saliency_map = grad_scores['saliency']
            lrp_map = grad_scores['lrp']
            IG_map = grad_scores['IG']
        else:
            saliency_map = compute_gradient_scores(model, test_dloader, attr_method = 'gxi', domain = args.explanation_domain, stft_params = lrp_stft_args)
            # compute LRP
            lrp_map = compute_gradient_scores(model, test_dloader, attr_method = 'lrp', domain = args.explanation_domain, stft_params = lrp_stft_args)
            # compute integrated gradients
            IG_map = compute_gradient_scores(model, test_dloader, attr_method = 'ig', domain = args.explanation_domain, stft_params = lrp_stft_args)
            grad_scores = {
                'saliency': saliency_map,
                'lrp': lrp_map,
                'IG': IG_map
            }
            with open(grad_path, 'wb') as f:
                pickle.dump(grad_scores, f)
        print("Saliency done")
    else:
        saliency_map = None
        lrp_map = None
        IG_map = None

    model.to(device)
    if args.labeltype == 'gender' and args.rise_softmax:
        exp_label = 1
    else:
        exp_label = None
    freqrise_stft_params = {'n_fft': 455, 'hop_length': 455-420, 'window': torch.hann_window(455, periodic = False).to(device)}
    if args.compute_freqrise_scores:
        rise_path = f'outputs/relax_scores_{args.explanation_domain}_{args.labeltype}_{args.num_cells}_{args.n_samples}_{args.rise_relax}_softmax_{args.rise_softmax}_iter_{args.freqrise_samples}_dropprob_{args.probability_of_drop}.pkl'
        if os.path.exists(rise_path):
            with open(rise_path, 'rb') as f:
                freqrise_attributions = pickle.load(f)
        else:
            freqrise_attributions = compute_freqrise_scores(model, 
                                                    test_dloader, 
                                                    exp_label=exp_label, 
                                                    n_samples = args.freqrise_samples, 
                                                    num_cells = args.num_cells, 
                                                    probability_of_drop = args.probability_of_drop, 
                                                    domain = args.explanation_domain,
                                                    use_softmax=args.rise_softmax,
                                                    stft_params = freqrise_stft_params,
                                                    device = device)
            with open(rise_path, 'wb') as f:
                pickle.dump(freqrise_attributions, f)
        print("FreqRISE done")
    else:
        freqrise_attributions = None
    
    attributions = {
        'lrp': lrp_map,
        'freqrise': freqrise_attributions,
        'saliency': saliency_map,
        'IG': IG_map
    }
    # get predictions and labels
    predictions = []
    labels = []
    for data, target in test_dloader:
        data = data.to(device)
        output = model(data.float())
        predictions.append(output.detach().cpu())
        labels.append(target)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    sampling_percent = np.arange(0, 1, 0.05)

    accuracies = {
        'lrp': [],
        'freqrise': [],
        'random': [],
        'amplitude': [],
        'saliency': [],
        'IG': []
    }

    if args.compute_accuracy_scores:
        for percent in sampling_percent:
            print(f'Percent: {percent}')
            quantile = 1-percent
            # get random mask
            random_acc = mask_and_predict(model, test_dloader, 'random', quantile, domain=args.explanation_domain, stft_params=freqrise_stft_params, device = device)
            accuracies['random'].append(random_acc)

            # get amplitude mask
            amplitude_acc = mask_and_predict(model, test_dloader, 'amplitude', quantile, domain=args.explanation_domain, stft_params=freqrise_stft_params, device = device)
            accuracies['amplitude'].append(amplitude_acc)

            # get lrp mask
            if lrp_map is not None:
                lrp_acc = mask_and_predict(model, test_dloader, lrp_map, quantile, domain=args.explanation_domain, stft_params=lrp_stft_args, device=device)
                accuracies['lrp'].append(lrp_acc)

            # get freqrise mask
            freqrise_acc = mask_and_predict(model, test_dloader, freqrise_attributions, quantile, domain=args.explanation_domain, stft_params=freqrise_stft_params, device=device)
            accuracies['freqrise'].append(freqrise_acc)

            # get saliency mask
            if saliency_map is not None:
                saliency_acc = mask_and_predict(model, test_dloader, saliency_map, quantile, domain=args.explanation_domain, stft_params=lrp_stft_args, device=device)
                accuracies['saliency'].append(saliency_acc)

            # get IG mask
            if IG_map is not None:
                IG_acc = mask_and_predict(model, test_dloader, IG_map, quantile, domain=args.explanation_domain, stft_params=lrp_stft_args, device=device)
                accuracies['IG'].append(IG_acc)

        all_res = {
            'accuracies': accuracies,
            'attributions': attributions,
            'predictions': predictions,
            'labels': labels
        }
        with open(f'outputs/accuracies_{args.explanation_domain}_{args.labeltype}_{args.num_cells}_{args.n_samples}_softmax_{args.rise_softmax}_iter_{args.freqrise_samples}_dropprob_{args.probability_of_drop}.pkl', 'wb') as f:
            pickle.dump(all_res, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic data explainability')
    parser.add_argument('--data_path', type = str, default = '/Users/theb/Documents/PhD/code/AudioMNIST/', help='Path to AudioMNIST data')
    parser.add_argument('--compute_grad_scores', type = eval, default = True, help='Compute gradient scores')
    parser.add_argument('--compute_freqrise_scores', type = eval, default = False, help='Compute FreqRISE scores')
    parser.add_argument('--compute_accuracy_scores', type = eval, default = True, help='Compute accuracy scores')
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Type of label to use')
    parser.add_argument('--rise_softmax', type = eval, default = True, help='Use softmax in RISE')
    parser.add_argument('--num_cells', type = int, default = 200, help='Number of cells in mask')
    parser.add_argument('--probability_of_drop', type = float, default = 0.5, help='Probability of dropping')
    parser.add_argument('--n_samples', type = int, default = 1, help='Number of samples to compute attributions for')
    parser.add_argument('--freqrise_samples', type = int, default = 3000, help='Number of samples to use to compute FreqRISE')
    parser.add_argument('--lrp_window', type = int, default = 800, help='Window size for LRP')
    parser.add_argument('--lrp_hop', type = int, default = 800, help='Hop size for LRP')
    parser.add_argument('--explanation_domain', type = str, default = 'stft', help='Domain of explanation')
    args = parser.parse_args()
    main(args)