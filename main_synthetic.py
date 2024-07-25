import torch
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from src.data.generators import frequency_lrp_dataset
from src.models.synth_mlp import LinearModel, LightningModel
from src.freqrise import FreqRISE
from src.freqrise import mask_generator  
from src.lrp import compute_gradient_scores

def main(args):
    model = LinearModel(2560, hidden_layers = 2, hidden_size = 64, output_size = 16)
    lightning_model = LightningModel(model, 'cross_entropy', 0.001)
    logger = WandbLogger(name = 'synth_mlp', project = 'explainability')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = L.Trainer(max_epochs=args.epochs, accelerator = device, logger = logger)
    model_path = f'models/synth_mlp_{args.noise_level}_Mmin_{args.M_min}_Mmax_{args.M_max}.pth'
    # check if model is saved, othwerwise train
    if not os.path.exists(model_path):
        train_data, train_labels = frequency_lrp_dataset(10**5, length = 2560, noiselevel = args.noise_level, M_min = args.M_min, M_max = args.M_max)
        train_loader = DataLoader(TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels).long()), 
                            batch_size = 256, shuffle = True, num_workers = 9, persistent_workers=True)
        val_data, val_labels = frequency_lrp_dataset(10**3, length = 2560, noiselevel = 0.8, integer_freqs=True, M_min = args.M_min, M_max = args.M_max)
        val_loader = DataLoader(TensorDataset(torch.tensor(val_data).float(), torch.tensor(val_labels)), batch_size=1000)
        trainer.fit(lightning_model, train_loader, val_loader)
        torch.save(model.state_dict(), model_path)
        # delete train data
        del train_data, train_labels, train_loader
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print('Model loaded')
        # save model

    test_data, test_labels = frequency_lrp_dataset(10**3, length = 2560, noiselevel = 0., M_min = args.M_min, M_max = args.M_max)
    test_data, test_labels = torch.tensor(test_data).float(), torch.tensor(test_labels).long()
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=256)

    test_acc = lightning_model.test_model(test_loader, device = device)
    print(f'Accuracy of the network on the 1000 test samples: {test_acc*100}%')

    # compute saliency
    saliency_map = compute_gradient_scores(model, test_loader, attr_method = 'gxi')
    # compute LRP
    lrp_map = compute_gradient_scores(model, test_loader, attr_method = 'lrp')
    # compute integrated gradients
    IG_map = compute_gradient_scores(model, test_loader, attr_method = 'ig')

    # compute FreqRISE
    rise_map = []
    for sample, label in zip(test_data, test_labels):
        rise_fft = FreqRISE(torch.tensor(sample).float(), model, batch_size=30, num_batches=100, device=device)
        with torch.no_grad(): 
            rise_fft.forward(mask_generator = mask_generator, num_cells = 1281, probablity_of_drop = 0.5)
        rise_fft_importance = rise_fft.importance.cpu().squeeze()[:,label]
        rise_map.append(rise_fft_importance)
    rise_map = torch.stack(rise_map)

    # save all attributions in dictionary and dump
    attributions = {'data': test_data, 'labels': test_labels, 'saliency': saliency_map, 'lrp': lrp_map, 'rise_fft': rise_map, 'ig_map': IG_map}
    torch.save(attributions, f'outputs/attributions_{args.noise_level}_Mmin_{args.M_min}_Mmax_{args.M_max}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic data explainability')
    parser.add_argument('--noise_level', type=float, default=0.8, help='Noise level of the synthetic data')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--M_min', type=int, default=10, help='Number of frequency components')
    parser.add_argument('--M_max', type=int, default=50, help='Number of frequency components')
    args = parser.parse_args()
    main(args)



