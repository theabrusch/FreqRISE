import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import glob
from src.models.audiomnist import AudioNet, EarlyStopping
from src.data.dataloader import AudioNetDataset
import argparse
import os


def main(args):
    dset = AudioNetDataset(args.data_path, True, 'train', splits = [0], labeltype = args.labeltype)
    val_dset = AudioNetDataset(args.data_path, True, 'validate', splits = [0], labeltype = args.labeltype)
    test_dset = AudioNetDataset(args.data_path, True, 'test', splits = [0], labeltype = args.labeltype)                     
    test_dloader = DataLoader(test_dset, batch_size=100, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    dloader = DataLoader(dset, batch_size=100, shuffle=True)
    val_dloader = DataLoader(val_dset, batch_size=100, shuffle=True)

    epochs = 50
    collect_loss = []
    loss_fn = torch.nn.CrossEntropyLoss()
    if args.labeltype == 'digit':
        model =  AudioNet(input_shape=(1, 8000), num_classes=10).to(device)
    else:
        model =  AudioNet(input_shape=(1, 8000), num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    model.train()
    earlystopping = EarlyStopping(patience = 7, verbose = True, path = f'{args.model_path}/checkpoint.pt')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dloader:
            optimizer.zero_grad()
            output = model(batch[0].float().to(device))
            loss = loss_fn(output, batch[1].long().to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} Loss: {epoch_loss/len(dloader)}")
        collect_loss.append(epoch_loss/len(dloader))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in val_dloader:
                output = model(batch[0].float().to(device))
                _, predicted = torch.max(output, 1)
                total += batch[1].size(0)
                correct += (predicted == batch[1].to(device)).sum().item()
            print(f"Validation Accuracy: {100 * correct / total}")
        earlystopping(-correct / total, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break
    best_model = model
    best_model.load_state_dict(torch.load(f'{args.model_path}/checkpoint.pt'))

    best_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_dloader:
            output = best_model(batch[0].float().to(device))
            _, predicted = torch.max(output, 1)
            total += batch[1].size(0)
            correct += (predicted == batch[1].to(device)).sum().item()
        print(f"Test Accuracy: {100 * correct / total}")

    torch.save(best_model.cpu().state_dict(), f'{args.model_path}/AudioNet_{args.labeltype}.pt')
    # delete checkpoint
    for f in glob.glob('checkpoint.pt'):
        os.remove(f)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models', help='Path to save model')
    parser.add_argument('--data_path', type = str, default = '/Users/theb/Desktop/data/AudioMNIST/', help='Path to AudioMNIST data')
    parser.add_argument('--labeltype', type = str, default = 'digit', help='Label type to use')

    args = parser.parse_args()
    main(args)