from dataset.build import build_loader
from model.TWNet import TWNet
from config.config_loader import load_config
from torch import optim as optim
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import os

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        type=str,
                        metavar='FILE',
                        help='path to config file')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'val', 'test'],
                        help='train/val/test mode')
    args = parser.parse_args()
    return args

def plot_loss_curves(train_avg_loss_list, val_avg_loss_list, save_path):
    epochs = range(1, len(train_avg_loss_list) + 1)

    # Plotting training loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_avg_loss_list, label='Training Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_loss.png'))
    plt.close()

    # Plotting validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_avg_loss_list, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'validation_loss.png'))
    plt.close()

def main():
    args = parse_option()
    config = load_config(args.cfg)
    device = torch.device('cuda')
    train_data_loader, val_data_loader = build_loader(config)

    model = TWNet()
    model.load(device=device)
    model.to(device)
    model.show_parameter_number()
    optimizer = optim.Adam(model.parameters(), eps=1e-8, lr=float(config['TRAIN']['BASE_LR']))

    train_avg_loss_list = []
    val_avg_loss_list = []


    for epoch in range(config['TRAIN']['EPOCHS']):
        model.train()
        epoch_loss = 0.0
        tqdm_train = tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{config['TRAIN']['EPOCHS']}")
        for i, gt in enumerate(tqdm_train):
            img = gt['image'].to(device, non_blocking=True)
            TWV = gt['TWV'].to(device, non_blocking=True)
            TWV = (TWV >= 0.5).float()
            optimizer.zero_grad()

            predict = model(img)
            loss = nn.BCELoss()(predict.squeeze(-1), TWV.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            tqdm_train.set_description(f"Epoch {epoch+1}/{config['TRAIN']['EPOCHS']} Batch {i+1} Loss: {loss.item():.4f}")

        average_loss = epoch_loss / len(train_data_loader)
        train_avg_loss_list.append(average_loss)
        print(f"Epoch {epoch+1} Loss: {average_loss:.4f}")

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            tqdm_val = tqdm(val_data_loader, desc="Validation")
            for i, gt in enumerate(tqdm_val):
                img = gt['image'].to(device, non_blocking=True)
                TWV = gt['TWV'].to(device, non_blocking=True)
                TWV = (TWV >= 0.5).float()
                predict = model(img)
                loss = nn.BCELoss()(predict.squeeze(-1).float(), TWV.float())
                val_loss += loss.item()
                tqdm_val.set_description(f"Validation Batch {i+1} Loss: {loss.item():.4f}")

        avg_val_loss = val_loss / len(val_data_loader)
        val_avg_loss_list.append(avg_val_loss)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        model.save(epoch, avg_val_loss)
    
    plot_loss_curves(train_avg_loss_list, val_avg_loss_list, save_path='./ckpt')

if __name__ == '__main__':
    main()