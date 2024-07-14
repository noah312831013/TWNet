from dataset.build import build_loader
from model.TWNet import TWNet
from config.config_loader import load_config
from torch import optim as optim
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter

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

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/TWNet_training')  # Specify the log directory

    for epoch in range(config['TRAIN']['EPOCHS']):
        # train
        model.train()  # Set model to training mode
        optimizer.zero_grad()
        epoch_loss = 0.0  # Initialize epoch loss
        for i, gt in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{config['TRAIN']['EPOCHS']}")):
            img = gt['image'].to(device,non_blocking=True)
            TWV = gt['TWV'].to(device,non_blocking=True)
            predict = model(img)
            loss = nn.L1Loss()(predict, TWV)  # Calculate L1 loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # Accumulate loss for the epoch
            print(f"Batch {i+1} Loss: {loss.item():.4f}")

            # Log training loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_data_loader) + i)

        # Display average loss for the epoch
        average_loss = epoch_loss / len(train_data_loader)
        print(f"Epoch {epoch+1} Loss: {average_loss:.4f}")

        # Log epoch loss to TensorBoard
        writer.add_scalar('Loss/epoch', average_loss, epoch)

        val_loss = 0.0
        torch.no_grad()
        model.eval()
        for i, gt in enumerate(tqdm(val_data_loader)):
            img = gt['image'].to(device,non_blocking=True)
            TWV = gt['TWV'].to(device,non_blocking=True)
            predict = model(img)
            loss = nn.L1Loss()(predict, TWV)  # Calculate L1 loss
            val_loss += loss.item()  # Accumulate loss for the epoch
            print(f"Batch {i+1} Loss: {loss.item():.4f}")

            # Log validation loss to TensorBoard
            writer.add_scalar('Loss/val', loss.item(), epoch * len(val_data_loader) + i)

        model.save(epoch,val_loss/len(val_data_loader))

    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
