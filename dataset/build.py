import numpy as np
import torch.utils.data
from dataset.zind_dataset import ZindDataset


def build_loader(config):
    name = config['DATA']['DATASET']
    train_dataset = None
    train_data_loader = None
    train_dataset = build_dataset(mode='train', config=config)

    val_dataset = build_dataset(mode='val', config=config)

    train_sampler = None
    val_sampler = None

    batch_size = config['DATA']['BATCH_SIZE']
    num_workers = config['DATA']['NUM_WORKERS']
    if train_dataset:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
    batch_size = batch_size - (len(val_dataset) % np.arange(batch_size, 0, -1)).tolist().index(0)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    return train_data_loader, val_data_loader


def build_dataset(mode, config):
    name = config['DATA']['DATASET']
    if name == 'zind':
        dataset = ZindDataset(
            root_dir=config['DATA']['DIR'],
            mode=mode,
            shape=config['DATA']['SHAPE'] if 'SHAPE' in config['DATA'] else None,
            aug=config['DATA']['AUG'] if mode == 'train' else None,
            is_simple=True,
            is_ceiling_flat=False,
            vp_align=config['EVAL']['VP']
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {name}")

    return dataset
