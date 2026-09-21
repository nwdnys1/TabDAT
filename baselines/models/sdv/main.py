import os
from ctgan import CTGAN,TVAE
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer

import torch
import os
import time
import argparse
import warnings
import pandas as pd
import numpy as np

import torch

warnings.filterwarnings('ignore')


def train_ctgan(args):

    num_epochs = getattr(args, 'epochs', 1) + 1
    dataname = args.dataname
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    from src.util import ensure_data_info, locate_dataset_csv

    info = ensure_data_info(dataname)
    csv_path = locate_dataset_csv(dataname, info)
    real = pd.read_csv(csv_path)
    real.dropna(inplace=True)

    column_names = info.get('column_names', list(real.columns))
    c_col_idx = info.get('cat_col_idx', [])
    c_col = list(np.array(column_names)[c_col_idx]) if len(c_col_idx) else []
    target_col_idx = info.get('target_col_idx', [])
    if info.get('task_type') != 'regression':
        c_col_idx = c_col_idx + target_col_idx
        c_col = list(np.array(column_names)[c_col_idx])

    ckpt_path = os.path.join(curr_dir, 'ckpt/ctgan', dataname)
    os.makedirs(ckpt_path, exist_ok=True)

    start_time = time.time()
    ctgan = CTGAN(verbose=True, epochs=num_epochs)
    ctgan.fit(real, c_col)

    synthetic = ctgan.sample(real.shape[0])
    synthetic.to_csv(save_path, index=False)

    end_time = time.time()
    print('Time: ', end_time - start_time)
    print('Saving sampled data to {}'.format(save_path))


def train_tvae(args):
    num_epochs = getattr(args, 'epochs', 1) + 1
    dataname = args.dataname
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    from src.util import ensure_data_info, locate_dataset_csv

    info = ensure_data_info(dataname)
    csv_path = locate_dataset_csv(dataname, info)
    real = pd.read_csv(csv_path)

    column_names = info.get('column_names', list(real.columns))
    c_col_idx = info.get('cat_col_idx', [])
    c_col = list(np.array(column_names)[c_col_idx]) if len(c_col_idx) else []
    target_col_idx = info.get('target_col_idx', [])
    if info.get('task_type') != 'regression':
        c_col_idx = c_col_idx + target_col_idx
        c_col = list(np.array(column_names)[c_col_idx])

    ckpt_path = os.path.join(curr_dir, 'ckpt/tvae', dataname)
    os.makedirs(ckpt_path, exist_ok=True)

    start_time = time.time()
    tvae = TVAE(epochs=num_epochs)
    tvae.fit(real, c_col)

    synthetic = tvae.sample(real.shape[0])
    synthetic.to_csv(save_path, index=False)

    end_time = time.time()
    print('Time: ', end_time - start_time)
    print('Saving sampled data to {}'.format(save_path))


def train_copulagan(args):
    device = getattr(args, 'device', 'cpu')
    num_epochs = getattr(args, 'epochs', 1) + 1
    dataname = args.dataname
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    from src.util import ensure_data_info, locate_dataset_csv

    info = ensure_data_info(dataname)
    csv_path = locate_dataset_csv(dataname, info)
    real = pd.read_csv(csv_path)

    ckpt_path = os.path.join(curr_dir, 'ckpt/copulagan', dataname)
    os.makedirs(ckpt_path, exist_ok=True)

    start_time = time.time()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real)

    copulagan = CopulaGANSynthesizer(metadata, epochs=num_epochs)
    copulagan.fit(real)
    copulagan.save(os.path.join(ckpt_path, 'CopulaGAN.pkl'))

    copulagan_load = CopulaGANSynthesizer.load(os.path.join(ckpt_path, 'CopulaGAN.pkl'))
    synthetic = copulagan_load.sample(real.shape[0])
    synthetic.to_csv(save_path, index=False)

    end_time = time.time()
    print('Time: ', end_time - start_time)
    print('Saving sampled data to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of SDV wrappers')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--sdv_epochs', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='synthetic.csv')

    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'