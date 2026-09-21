import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np

import argparse
import warnings
import json

from tqdm import tqdm
from models.ttvae.model import TTVAE


warnings.filterwarnings('ignore')

INFO_PATH = 'data_profile'

def train(args): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'
    num_epochs= args.epochs
    
    from src.util import ensure_data_info, locate_dataset_csv
    info = ensure_data_info(dataname)

    column_names=info['column_names']
    print(column_names)

    c_col_idx=info['cat_col_idx']
    c_col=list(np.array(column_names)[c_col_idx])

    target_col_idx=info['target_col_idx']

    if info['task_type']!="regression":
      c_col_idx=c_col_idx+target_col_idx
      c_col=list(np.array(column_names)[c_col_idx])
    
    print(c_col)

    ckpt_path = f'{curr_dir}/ckpt/{dataname}'
    info = ensure_data_info(dataname)
    csv_path = locate_dataset_csv(dataname, info)
    real = pd.read_csv(csv_path)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    model=TTVAE(verbose=True,epochs=num_epochs)
    model.fit(real,c_col,ckpt_path)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    save_path = args.save_path
    ckpt_path = f'{curr_dir}/ckpt/{dataname}'

    from src.util import ensure_data_info
    info = ensure_data_info(dataname)

    n_samples=real.shape[0]
    start_time = time.time()

    # determine device from args (when called from simple runner args may only have .gpu)
    device = f'cuda:{args.gpu}' if getattr(args, 'gpu', -1) != -1 and torch.cuda.is_available() else 'cpu'

    model_path = os.path.join(ckpt_path, 'model.pt')

    # Try safe unpickling for PyTorch >=2.6: allowlist the TTVAE class, falling back to legacy load
    try:
        from torch import serialization
        try:
            # context manager available in newer torch versions
            with serialization.safe_globals([TTVAE]):
                model = torch.load(model_path, map_location=device)
        except AttributeError:
            # safe_globals context manager not present; try global registration
            try:
                serialization.add_safe_globals([TTVAE])
            except Exception:
                pass
            model = torch.load(model_path, map_location=device)
    except Exception:
        # final fallback: try loading with weights_only=False (may be needed on torch 2.6+)
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # older torch doesn't accept weights_only keyword
            model = torch.load(model_path, map_location=device)

    # move model to device when possible
    try:
        model.to(device)
    except Exception:
        pass

    syn_df = model.sample(n_samples)

    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TTVAE')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'