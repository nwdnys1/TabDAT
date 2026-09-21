import os
import time
from numpy.lib.type_check import real_if_close

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np

import argparse
import warnings
import json
import pickle

from models.ctabgan.model import CTABGANSynthesizer
from models.ctabgan.util import DataPrep

warnings.filterwarnings('ignore')

INFO_PATH = 'data_profile'

class CTABGAN():

    def __init__(self,
                 df,
                 test_ratio = 0.20,
                 categorical_columns = [],
                 log_columns = [],
                 mixed_columns= {},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = [],
                 problem_type= {},
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=150,
                 lr=2e-4
                 ):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(
                class_dim=class_dim,
                random_dim=random_dim,
                num_channels=num_channels,
                l2scale=l2scale,
                lr=lr,
                batch_size=batch_size,
                epochs=epochs,
        )
        self.raw_df = df
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
                
    def fit(self):
        
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, num_samples):
        
        sample = self.synthesizer.sample(num_samples) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df

def train(args): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname

    num_epochs= args.epochs
    
    # ensure canonical info.json exists under data/{dataname}
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
    real_data_path = f'data/{dataname}'
    info = ensure_data_info(dataname)
    csv_path = locate_dataset_csv(dataname, info)
    real = pd.read_csv(csv_path)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # Prefer configs/features.json when present (legacy), otherwise derive
    # parameters from the canonical data profile (data_profile/{dataname}.json).
    ctabgan_params = {}
    configs_path = os.path.join(curr_dir, 'configs', 'features.json')
    if os.path.exists(configs_path):
        try:
            with open(configs_path, 'r') as f:
                configs = json.load(f)
            if isinstance(configs, dict):
                ctabgan_params = configs.get(dataname, {}) or {}
        except Exception:
            ctabgan_params = {}

    # allow per-profile overrides: add a `ctabgan_params` object in the data profile
    profile_overrides = info.get('ctabgan_params') or info.get('ctabgan') or {}
    if isinstance(profile_overrides, dict):
        ctabgan_params.update(profile_overrides)

    # build sensible defaults from info if keys are missing
    column_names = info.get('column_names', list(real.columns))
    num_idx = info.get('num_col_idx', []) or []
    cat_idx = info.get('cat_col_idx', []) or []
    target_idx = info.get('target_col_idx', []) or []

    # categorical column names
    categorical_columns = ctabgan_params.get('categorical_columns')
    if categorical_columns is None:
        # prefer the c_col (which includes the target column when the task
        # is not regression) so the target is treated as categorical and
        # label-encoded by DataPrep.
        categorical_columns = c_col if len(c_col) else []

    integer_columns = ctabgan_params.get('integer_columns')
    if integer_columns is None:
        integer_columns = list(np.array(column_names)[num_idx]) if len(num_idx) else []

    # other optional groups
    log_columns = ctabgan_params.get('log_columns', [])
    mixed_columns = ctabgan_params.get('mixed_columns', {})
    general_columns = ctabgan_params.get('general_columns', [])
    non_categorical_columns = ctabgan_params.get('non_categorical_columns', [])

    # problem_type: legacy profiles sometimes only store task; DataPrep expects
    # a dict whose one of the values is the target column name (it does
    # target_col = list(type.values())[0] and uses it as a column key).
    # Build a sensible default that includes both the task and the target
    # column name (if available).
    profile_problem = ctabgan_params.get('problem_type') or ctabgan_params.get('problem')
    # determine target column name from info (target_col_idx may be list)
    target_name = None
    try:
        t_idx = target_idx if isinstance(target_idx, (list, tuple, np.ndarray)) else [target_idx]
        if len(t_idx):
            # take first target index
            target_name = column_names[int(t_idx[0])]
    except Exception:
        target_name = None

    if isinstance(profile_problem, dict):
        # if profile provided a mapping that already includes the column name,
        # keep it; otherwise ensure a value contains the target column name.
        if any([isinstance(v, str) and v in column_names for v in profile_problem.values()]):
            problem_type = profile_problem
        else:
            # merge task (if present) and target
            problem_type = dict(profile_problem)
            if target_name is not None:
                problem_type.setdefault('target', target_name)
            else:
                problem_type.setdefault('task', info.get('task_type'))
    else:
        # default: include both target and task
        if target_name is not None:
            problem_type = {'target': target_name, 'task': info.get('task_type')}
        else:
            problem_type = {'task': info.get('task_type')}

    model_kwargs = {
        'class_dim': ctabgan_params.get('class_dim', (256, 256, 256, 256)),
        'random_dim': ctabgan_params.get('random_dim', 100),
        'num_channels': ctabgan_params.get('num_channels', 64),
        'l2scale': ctabgan_params.get('l2scale', 1e-5),
        'batch_size': ctabgan_params.get('batch_size', 500),
        'epochs': num_epochs,
        'lr': ctabgan_params.get('lr', 2e-4),
    }

    synthesizer = CTABGAN(
        df=real,
        test_ratio=0.2,
        categorical_columns=categorical_columns,
        log_columns=log_columns,
        mixed_columns=mixed_columns,
        general_columns=general_columns,
        non_categorical_columns=non_categorical_columns,
        integer_columns=integer_columns,
        problem_type=problem_type,
        **model_kwargs,
    )
        
    synthesizer.fit()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    save_path = args.save_path
    ckpt_path = f'{curr_dir}/ckpt/{dataname}'
    # Use canonical profile and dataset CSV to get dataset info.
    from src.util import ensure_data_info, locate_dataset_csv
    info = ensure_data_info(dataname)
    n_samples = real.shape[0]

    start_time = time.time()

    syn_df=synthesizer.generate_samples(n_samples)
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