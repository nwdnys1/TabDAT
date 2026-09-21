import numpy as np
import os
import pandas as pd
import src
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        if self.X_num is None:
            return self.X_cat[index]
        if self.X_cat is None:
            return self.X_num[index]
        
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        if self.X_cat is not None:
            return self.X_cat.shape[0]
        else:
            return self.X_num.shape[0]


def preprocess(
    raw_config,
    dataset_path,
    task_type="binclass",
    inverse=False,
    cat_encoding=None,
    concat=True,
):

    T_dict = {}

    T_dict["normalization"] = "quantile"
    T_dict["num_nan_policy"] = "mean"
    T_dict["cat_nan_policy"] = None
    T_dict["cat_min_frequency"] = None
    T_dict["cat_encoding"] = cat_encoding
    T_dict["y_policy"] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        raw_config=raw_config,
        n_classes=raw_config['num_classes'],
        data_path=dataset_path,
        T=T,
        task_type=task_type,
        change_val=False,
        concat=concat,
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        if X_num is not None:   
            X_train_num, X_test_num = X_num["train"], X_num["test"]
            d_numerical = X_train_num.shape[1]
            X_num = (X_train_num, X_test_num)
        else:
            d_numerical = 0
        if X_cat is not None:
            X_train_cat, X_test_cat = X_cat["train"], X_cat["test"]
            categories = src.get_categories(X_train_cat)        
            X_cat = (X_train_cat, X_test_cat)
        else:
            categories = []    
        

        if inverse:
            if X_num is not None:
                num_inverse = dataset.num_transform.inverse_transform
            else:
                num_inverse = None
            if X_cat is not None:
                cat_inverse = dataset.cat_transform.inverse_transform
            else:
                cat_inverse = None

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    raw_config: dict,
    data_path: str,
    T: src.Transformations,
    task_type,
    n_classes,
    change_val: bool,
    concat=True,
):
    num_cols = raw_config["num_col_idx"]
    cat_cols = raw_config["cat_col_idx"]
    target_col = raw_config["target_col_idx"]

    X = pd.read_csv(data_path).dropna().values
    X_cat = {}
    X_num = {}
    y = {}
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_size = int(0.7 * X.shape[0])
    train_indices = indices[:]
    test_indices = indices[train_size:]
    # classification
    if task_type == "binclass" or task_type == "multiclass":
        cat_cols = target_col + cat_cols if concat else cat_cols
        X_cat["train"] = X[train_indices][:, cat_cols]
        X_cat["test"] = X[test_indices][:, cat_cols]
        X_num["train"] = X[train_indices][:, num_cols].astype(np.float32)
        X_num["test"] = X[test_indices][:, num_cols].astype(np.float32)
        y["train"] = X[train_indices][:, target_col]
        y["test"] = X[test_indices][:, target_col]

    else:
        # regression
        num_cols = target_col + num_cols if concat else num_cols
        X_cat["train"] = X[train_indices][:, cat_cols]
        X_cat["test"] = X[test_indices][:, cat_cols]
        X_num["train"] = X[train_indices][:, num_cols].astype(np.float32)
        X_num["test"] = X[test_indices][:, num_cols].astype(np.float32)
        y["train"] = X[train_indices][:, target_col]
        y["test"] = X[test_indices][:, target_col]

    if len(cat_cols) == 0:
        X_cat = None
    if len(num_cols) == 0:
        X_num = None

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(task_type),
        n_classes=n_classes,
    )

    if change_val:
        D = src.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return src.transform_dataset(D, T, None)
