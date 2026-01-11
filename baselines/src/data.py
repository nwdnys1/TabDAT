import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from importlib.resources import path
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
import torch
import os
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from . import env, util
from .metrics import calculate_metrics as calculate_metrics_
from .util import TaskType

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_MISSING_VALUE = 'nan'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> 'Dataset':
        # Construct Dataset from the canonical CSV loaded into memory.
        # This avoids any reads of data/{dataname}/info.json or .npy files.
        dir_ = Path(dir_)
        dataset_name = os.path.basename(os.path.normpath(dir_))

        # read_pure_data will load datasets/{dataset}.csv once into an in-memory
        # DataFrame and cache it. It returns arrays for the requested split and
        # does not touch any auxiliary files.
        X_num_train, X_cat_train, y_train = read_pure_data(dir_, 'train')
        # attempt to read 'val' and 'test' splits (will return empties if not present)
        X_num_val, X_cat_val, y_val = read_pure_data(dir_, 'val')
        X_num_test, X_cat_test, y_test = read_pure_data(dir_, 'test')

        X_num: Optional[ArrayDict] = None
        X_cat: Optional[ArrayDict] = None
        y: ArrayDict = {}

        if X_num_train is not None or X_num_val is not None or X_num_test is not None:
            X_num = {}
            if X_num_train is not None:
                X_num['train'] = X_num_train
            if X_num_val is not None:
                X_num['val'] = X_num_val
            if X_num_test is not None:
                X_num['test'] = X_num_test

        if X_cat_train is not None or X_cat_val is not None or X_cat_test is not None:
            X_cat = {}
            if X_cat_train is not None:
                X_cat['train'] = X_cat_train
            if X_cat_val is not None:
                X_cat['val'] = X_cat_val
            if X_cat_test is not None:
                X_cat['test'] = X_cat_test

        if y_train is not None:
            y['train'] = y_train
        if y_val is not None and y_val.size > 0:
            y['val'] = y_val
        if y_test is not None and y_test.size > 0:
            y['test'] = y_test

        try:
            info = util.ensure_data_info(dataset_name)
            task_type = TaskType(info.get('task_type'))
            n_classes = info.get('n_classes')
        except Exception:
            # fallback conservative defaults
            info = {}
            task_type = TaskType.BINCLASS
            n_classes = None

        return Dataset(X_num, X_cat, y, {}, task_type, n_classes)

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = 'rmse'
            score_sign = -1
        else:
            score_key = 'accuracy'
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics['score'] = score_sign * part_metrics[score_key]
        return metrics

def change_val(dataset: Dataset, val_size: float = 0.2):
    # should be done before transformations

    y = np.concatenate([dataset.y['train'], dataset.y['val']], axis=0)

    ixs = np.arange(y.shape[0])
    if dataset.is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)

    dataset.y['train'] = y[train_ixs]
    dataset.y['val'] = y[val_ixs]

    if dataset.X_num is not None:
        X_num = np.concatenate([dataset.X_num['train'], dataset.X_num['val']], axis=0)
        dataset.X_num['train'] = X_num[train_ixs]
        dataset.X_num['val'] = X_num[val_ixs]

    if dataset.X_cat is not None:
        X_cat = np.concatenate([dataset.X_cat['train'], dataset.X_cat['val']], axis=0)
        dataset.X_cat['train'] = X_cat[train_ixs]
        dataset.X_cat['val'] = X_cat[val_ixs]

    return dataset

def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:

    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        # assert policy is None
        print('No NaNs in numerical features, skipping')
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert util.raise_unknown('policy', policy)
    return dataset


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: ArrayDict, normalization: Normalization, seed: Optional[int], return_normalizer : bool = False
) -> ArrayDict:
    X_train = X['train']

    # If train split is empty, skip normalization entirely and return X as-is.
    if X_train is None or X_train.shape[0] == 0:
        if return_normalizer:
            return X, None
        return X

    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        # n_quantiles must be <= n_samples and >= 1. Choose a safe value based on n_samples.
        n_samples = max(1, X_train.shape[0])
        n_quantiles = min(n_samples, 1000)
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=n_quantiles,
            subsample=int(1e9),
            random_state=seed,
        )
        # noise = 1e-3
        # if noise > 0:
        #     assert seed is not None
        #     stds = np.std(X_train, axis=0, keepdims=True)
        #     noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
        #     X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
        #         X_train.shape
        #     )
    else:
        util.raise_unknown('normalization', normalization)

    normalizer.fit(X_train)

    def _transform_if_nonempty(v: np.ndarray) -> np.ndarray:
        # If v has zero rows, skip transform (transform would raise).
        if v is None:
            return v
        if v.shape[0] == 0:
            return v
        return normalizer.transform(v)

    transformed = {k: _transform_if_nonempty(v) for k, v in X.items()}
    if return_normalizer:
        return transformed, normalizer
    return transformed


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    # allow the config sentinel "__none__" (used in train configs) to mean None
    if policy == "__none__":
        policy = None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            util.raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
    return_encoder : bool = False
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # If train split is empty or None, skip categorical encoding entirely.
    # Several sklearn encoders require at least one sample to fit/transform.
    if X is None or X.get('train') is None or X['train'].shape[0] == 0:
        if return_encoder:
            return X, False, None
        return X, False

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X['train'])
        encoder = make_pipeline(oe)
        encoder.fit(X['train'])

        def _transform_if_nonempty(v: np.ndarray) -> np.ndarray:
            if v is None:
                return v
            if v.shape[0] == 0:
                return v
            return encoder.transform(v)

        X = {k: _transform_if_nonempty(v) for k, v in X.items()}
        max_values = X['train'].max(axis=0)
        for part in X.keys():
            if part == 'train': continue
            if X[part] is None or X[part].shape[0] == 0:
                continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode.

    elif encoding == 'one-hot':
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, dtype=np.float32 # type: ignore[code]
        )
        encoder = make_pipeline(ohe)
        encoder.fit(X['train'])

        def _transform_if_nonempty(v: np.ndarray) -> np.ndarray:
            if v is None:
                return v
            if v.shape[0] == 0:
                return v
            return encoder.transform(v)

        X = {k: _transform_if_nonempty(v) for k, v in X.items()}

    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(('loe', loe))
        encoder.fit(X['train'], y_train)

        def _transform_if_nonempty(v: np.ndarray) -> np.ndarray:
            if v is None:
                return v
            if v.shape[0] == 0:
                # LeaveOneOutEncoder returns a DataFrame; mimic empty float32 array
                return np.zeros((0, 0), dtype='float32')
            return encoder.transform(v).astype('float32')  # type: ignore[code]

        X = {k: _transform_if_nonempty(v) for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
    else:
        util.raise_unknown('encoding', encoding)
    
    if return_encoder:
        return X, True, encoder # type: ignore[code]
    return (X, True)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        util.raise_unknown('policy', policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    return_transforms: bool = False
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True
        )
        num_transform = num_transform
    
    # Coerce transformation parameters that may come from config files as strings.
    cat_nan_policy = transformations.cat_nan_policy
    cat_min_frequency = transformations.cat_min_frequency
    cat_encoding = transformations.cat_encoding

    if isinstance(cat_nan_policy, str) and cat_nan_policy == "__none__":
        cat_nan_policy = None
    if isinstance(cat_encoding, str) and cat_encoding == "__none__":
        cat_encoding = None
    if isinstance(cat_min_frequency, str):
        if cat_min_frequency == "__none__":
            cat_min_frequency = None
        else:
            try:
                cat_min_frequency = float(cat_min_frequency)
            except Exception:
                raise ValueError(f"Cannot parse cat_min_frequency={transformations.cat_min_frequency}")

    if dataset.X_cat is None:
        assert cat_nan_policy is None
        assert cat_min_frequency is None
        # assert cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, cat_nan_policy)

        if cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, cat_min_frequency)
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            cat_encoding,
            dataset.y['train'],
            transformations.seed,
            return_encoder=True
        )

        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

        
    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        util.dump_pickle((transformations, dataset), cache_path)
    # if return_transforms:
        # return dataset, num_transform, cat_transform
    return dataset


def build_dataset(
    path: Union[str, Path],
    transformations: Transformations,
    cache: bool
) -> Dataset:
    path = Path(path)
    dataset = Dataset.from_dir(path)
    return transform_dataset(dataset, transformations, path if cache else None)


def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y

###############
## DataLoader##
###############

class TabDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset : Dataset, split : Literal['train', 'val', 'test']
    ):
        super().__init__()
        
        self.X_num = torch.from_numpy(dataset.X_num[split]) if dataset.X_num is not None else None
        self.X_cat = torch.from_numpy(dataset.X_cat[split]) if dataset.X_cat is not None else None
        self.y = torch.from_numpy(dataset.y[split])

        assert self.y is not None
        assert self.X_num is not None or self.X_cat is not None 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        out_dict = {
            'y': self.y[idx].long() if self.y is not None else None,
        }

        x = np.empty((0,))
        if self.X_num is not None:
            x = self.X_num[idx]
        if self.X_cat is not None:
            x = torch.cat([x, self.X_cat[idx]], dim=0)
        return x.float(), out_dict

def prepare_dataloader(
    dataset : Dataset,
    split : str,
    batch_size: int,
):

    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=1,
    )
    while True:
        yield from loader

def prepare_torch_dataloader(
    dataset : Dataset,
    split : str,
    shuffle : bool,
    batch_size: int,
) -> torch.utils.data.DataLoader:

    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return loader

def dataset_from_csv(paths : Dict[str, str], cat_features, target, T):
    assert 'train' in paths
    y = {}
    X_num = {}
    X_cat = {} if len(cat_features) else None
    for split in paths.keys():
        df = pd.read_csv(paths[split])
        y[split] = df[target].to_numpy().astype(float)
        if X_cat is not None:
            X_cat[split] = df[cat_features].to_numpy().astype(str)
        X_num[split] = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)

    dataset = Dataset(X_num, X_cat, y, {}, None, len(np.unique(y['train'])))
    return transform_dataset(dataset, T, None)

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def prepare_fast_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
   
    # Build numpy array X depending on which feature groups are present.
    if D.X_num is None and D.X_cat is None:
        raise ValueError('Both X_num and X_cat are missing in Dataset; cannot build dataloader')

    if D.X_num is None:
        X_np = D.X_cat[split]
    elif D.X_cat is None:
        X_np = D.X_num[split]
    else:
        X_np = np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)

    # Ensure numeric type for torch conversion
    if not np.issubdtype(X_np.dtype, np.number):
        X_np = X_np.astype(float)

    X = torch.from_numpy(X_np).float()
    dataloader = FastTensorDataLoader(X, batch_size=batch_size, shuffle=(split=='train'))
    while True:
        yield from dataloader

def prepare_fast_torch_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat is not None:
        X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split=='train'))
    return dataloader

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

def concat_features(D : Dataset):
    if D.X_num is None:
        assert D.X_cat is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_cat.items()}
    elif D.X_cat is None:
        assert D.X_num is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_num.items()}
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(D.X_num[part], columns=range(D.n_num_features)),
                    pd.DataFrame(
                        D.X_cat[part],
                        columns=range(D.n_num_features, D.n_features),
                    ),
                ],
                axis=1,
            )
            for part in D.y.keys()
        }

    return X

def concat_to_pd(X_num, X_cat, y):
    if X_num is None:
        return pd.concat([
            pd.DataFrame(X_cat, columns=list(range(X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    if X_cat is not None:
        return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(X_cat, columns=list(range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)

def read_pure_data(path, split='train'):
    """
    Load dataset from datasets/{dataname}.csv into an in-memory pandas DataFrame
    and reuse that DataFrame for subsequent calls. This function will NOT read
    or write any other auxiliary files (npy/json/toml). It only consults
    data_profile/{dataname}.json via `util.ensure_data_info` for metadata.
    """
    # Use the shared DataFrame accessor to ensure CSV is loaded once into memory.
    df, info = get_dataset_df(path)
    # determine column names and target index
    cols = info.get('column_names') if info is not None else list(df.columns)
    if cols is None:
        cols = list(df.columns)

    # target_col_idx may be list - take first element if present
    target_idx = None
    if info is not None and 'target_col_idx' in info and info['target_col_idx']:
        try:
            target_idx = int(info['target_col_idx'][0])
        except Exception:
            target_idx = None

    if target_idx is None:
        # fallback to last column
        target_idx = len(cols) - 1

    # resolve target column name
    try:
        target_col = cols[target_idx]
    except Exception:
        # if cols are not indexed by int, fallback to last column name
        target_col = list(df.columns)[-1]

    # categorical columns list if available (prefer explicit indices from info)
    cat_cols = []
    if info is not None and 'cat_col_idx' in info:
        try:
            cat_cols = [cols[int(i)] for i in info.get('cat_col_idx', [])]
        except Exception:
            cat_cols = []

    # build arrays
    y_all = df[target_col].to_numpy()

    X_cat_arr = None
    if len(cat_cols) > 0:
        X_cat_arr = df[cat_cols].to_numpy().astype(str)

    # numerical columns: prefer explicit list from info when available to
    # avoid accidentally including categorical features (e.g. 'Sex').
    if info is not None and 'num_col_idx' in info:
        try:
            num_cols = [cols[int(i)] for i in info.get('num_col_idx', [])]
        except Exception:
            num_cols = [c for c in list(df.columns) if c not in cat_cols and c != target_col]
    else:
        num_cols = [c for c in list(df.columns) if c not in cat_cols and c != target_col]
    X_num_arr = None
    if len(num_cols) > 0:
        # Convert numeric columns to float. If conversion fails, raise a clear error
        # that points to mismatched metadata so the profile can be inspected.
        try:
            X_num_arr = df[num_cols].to_numpy().astype(float)
        except Exception as e:
            raise ValueError(f"Failed to convert numeric columns {num_cols} to float: {e}")

    # For backward compatibility: return full dataset for 'train', and empty arrays for 'test'
    if split == 'train':
        return X_num_arr, X_cat_arr, y_all
    else:
        empty_y = np.array([], dtype=y_all.dtype)
        empty_X_num = np.empty((0, X_num_arr.shape[1])) if X_num_arr is not None else None
        empty_X_cat = np.empty((0, X_cat_arr.shape[1]), dtype=object) if X_cat_arr is not None else None
        return empty_X_num, empty_X_cat, empty_y

def read_changed_val(path, val_size=0.2):
    path = Path(path)
    X_num_train, X_cat_train, y_train = read_pure_data(path, 'train')
    X_num_val, X_cat_val, y_val = read_pure_data(path, 'val')
    # Use ensure_data_info to get canonical task_type. Do not read local info.json.
    try:
        dataset_name = os.path.basename(os.path.normpath(path))
        info = util.ensure_data_info(dataset_name)
        is_regression = info.get('task_type') == 'regression'
    except Exception:
        is_regression = False

    y = np.concatenate([y_train, y_val], axis=0)

    ixs = np.arange(y.shape[0])
    if is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)
    y_train = y[train_ixs]
    y_val = y[val_ixs]

    if X_num_train is not None:
        X_num = np.concatenate([X_num_train, X_num_val], axis=0)
        X_num_train = X_num[train_ixs]
        X_num_val = X_num[val_ixs]

    if X_cat_train is not None:
        X_cat = np.concatenate([X_cat_train, X_cat_val], axis=0)
        X_cat_train = X_cat[train_ixs]
        X_cat_val = X_cat[val_ixs]
    
    return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val

#############

def load_dataset_info(dataset_dir_name: str) -> Dict[str, Any]:
    # Load canonical profile and compute dataset-level metadata without
    # performing additional CSV reads if the DataFrame is already cached.
    info = util.ensure_data_info(dataset_dir_name)
    try:
        df, _ = get_dataset_df(dataset_dir_name)
        size = len(df)
        csv_path = util.locate_dataset_csv(dataset_dir_name)
    except Exception:
        size = info.get('train_num') or 0
        csv_path = None

    info['size'] = size
    info['n_features'] = info.get('n_num_features', 0) + info.get('n_cat_features', 0)
    info['path'] = str(Path(csv_path).resolve()) if csv_path is not None else None
    return info


def get_dataset_df(path_or_name: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """
    Return (DataFrame, info) for the dataset identified by path_or_name.
    The CSV is read once and cached in-memory for subsequent calls.
    """
    if not hasattr(read_pure_data, "_DF_CACHE"):
        read_pure_data._DF_CACHE = {}

    if isinstance(path_or_name, Path):
        dataset_name = os.path.basename(os.path.normpath(path_or_name))
    else:
        dataset_name = os.path.basename(os.path.normpath(str(path_or_name)))

    if dataset_name in read_pure_data._DF_CACHE:
        return read_pure_data._DF_CACHE[dataset_name]

    if dataset_name.endswith('.csv'):
        dataset_name = dataset_name[:-4]
    
    # Try to locate the CSV on disk. If it fails, fall back to any cached
    # DataFrame that might have been preloaded by the caller (e.g. the
    # runner that reads datasets/{dataset}.csv once and caches it).
    try:
        csv_path = util.locate_dataset_csv(dataset_name)
        df = pd.read_csv(csv_path)
        info = util.ensure_data_info(dataset_name)
        read_pure_data._DF_CACHE[dataset_name] = (df, info)
        return df, info
    except FileNotFoundError:
        # Accept dataset_name values like 'train'/'test' when callers pass them
        # through; try to find a cached entry for the canonical dataset.
        # If dataset_name itself is a split name, attempt to use the caller's
        # cache by checking entries that map to a dataset with that split.
        if dataset_name in read_pure_data._DF_CACHE:
            return read_pure_data._DF_CACHE[dataset_name]
        # As a last resort, search for any cached dataset and return it.
        if len(read_pure_data._DF_CACHE) > 0:
            # Return the first cached dataset (best-effort fallback)
            first_key = next(iter(read_pure_data._DF_CACHE.keys()))
            return read_pure_data._DF_CACHE[first_key]
        # No cache available — re-raise the original error to be explicit
        raise