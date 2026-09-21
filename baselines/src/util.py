import argparse
import atexit
import enum
import json
import os
import pickle
import shutil
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast, get_args, get_origin

import __main__
import numpy as np
import tomli
import tomli_w
import torch
# import zero
import typing as ty

from . import env

RawConfig = Dict[str, Any]
Report = Dict[str, Any]
T = TypeVar('T')


class Part(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __str__(self) -> str:
        return self.value


class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value




def update_training_log(training_log, data, metrics):
    def _update(log_part, data_part):
        for k, v in data_part.items():
            if isinstance(v, dict):
                _update(log_part.setdefault(k, {}), v)
            elif isinstance(v, list):
                log_part.setdefault(k, []).extend(v)
            else:
                log_part.setdefault(k, []).append(v)

    _update(training_log, data)
    transposed_metrics = {}
    for part, part_metrics in metrics.items():
        for metric_name, value in part_metrics.items():
            transposed_metrics.setdefault(metric_name, {})[part] = value
    _update(training_log, transposed_metrics)


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = '__none__'


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        tomli_w.dump(pack_config(config), f)
    # check that there are no bugs in all these "pack/unpack" things
    assert config == load_config(path)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    # Only allow writing canonical profile JSONs under data_profile/,
    # otherwise avoid creating arbitrary JSON files during baseline runs.
    kwargs.setdefault('indent', 4)
    p = Path(path)
    try:
        if 'data_profile' in str(p.resolve()):
            p.write_text(json.dumps(x, **kwargs) + '\n')
        else:
            # skip writes to other locations
            return
    except Exception:
        return


def ensure_data_info(dataname: str, data_dir_root: str = 'data', profile_dir: str = 'data_profile') -> dict:
    """
    Load and normalize the canonical dataset profile stored at
    `data_profile/{dataname}.json` and return an `info` dict.

    This function does NOT create or write any `data/{dataname}/info.json` or
    other auxiliary files. It computes idx mappings and sensible defaults
    consumed by baselines.
    """
    profile_path = Path(profile_dir) / f"{dataname}.json"
    data_path = Path(data_dir_root) / dataname

    if not profile_path.exists():
        raise FileNotFoundError(f'Profile not found: {profile_path}')

    info = json.loads(profile_path.read_text())

    # do NOT create or write any files on disk; only read canonical profile
    # normalize expected keys and provide sensible defaults
    column_names = info.get('column_names') or []
    n_cols = len(column_names)
    num_idx = info.get('num_col_idx') or []
    cat_idx = info.get('cat_col_idx') or []
    target_idx = info.get('target_col_idx') or []

    # compute idx_mapping: position in concatenated vector [num..., cat..., target...]
    # If profile already contains a precomputed idx_mapping, prefer it.
    if 'idx_mapping' in info and isinstance(info['idx_mapping'], dict) and info['idx_mapping']:
        try:
            # Use precomputed mapping if valid
            idx_mapping = {int(k): int(v) for k, v in info['idx_mapping'].items()}
        except (ValueError, TypeError):
            # Fallback to recomputing if precomputed is invalid
            idx_mapping = {}
    else:
        idx_mapping = {}

    if not idx_mapping:
        # Dynamically compute idx_mapping with a strict [num, cat, target] order
        pos = 0
        
        # 1. Numerical columns
        for i in sorted(num_idx): # Sort to ensure deterministic order
            idx_mapping[int(i)] = pos
            pos += 1
        
        # 2. Categorical columns
        # This part handles complex cases like one-hot encoding widths
        cat_widths = None
        for key in ('cat_col_widths', 'cat_sizes', 'cat_cardinalities', 'category_sizes'):
            if key in info:
                cat_widths = list(info[key])
                break
        
        if cat_widths is not None and len(cat_widths) == len(cat_idx):
            # Create a mapping from original cat_idx to its width
            width_map = {int(idx): width for idx, width in zip(cat_idx, cat_widths)}
            for i in sorted(cat_idx):
                idx_mapping[int(i)] = pos
                try:
                    w = int(width_map.get(i, 1))
                except (ValueError, TypeError):
                    w = 1
                pos += max(1, w)
        else:
            # Simple case: one slot per categorical column
            for i in sorted(cat_idx):
                idx_mapping[int(i)] = pos
                pos += 1
        
        # 3. Target columns
        for i in sorted(target_idx):
            idx_mapping[int(i)] = pos
            pos += 1

    # name mapping
    idx_name_mapping = {i: column_names[i] for i in range(n_cols)} if column_names else {i: str(i) for i in range(n_cols)}

    out_info = dict(info)  # shallow copy
    out_info.setdefault('name', dataname)
    out_info.setdefault('file_type', info.get('file_type', 'csv'))
    out_info.setdefault('data_path', str((data_path / 'train.csv').resolve()))
    out_info.setdefault('train_num', info.get('train_num', None))
    out_info.setdefault('num_col_idx', num_idx)
    out_info.setdefault('cat_col_idx', cat_idx)
    out_info.setdefault('target_col_idx', target_idx)
    out_info.setdefault('column_names', column_names)
    # export idx maps as json-serializable dicts
    out_info['idx_mapping'] = {str(k): int(v) for k, v in idx_mapping.items()}
    out_info['idx_name_mapping'] = {str(k): v for k, v in idx_name_mapping.items()}

    # Do not write info.json to disk. Return the composed info dict only.

    return out_info


def locate_dataset_csv(dataname: str, info: Optional[dict] = None) -> str:
    """Return the canonical datasets/{dataname}.csv absolute path.

    This intentionally restricts dataset lookup to the repository's `datasets/`
    folder so the code never reads auxiliary CSVs elsewhere.
    """
    candidate = os.path.join(os.getcwd(), 'datasets', f'{dataname}.csv')
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Dataset CSV for {dataname} not found at {candidate}")


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    # Avoid writing pickles during baseline runs to satisfy single-file requirement.
    return


def load(path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](Path(path), **kwargs)


def dump(x: Any, path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'dump_{Path(path).suffix[1:]}'](x, Path(path), **kwargs)


def _get_output_item_path(
    path: Union[str, Path], filename: str, must_exist: bool
) -> Path:
    path = env.get_path(path)
    if path.suffix == '.toml':
        path = path.with_suffix('')
    if path.is_dir():
        path = path / filename
    else:
        assert path.name == filename
    assert path.parent.exists()
    if must_exist:
        assert path.exists()
    return path


def load_report(path: Path) -> Report:
    # Do not read report.json from disk. Return an empty report placeholder.
    return {}


def dump_report(report: dict, path: Path) -> None:
    # Intentionally no-op to avoid creating report.json on disk.
    return


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    # Loading persisted predictions is disabled in this baseline; return empty.
    return {}


def dump_predictions(predictions: Dict[str, np.ndarray], path: Path) -> None:
    # Persisting predictions is disabled to avoid writing .npz files.
    return


def dump_metrics(metrics: Dict[str, Any], path: Path) -> None:
    # Avoid writing metrics.json to disk.
    return


def load_checkpoint(path: Path, *args, **kwargs) -> Dict[str, np.ndarray]:
    return torch.load(
        _get_output_item_path(path, 'checkpoint.pt', True), *args, **kwargs
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def _print_sep(c, size=100):
    print(c * size)


def start(
    config_cls: Type[T] = RawConfig,
    argv: Optional[List[str]] = None,
    patch_raw_config: Optional[Callable[[RawConfig], None]] = None,
) -> Tuple[T, Path, Report]:  # config  # output dir  # report
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--continue', action='store_true', dest='continue_')
    if argv is None:
        program = __main__.__file__
        args = parser.parse_args()
    else:
        program = argv[0]
        try:
            args = parser.parse_args(argv[1:])
        except Exception:
            print(
                'Failed to parse `argv`.'
                ' Remember that the first item of `argv` must be the path (relative to'
                ' the project root) to the script/notebook.'
            )
            raise
    args = parser.parse_args(argv)

    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert args.continue_

    config_path = env.get_path(args.config)
    output_dir = config_path.with_suffix('')
    _print_sep('=')
    print(f'[output] {output_dir}')
    _print_sep('=')

    assert config_path.exists()
    raw_config = load_config(config_path)
    if patch_raw_config is not None:
        patch_raw_config(raw_config)
    if is_dataclass(config_cls):
        config = from_dict(config_cls, raw_config)
        full_raw_config = asdict(config)
    else:
        assert config_cls is dict
        full_raw_config = config = raw_config
    full_raw_config = asdict(config)

    if output_dir.exists():
        if args.force:
            print('Removing the existing output and creating a new one...')
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        elif not args.continue_:
            backup_output(output_dir)
            print('The output directory already exists. Done!\n')
            sys.exit()
        elif output_dir.joinpath('DONE').exists():
            backup_output(output_dir)
            print('The "DONE" file already exists. Done!')
            sys.exit()
        else:
            print('Continuing with the existing output...')
    else:
        print('Creating the output...')
        output_dir.mkdir()

    report = {
        'program': str(env.get_relative_path(program)),
        'environment': {},
        'config': full_raw_config,
    }
    if torch.cuda.is_available():  # type: ignore[code]
        env_info = {'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'torch.version.cuda': torch.version.cuda,
                    'torch.backends.cudnn.version()': torch.backends.cudnn.version(),  # type: ignore[code]
                    'torch.cuda.nccl.version()': torch.cuda.nccl.version(),  # type: ignore[code]
                    }
        # zero is an optional dependency; include GPU details if available
        try:
            import zero  # type: ignore

            env_info['gpus'] = zero.hardware.get_gpus_info()
        except Exception:
            pass
        report['environment'].update(env_info)
    # Intentionally do not write report/raw_config JSON files to disk to
    # comply with the single-profile requirement. Training outputs (checkpoints)
    # are still created by the training code.
    _print_sep('-')
    pprint(full_raw_config, width=100)
    _print_sep('-')
    return config, output_dir, report


_LAST_SNAPSHOT_TIME = None


def backup_output(output_dir: Path) -> None:
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output_dir.relative_to(env.PROJ)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output_dir = dir_ / relative_output_dir
        prev_backup_output_dir = new_output_dir.with_name(new_output_dir.name + '_prev')
        new_output_dir.parent.mkdir(exist_ok=True, parents=True)
        if new_output_dir.exists():
            new_output_dir.rename(prev_backup_output_dir)
        shutil.copytree(output_dir, new_output_dir)
        # the case for evaluate.py which automatically creates configs
        # We intentionally do not copy or rely on .toml files; canonical
        # configuration is stored in data_profile/{dataname}.json.
        if prev_backup_output_dir.exists():
            shutil.rmtree(prev_backup_output_dir)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def _get_scores(metrics: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, float]]:
    return (
        {k: v['score'] for k, v in metrics.items()}
        if 'score' in next(iter(metrics.values()))
        else None
    )


def format_scores(metrics: Dict[str, Dict[str, Any]]) -> str:
    return ' '.join(
        f"[{x}] {metrics[x]['score']:.3f}"
        for x in ['test', 'val', 'train']
        if x in metrics
    )


def finish(output_dir: Path, report: dict) -> None:
    print()
    _print_sep('=')

    metrics = report.get('metrics')
    if metrics is not None:
        scores = _get_scores(metrics)
        if scores is not None:
            # Do not persist scores.json to disk in this baseline; print instead.
            print(format_scores(metrics))
            _print_sep('-')

    dump_report(report, output_dir)
    # JSON_OUTPUT_FILE handling removed: report.json is not produced by this baseline.

    output_dir.joinpath('DONE').touch()
    backup_output(output_dir)
    print(f'Done! | {report.get("time")} | {output_dir}')
    _print_sep('=')
    print()


def from_dict(datacls: Type[T], data: dict) -> T:
    assert is_dataclass(datacls)
    data = deepcopy(data)
    for field in fields(datacls):
        if field.name not in data:
            continue
        if is_dataclass(field.type):
            data[field.name] = from_dict(field.type, data[field.name])
        elif (
            get_origin(field.type) is Union
            and len(get_args(field.type)) == 2
            and get_args(field.type)[1] is type(None)
            and is_dataclass(get_args(field.type)[0])
        ):
            if data[field.name] is not None:
                data[field.name] = from_dict(get_args(field.type)[0], data[field.name])
    return datacls(**data)


def replace_factor_with_value(
    config: RawConfig,
    key: str,
    reference_value: int,
    bounds: Tuple[float, float],
) -> None:
    factor_key = key + '_factor'
    if factor_key not in config:
        assert key in config
    else:
        assert key not in config
        factor = config.pop(factor_key)
        assert bounds[0] <= factor <= bounds[1]
        config[key] = int(factor * reference_value)


def get_temporary_copy(path: Union[str, Path]) -> Path:
    path = env.get_path(path)
    assert not path.is_dir() and not path.is_symlink()
    tmp_path = path.with_name(
        path.stem + '___' + str(uuid.uuid4()).replace('-', '') + path.suffix
    )
    shutil.copyfile(path, tmp_path)
    atexit.register(lambda: tmp_path.unlink())
    return tmp_path


def get_python():
    python = Path('python3.9')
    return str(python) if python.exists() else 'python'

def get_catboost_config(real_data_path, is_cv=False):
    ds_name = Path(real_data_path).name
    C = load_json(f'tuned_models/catboost/{ds_name}_cv.json')
    return C

def get_categories(X_train_cat):
    return (
        None
        if X_train_cat is None
        else [
            len(set(X_train_cat[:, i]))
            for i in range(X_train_cat.shape[1])
        ]
    )