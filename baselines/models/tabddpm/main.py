import os
import pandas as pd
import sys
from pathlib import Path
import json

# 获取当前文件的绝对路径，并向上追溯到项目根目录
curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent.parent  # 假设项目根目录是FL的上级目录
sys.path.append(str(root_dir))  # 添加根目录到Python路径

from models.tabddpm.train import train
from models.tabddpm.sample import sample


def main(args):

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = args.dataset
    device = f"cuda:{args.gpu}"

    config_path = f"data_profile/{dataset}.json"
    model_save_path = f"{curr_dir}/ckpt/{dataset}"
    real_data_path = f"datasets/{dataset}.csv"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    args.train = True

    with open(config_path, "r") as f:
        raw_config = json.load(f)
        print(raw_config)

    """ 
    Modification of configs
    """
    print("START TRAINING")

    model_params = {
        # "d_in": d_in,
        "num_classes": raw_config["num_classes"],
        "is_y_cond": False,
        "rtdl_params": {
            "d_layers": [
                1024,
                2048,
                2048,
                1024,
            ],
            "dropout": 0.0,
        },
    }

    T_dict = {
        "seed": 0,
        "normalization": "quantile",
        "num_nan_policy": "mean",
        "cat_nan_policy": "__none__",
        "cat_min_frequency": "__none__",
        "cat_encoding": "one-hot",
        "y_policy": "default",
    }

    train(
        raw_config=raw_config,
        model_save_path=model_save_path,
        real_data_path=real_data_path,
        task_type=raw_config["task_type"],
        model_type="mlp",
        model_params=model_params,
        T_dict=T_dict,
        steps=args.epochs,
        device=device,
    )

    dataset = args.dataset
    device = f"cuda:{args.gpu}"

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f"data_profile/{dataset}.json"
    model_save_path = f"{curr_dir}/ckpt/{dataset}"
    real_data_path = f"datasets/{dataset}.csv"
    sample_save_path = f"synthetic/{dataset}"

    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    args.train = True

    with open(config_path, "r") as f:
        raw_config = json.load(f)
        print(raw_config)

    """ 
    Modification of configs
    """
    print("START SAMPLING")

    model_params = {
        "num_classes": raw_config["num_classes"],
        "is_y_cond": False,
        "rtdl_params": {
            "d_layers": [
                1024,
                2048,
                2048,
                1024,
            ],
            "dropout": 0.0,
        },
    }

    T_dict = {
        "seed": 0,
        "normalization": "quantile",
        "num_nan_policy": "mean",
        "cat_nan_policy": "__none__",
        "cat_min_frequency": "__none__",
        "cat_encoding": "one-hot",
        "y_policy": "default",
    }

    sample(
        raw_config=raw_config,
        num_samples=pd.read_csv(real_data_path).shape[0],
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        task_type=raw_config["task_type"],
        model_params=model_params,
        T_dict=T_dict,
        device=device,
        ddim=args.ddim,
        save_path=args.save_path,
    )
