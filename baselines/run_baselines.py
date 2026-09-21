import os
from types import SimpleNamespace

# import baseline modules
from models.sdv import main as sdv_main
from models.ctabgan import main as ctabgan_main
from models.ctabganplus import ctabgan as ctabganplus_main
from models.tabddpm import main as tabddpm_main
from models.ttvae import main as ttvae_main
from models.dp.pate_gan import train_pate_gan
from models.dp.dp_wgan import train_dpwgan
from models.dp.dp_tbart import main as dp_tbart_main
from models.tabmt import main as tabmt_main
from models.tabnat import main as tabnat_main
from models.tabsyn import main as tabsyn_main

BASE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(BASE, "..", ".."))
DATASETS = os.path.join("datasets")
DATA = os.path.join(REPO, "data")
DATAPROFILE = os.path.join(BASE, "data_profile")
TABDDPM_CONFIGS = os.path.join(BASE, "models", "tabddpm", "configs")
FAKE_ROOT = os.path.join("TabDAT", "eval", "fake_datasets")


cat_cols = {
    "adult": [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "class",
    ],
    "barley": "all",
    "ecoli70": [],
    "loan": [
        "ZIP Code",
        "Education",
        "Personal Loan",
        "Securities Account",
        "CD Account",
        "Online",
        "CreditCard",
        "Mortgage",
    ],
    "credit": ["Class"],
    "insurance": [
        "sex",
        "smoker",
        "region",
    ],
    "king": [
        "waterfront",
        "yr_renovated",
        "zipcode",
    ],
    "covertype": [f"Soil_Type{i}" for i in range(1, 41)]
    + [f"Wilderness_Area{i}" for i in range(1, 5)]
    + ["Covertype"],
    "pm25": ["cbwd"],
}


def run_ctgan(dataset):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    args.epochs = 300

    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_ctgan_{i}.csv")

    sdv_main.train_ctgan(args)


def run_tvae(dataset):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    args.epochs = 300

    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_tvae_{i}.csv")

    sdv_main.train_tvae(args)


def run_ctabgan(dataset):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    args.epochs = 150

    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_ctabgan_{i}.csv")

    ctabgan_main.train(args)


def run_ctabganplus(dataset):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    args.epochs = 150
    args.private = False
    args.eps = 1.0

    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_ctabganplus_{i}.csv")

    ctabganplus_main.train(args)


def run_tabddpm(dataset):
    args = SimpleNamespace()
    args.dataset = dataset
    args.gpu = 0
    args.ddim = True
    args.epochs = 1000

    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_tabddpm_{i}.csv")
    tabddpm_main.main(args)


def run_ttvae(dataset):
    args = SimpleNamespace()
    args.dataname = dataset
    args.gpu = 0
    args.epochs = 300

    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_ttvae_{i}.csv")

    ttvae_main.train(args)


def run_tabmt(dataset):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    args.epochs = 100
    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_tabmt_{i}.csv")
    tabmt_main.main(args)


def run_tabnat(dataset):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    args.epochs = 100
    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_tabnat_{i}.csv")
    tabnat_main.main(args)


def run_tabsyn(dataset):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    out_dir = os.path.join(FAKE_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_tabsyn_{i}.csv")
    tabsyn_main.main(args)


# baselines with dp
def run_pategan(dataset, eps=1.0):

    print(f"Running PATE-GAN for {dataset}...")

    data_path = os.path.join(DATASETS, f"{dataset}.csv")
    out_dir = os.path.join(FAKE_ROOT, f"{dataset}/eps{eps}")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"sampled_{dataset}_pategan_{i}.csv")

    train_pate_gan(data_path, save_path, cat_cols=cat_cols[dataset], eps=eps)


def run_dpwgan(dataset, eps=1.0):
    print(f"Running DP-WGAN for {dataset} with epsilon={eps}...")

    data_path = os.path.join(DATASETS, f"{dataset}.csv")
    out_dir = os.path.join(FAKE_ROOT, f"{dataset}/eps{eps}")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"sampled_{dataset}_dpwgan_{i}.csv")

    train_dpwgan(data_path, save_path, cat_cols=cat_cols[dataset], eps=eps)


def run_ctabganplus_dp(dataset, eps=1.0):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    args.epochs = 150
    args.private = True
    args.eps = eps

    out_dir = os.path.join(FAKE_ROOT, f"{dataset}/eps{eps}")
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_ctabganplus_{i}.csv")

    ctabganplus_main.train(args)


def run_dptbart(dataset, eps=1.0):
    args = SimpleNamespace()
    args.dataname = dataset
    args.device = "gpu"
    args.epochs = 100
    args.dp = True
    args.eps = eps

    out_dir = os.path.join(FAKE_ROOT, f"{dataset}/eps{eps}")
    os.makedirs(out_dir, exist_ok=True)
    args.save_path = os.path.join(out_dir, f"sampled_{dataset}_dptbart_{i}.csv")

    dp_tbart_main.main(args)


def run_baselines(dataset):
    global i
    i = 1

    for t in range(5):
        # run_ctgan(dataset)
        # run_tvae(dataset)
        # run_ctabgan(dataset)
        # run_ctabganplus(dataset)
        # run_tabddpm(dataset)
        # run_ttvae(dataset)
        # run_great(dataset)
        # run_tabmt(dataset)
        # run_tabsyn(dataset)
        # run_tabnat(dataset)
        # run_pategan(dataset)
        # run_dpwgan(dataset)
        # run_ctabganplus_dp(dataset)
        # run_dptbart(dataset)

        i += 1


if __name__ == "__main__":

    os.makedirs(FAKE_ROOT, exist_ok=True)

    for dataset in [
        "adult",
        "barley",
        "ecoli70",
        "insurance",
        "loan",
        "credit",
        "king",
        "covertype",
        "pm25",
    ]:
        run_baselines(dataset)
