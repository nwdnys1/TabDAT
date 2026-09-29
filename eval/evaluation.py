import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    BayesianRidge,
)
from sklearn import svm, tree
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
try:
    from dython.nominal import compute_associations
except ImportError:
    compute_associations = None
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
try:
    from synthcity.metrics import eval_statistical
    from synthcity.plugins.core.dataloader import GenericDataLoader
except ImportError:
    eval_statistical = None
    GenericDataLoader = None
try:
    from sdmetrics.single_table import LogisticDetection
except ImportError:
    LogisticDetection = None
from sklearn.preprocessing import OneHotEncoder


def supervised_model_training(
    x_train, y_train, x_test, y_test, model_name, problem_type
):

    if problem_type == "Classification" and len(np.unique(y_train)) < 2:
        # Handle case where synthetic training data has only one class
        single_class = np.unique(y_train)[0]
        pred = np.full(y_test.shape, single_class)
        acc = metrics.accuracy_score(y_test, pred) * 100
        f1_score = metrics.precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0
        )[2]
        auc = 0.0  # Penalize AUC as it cannot be computed
        return [acc, auc, f1_score]

    if model_name == "lr":
        model = LogisticRegression(random_state=42, max_iter=500)
    elif model_name == "svm":
        model = svm.SVC(random_state=42, probability=True)
    elif model_name == "dt":
        model = tree.DecisionTreeClassifier(random_state=42)
    elif model_name == "rf":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "mlp":
        model = MLPClassifier(random_state=42, max_iter=100)
    elif model_name == "l_reg":
        model = LinearRegression()
    elif model_name == "ridge":
        model = Ridge(random_state=42)
    elif model_name == "lasso":
        model = Lasso(random_state=42)
    elif model_name == "B_ridge":
        model = BayesianRidge()
    elif model_name == "mlpr":
        model = MLPRegressor(random_state=42, max_iter=100)
    elif model_name == "dtr":
        model = tree.DecisionTreeRegressor(random_state=42)
    elif model_name == "rfr":
        model = RandomForestRegressor(random_state=42)
    elif model_name == "gbr":
        model = GradientBoostingRegressor(random_state=42)

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    if problem_type == "Classification":
        try:
            # The decision for multi-class or binary should be based on the ground truth (y_test)
            if len(np.unique(y_test)) > 2:
                predict = model.predict_proba(x_test)
                acc = metrics.accuracy_score(y_test, pred) * 100
                auc = metrics.roc_auc_score(
                    y_test, predict, average="weighted", multi_class="ovr"
                )
                f1_score = metrics.precision_recall_fscore_support(
                    y_test, pred, average="macro", zero_division=0
                )[2]
                return [acc, auc, f1_score]

            else:
                predict = model.predict_proba(x_test)[:, 1]
                acc = metrics.accuracy_score(y_test, pred) * 100
                auc = metrics.roc_auc_score(y_test, predict)
                f1_score = metrics.f1_score(
                    y_test, pred, average="macro", zero_division=0
                )
                return [acc, auc, f1_score]
        except ValueError as e:
            # This error occurs if the synthetic data is missing some classes from the real data.
            # Instead of crashing, we return degenerated metrics.
            error_str = str(e)
            if (
                "Number of classes in y_true not equal" in error_str
                or "needs samples from 2 classes" in error_str
            ):
                acc = metrics.accuracy_score(y_test, pred) * 100
                f1_score = metrics.precision_recall_fscore_support(
                    y_test, pred, average="macro", zero_division=0
                )[2]
                # AUC cannot be calculated if predict_proba has wrong shape, so we return 0 as a penalty.
                auc = 0.0
                return [acc, auc, f1_score]
            else:
                # Re-raise other unexpected ValueErrors
                raise e

    else:
        mape = metrics.mean_absolute_percentage_error(y_test, pred)
        evs = metrics.explained_variance_score(y_test, pred)
        r2_score = metrics.r2_score(y_test, pred)
        return [mape, evs, r2_score]


def get_utility_metrics(
    real_path,
    fake_paths,
    scaler="MinMax",
    type={"Classification": ["lr", "dt", "rf", "mlp"]},
    cat_cols=[],
    target_col=None,
    real_test_path=None,
):
    """Train TRTR/TSTR models on disjoint real-train/synthetic rows and real test.

    ``real_path`` is the generator's real training split. The test split must
    be explicit; splitting ``real_path`` here could silently reuse rows that
    were already seen by the generator.
    """
    if real_test_path is None:
        raise ValueError("real_test_path must be an explicit generator-held-out CSV")
    data_real = pd.read_csv(real_path, dtype=object).dropna()
    data_test = pd.read_csv(real_test_path, dtype=object).dropna()
    if data_real.columns.tolist() != data_test.columns.tolist():
        raise ValueError("Real training and test columns must match exactly")

    if target_col is None:
        target_col = data_real.columns.tolist()[-1]

    data_real_y = data_real[target_col]
    data_real_X = data_real.drop([target_col], axis=1)
    data_test_y = data_test[target_col]
    data_test_X = data_test.drop([target_col], axis=1)
    # 将df的列名转换为np的列表索引
    if cat_cols != "all":
        cat_cols = [
            data_real_X.columns.get_loc(col)
            for col in cat_cols if col != target_col
        ]

    data_real_y = data_real_y.to_numpy()
    data_real_X = data_real_X.to_numpy()
    data_test_y = data_test_y.to_numpy()
    data_test_X = data_test_X.to_numpy()


    problem = list(type.keys())[0]
    models = list(type.values())[0]

    X_train_real, y_train_real = data_real_X, data_real_y
    X_test_real, y_test_real = data_test_X, data_test_y

    if scaler == "MinMax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    if cat_cols == "all":
        cat_cols = list(range(data_real_X.shape[1]))
    num_cols = [col for col in range(data_real_X.shape[1]) if col not in cat_cols]

    X_train_real_scaled = X_train_real.copy()
    X_test_real_scaled = X_test_real.copy()

    if len(cat_cols) > 0:
        X_train_real_scaled[:, cat_cols] = oe.fit_transform(X_train_real[:, cat_cols])
        X_test_real_scaled[:, cat_cols] = oe.transform(X_test_real[:, cat_cols])
    if len(num_cols) > 0:
        X_train_real_scaled[:, num_cols] = scaler.fit_transform(X_train_real[:, num_cols])
        X_test_real_scaled[:, num_cols] = scaler.transform(X_test_real[:, num_cols])

    if problem == "Classification":
        le = LabelEncoder()
        y_train_real = le.fit_transform(y_train_real)
        y_test_real = le.transform(y_test_real)
    else:
        y_train_real = y_train_real.astype(float)
        y_test_real = y_test_real.astype(float)

    all_real_results = []

    for model in models:
        real_results = supervised_model_training(
            X_train_real_scaled,
            y_train_real,
            X_test_real_scaled,
            y_test_real,
            model,
            problem,
        )
        all_real_results.append(real_results)

    all_fake_results_avg = []

    for fake_path in fake_paths:

        data_fake = pd.read_csv(fake_path, dtype=object).dropna()
        if data_fake.columns.tolist() != data_real.columns.tolist():
            raise ValueError(f"Synthetic columns do not match real training data: {fake_path}")
        data_fake_y = data_fake[target_col]
        data_fake_X = data_fake.drop([target_col], axis=1)
        data_fake_y = data_fake_y.to_numpy()
        data_fake_X = data_fake_X.to_numpy()

        X_train_fake, y_train_fake = data_fake_X, data_fake_y

        # Use the scaler fitted on real data to transform the fake training data
        X_train_fake_scaled = X_train_fake.copy()
        if len(cat_cols) > 0:
            X_train_fake_scaled[:, cat_cols] = oe.transform(X_train_fake[:, cat_cols])
        if len(num_cols) > 0:
            X_train_fake_scaled[:, num_cols] = scaler.transform(X_train_fake[:, num_cols])
        if problem == "Classification":
            y_train_fake = le.transform(y_train_fake)
        else:
            y_train_fake = y_train_fake.astype(float)

        all_fake_results = []
        for model in models:
            fake_results = supervised_model_training(
                X_train_fake_scaled,
                y_train_fake,
                X_test_real_scaled,
                y_test_real,
                model,
                problem,
            )
            all_fake_results.append(fake_results)

        all_fake_results_avg.append(all_fake_results)

    return np.array(all_real_results), np.array(all_fake_results_avg).squeeze()


def stat_sim(real_path, fake_path, cat_cols=None):
    if compute_associations is None:
        raise ImportError("stat_sim requires dython; install evaluation dependencies")

    Stat_dict = {}

    real = pd.read_csv(real_path)
    fake = pd.read_csv(fake_path)

    if cat_cols is None:
        cat_cols = []
    if cat_cols == "all":
        cat_cols = real.columns.tolist()

    really = real.copy()
    fakey = fake.copy()

    real_corr = compute_associations(real, cat_cols)

    fake_corr = compute_associations(fake, cat_cols)

    # Average each distinct pair once; a Frobenius norm grows with table width.
    pair_indices = np.triu_indices(len(real.columns), k=1)
    association_error = (
        float(np.mean(np.abs(np.asarray(real_corr) - np.asarray(fake_corr))[pair_indices]))
        if len(pair_indices[0]) else np.nan
    )

    cat_stat = []
    num_stat = []

    for column in real.columns:

        if column in cat_cols:
            # 1. 计算归一化概率分布
            real_pdf = really[column].value_counts(normalize=True)
            fake_pdf = fakey[column].value_counts(normalize=True)

            # 2. 统一索引类型为字符串，避免 int/str 不匹配问题
            real_pdf.index = real_pdf.index.astype(str)
            fake_pdf.index = fake_pdf.index.astype(str)

            # 3. 获取所有出现的类别的并集 (Union of supports)
            all_categories = sorted(list(set(real_pdf.index) | set(fake_pdf.index)))

            # 4. 安全地对齐两个分布 (缺失的类别填 0)
            real_pdf_values = [real_pdf.get(cat, 0.0) for cat in all_categories]
            fake_pdf_values = [fake_pdf.get(cat, 0.0) for cat in all_categories]

            # scipy returns Jensen-Shannon distance, not the divergence.
            Stat_dict[column] = distance.jensenshannon(
                real_pdf_values, fake_pdf_values, base=2.0
            )
            cat_stat.append(Stat_dict[column])

        else:
            scaler = MinMaxScaler()
            scaler.fit(real[column].values.reshape(-1, 1))
            l1 = scaler.transform(real[column].values.reshape(-1, 1)).flatten()
            l2 = scaler.transform(fake[column].values.reshape(-1, 1)).flatten()
            Stat_dict[column] = wasserstein_distance(l1, l2)
            num_stat.append(Stat_dict[column])

    return [
        fake_path,
        float(np.mean(num_stat)) if num_stat else np.nan,
        float(np.mean(cat_stat)) if cat_stat else np.nan,
        association_error,
    ]

def get_extra_metrics(real_path, fake_path, cat_cols=None, *, max_rows=2000, seed=42):
    """Compute naive alpha/beta and LogisticDetection on equal-size samples.

    Record n_rows with scores; formal comparisons need a predeclared common
    sample size. These quality metrics do not establish DP privacy.
    """
    missing = []
    if eval_statistical is None or GenericDataLoader is None:
        missing.append("synthcity==0.2.12")
    if LogisticDetection is None:
        missing.append("sdmetrics==0.21.0")
    if missing:
        raise ImportError("Extra metrics require: " + ", ".join(missing))
    if max_rows is not None and max_rows < 2:
        raise ValueError("max_rows must be at least 2 or None")

    real_data = pd.read_csv(real_path).dropna()
    fake_data = pd.read_csv(fake_path).dropna()
    if real_data.columns.tolist() != fake_data.columns.tolist():
        raise ValueError("Real and synthetic columns must match exactly")
    if cat_cols == "all":
        cat_cols = real_data.columns.tolist()
    else:
        cat_cols = [] if cat_cols is None else list(cat_cols)
    unknown = set(cat_cols) - set(real_data.columns)
    if unknown:
        raise ValueError(f"Unknown categorical columns: {sorted(unknown)}")

    n_rows = min(len(real_data), len(fake_data))
    if max_rows is not None:
        n_rows = min(n_rows, max_rows)
    if n_rows < 2:
        raise ValueError("Extra metrics need at least 2 complete rows per table")
    real_data = real_data.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    fake_data = fake_data.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    num_cols = [col for col in real_data.columns if col not in cat_cols]

    # Fit the categorical encoder on the reference only.
    arrays_real, arrays_fake = [], []
    if num_cols:
        arrays_real.append(real_data[num_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float))
        arrays_fake.append(fake_data[num_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float))
    if cat_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        arrays_real.append(encoder.fit_transform(real_data[cat_cols].astype(str)))
        arrays_fake.append(encoder.transform(fake_data[cat_cols].astype(str)))
    real_processed = pd.DataFrame(np.concatenate(arrays_real, axis=1))
    fake_processed = pd.DataFrame(np.concatenate(arrays_fake, axis=1))
    quality = eval_statistical.AlphaPrecision().evaluate(
        GenericDataLoader(real_processed), GenericDataLoader(fake_processed)
    )
    # Match TabSyn's naive-space variant, not synthcity's "_OC" score.
    alpha_precision = float(quality["delta_precision_alpha_naive"])
    beta_recall = float(quality["delta_coverage_beta_naive"])

    metadata = {"columns": {
        col: {"sdtype": "categorical" if col in cat_cols else "numerical"}
        for col in real_data.columns
    }}
    c2st_score = float(LogisticDetection.compute(
        real_data=real_data, synthetic_data=fake_data, metadata=metadata
    ))
    return {
        "alpha_precision": alpha_precision,
        "beta_recall": beta_recall,
        "c2st": c2st_score,
        "n_rows": n_rows,
    }


def privacy_metrics(real_path, fake_path, cat_cols):
    """
    Legacy privacy metrics, paused for the non-DP evaluation protocol.

    """
    raise RuntimeError("Privacy evaluation is currently disabled")
    real = pd.read_csv(real_path, dtype=object).dropna().drop_duplicates()
    fake = pd.read_csv(fake_path, dtype=object).dropna().drop_duplicates()

    # If the dataset is too large, sample it down to avoid memory issues
    max_rows = 10000
    if len(real) > max_rows:
        print(f"Real data has {len(real)} rows, sampling down to {max_rows}.")
        real = real.sample(n=max_rows, random_state=42)
    if len(fake) > max_rows:
        print(f"Fake data has {len(fake)} rows, sampling down to {max_rows}.")
        fake = fake.sample(n=max_rows, random_state=42)

    if cat_cols:
        if cat_cols == "all":
            cat_cols = real.columns.tolist()
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        real[cat_cols] = encoder.fit_transform(real[cat_cols])
        fake[cat_cols] = encoder.transform(fake[cat_cols])

    real_refined = real.copy()
    fake_refined = fake.copy()

    scalerR = MinMaxScaler()
    scalerR.fit(real_refined)
    scalerF = MinMaxScaler()
    scalerF.fit(fake_refined)
    df_real_scaled = scalerR.transform(real_refined)
    df_fake_scaled = scalerF.transform(fake_refined)

    dist_rf = metrics.pairwise_distances(
        df_real_scaled, Y=df_fake_scaled, metric="minkowski", n_jobs=1
    )
    dist_rr = metrics.pairwise_distances(
        df_real_scaled, Y=None, metric="minkowski", n_jobs=1
    )
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0], dtype=bool)].reshape(
        dist_rr.shape[0], -1
    )
    dist_ff = metrics.pairwise_distances(
        df_fake_scaled, Y=None, metric="minkowski", n_jobs=1
    )
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0], dtype=bool)].reshape(
        dist_ff.shape[0], -1
    )
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [
        dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))
    ]
    smallest_two_indexes_rr = [
        rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))
    ]
    smallest_two_rr = [
        rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))
    ]
    smallest_two_indexes_ff = [
        rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))
    ]
    smallest_two_ff = [
        rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))
    ]
    nn_ratio_rr = np.array([i[0] / i[1] for i in smallest_two_rr])
    nn_ratio_ff = np.array([i[0] / i[1] for i in smallest_two_ff])
    nn_ratio_rf = np.array([i[0] / i[1] for i in smallest_two_rf])
    nn_fifth_perc_rr = np.percentile(nn_ratio_rr, 5)
    nn_fifth_perc_ff = np.percentile(nn_ratio_ff, 5)
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf, 5)

    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf, 5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr, 5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff, 5)

    dcr_gain = fifth_perc_rf / fifth_perc_rr
    nndr_gain = nn_fifth_perc_rf / nn_fifth_perc_rr

    return np.array(
        [
            fifth_perc_rf,
            fifth_perc_rr,
            fifth_perc_ff,
            dcr_gain,
            nn_fifth_perc_rf,
            nn_fifth_perc_rr,
            nn_fifth_perc_ff,
            nndr_gain,
        ]
    ).reshape(1, 8)


# Privacy metrics are paused; do not import their optional dependencies here.
# from syntheval import SynthEval
SynthEval = None


def get_adr_metric(real_path, fake_path, cat_cols=None):
    """
    Computes the Adversarial Accuracy Reciprocal (ADR) using the syntheval library.
    """
    raise RuntimeError("Privacy evaluation is currently disabled")

    real_data = pd.read_csv(real_path)
    fake_data = pd.read_csv(fake_path)

    if cat_cols == "all":
        cat_cols = real_data.columns.tolist()

    try:
        evaluator = SynthEval(real_data, cat_cols=cat_cols, verbose=False)
        results = evaluator.evaluate(fake_data, att_discl={})
        print(f"\nADR results for {fake_path}: {results['val'].iloc[0]}")
        # ADR is the reciprocal of the adversarial accuracy
        return [fake_path, results["val"].iloc[0]]
    except Exception as e:
        print(f"Error calculating ADR for {fake_path}: {e}")
        return [fake_path, np.nan]


def get_eps_metric(real_path, fake_path, cat_cols=None):
    """
    Computes the Epsilon metric using the syntheval library.
    """
    raise RuntimeError("Privacy evaluation is currently disabled")

    real_data = pd.read_csv(real_path)
    fake_data = pd.read_csv(fake_path)

    if cat_cols == "all":
        cat_cols = real_data.columns.tolist()

    try:
        evaluator = SynthEval(real_data, cat_cols=cat_cols, verbose=False)
        results = evaluator.evaluate(fake_data, eps_risk={})
        print(f"\nEpsilon results for {fake_path}: {results['val'].iloc[0]}")
        return [fake_path, results["val"].iloc[0]]
    except Exception as e:
        print(f"Error calculating Epsilon for {fake_path}: {e}")
        return [fake_path, np.nan]


def get_mia_metric(real_path, fake_path, cat_cols=None):
    """
    Computes the Membership Inference Attack (MIA) metric using the syntheval library.
    """
    raise RuntimeError("Privacy evaluation is currently disabled")

    real_data = pd.read_csv(real_path)
    fake_data = pd.read_csv(fake_path)

    if cat_cols == "all":
        cat_cols = real_data.columns.tolist()

    try:
        evaluator = SynthEval(real_data, cat_cols=cat_cols, verbose=False)
        results = evaluator.evaluate(fake_data, mia_risk={})
        print(f"\nMIA results for {fake_path}: {results['val'].iloc[0]}")
        return [fake_path, results["val"].iloc[0]]
    except Exception as e:
        print(f"Error calculating MIA for {fake_path}: {e}")
        return [fake_path, np.nan]


def get_hit_metric(real_path, fake_path, cat_cols=None):
    """
    Computes the Hitting Rate (HIT) metric using the syntheval library.
    """
    raise RuntimeError("Privacy evaluation is currently disabled")

    real_data = pd.read_csv(real_path)
    fake_data = pd.read_csv(fake_path)

    if cat_cols == "all":
        cat_cols = real_data.columns.tolist()

    try:
        evaluator = SynthEval(real_data, cat_cols=cat_cols, verbose=False)
        results = evaluator.evaluate(fake_data, hit_rate={})
        print(f"\nHIT results for {fake_path}: {results['val'].iloc[0]}")
        return [fake_path, results["val"].iloc[0]]
    except Exception as e:
        print(f"Error calculating HIT for {fake_path}: {e}")
        return [fake_path, np.nan]


def get_nnaa_metric(real_path, fake_path, cat_cols=None):
    """
    Computes the Nearest Neighbor Adversarial Accuracy (NNAA) metric using the syntheval library.
    """
    raise RuntimeError("Privacy evaluation is currently disabled")

    real_data = pd.read_csv(real_path)
    fake_data = pd.read_csv(fake_path)

    if cat_cols == "all":
        cat_cols = real_data.columns.tolist()

    try:
        evaluator = SynthEval(real_data, cat_cols=cat_cols, verbose=False)
        results = evaluator.evaluate(fake_data, nnaa={})
        print(f"\nNNAA results for {fake_path}: {results['val'].iloc[0]}")
        return [fake_path, results["val"].iloc[0]]
    except Exception as e:
        print(f"Error calculating NNAA for {fake_path}: {e}")
        return [fake_path, np.nan]
