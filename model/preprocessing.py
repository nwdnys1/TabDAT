"""TabDAT methods extracted without changing their implementation."""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, MinMaxScaler


class PreprocessingMixin:
    def _load_and_preprocess_data(self, file_path):
        """
        Loads data from a file, infers types, preprocesses it, and stores it.
        """
        df = pd.read_csv(file_path).dropna()
        self.col_names = df.columns.tolist()
        self.num_vars = len(df.columns)

        # --- Infer Variable Types ---
        cat_cols = {}
        cont_cols = []
        if self.cat_cols is None:
            for i, col_name in enumerate(df.columns):
                col = df[col_name]
                # 1. Explicitly check for object or bool types first.
                if col.dtype == "object" or col.dtype == "bool":
                    cat_cols[i] = col.nunique(dropna=False)
                    continue
                numeric_col = pd.to_numeric(df[col_name], errors="coerce")
                if numeric_col.isnull().sum() > 0 or df[col_name].dtype == "object":
                    cardinality = df[col_name].nunique(dropna=False)
                    cat_cols[i] = cardinality
                else:
                    cont_cols.append(i)
        else:
            for idx in self.cat_cols:
                cat_cols[idx] = df.iloc[:, idx].nunique(dropna=False)
            for i in range(self.num_vars):
                if i not in cat_cols:
                    cont_cols.append(i)
        self.var_types = {"cat": cat_cols, "cont": cont_cols}

        # for log columns
        if self.log_cols:
            self.lower_bounds = {}
            for col_idx in self.log_cols:
                col = df.columns[col_idx]
                lower = np.min(df[col].values)
                # store lower bound for inverse transform
                self.lower_bounds[col] = lower
                if lower > 0:
                    df[col] = df[col].apply(lambda x: np.log(x))
                elif lower == 0:
                    df[col] = df[col].apply(lambda x: np.log(x + 1))
                else:
                    df[col] = df[col].apply(lambda x: np.log(x - lower + 1))

        # --- Fit Scalers and Transform Data ---
        processed_data = df.copy()
        for i in range(self.num_vars):
            col_name = df.columns[i]
            if i in self.var_types.get("cont", []):
                if self.cont_scaler == "minmax":
                    scaler = MinMaxScaler()
                elif self.cont_scaler == "quantile":
                    scaler = QuantileTransformer(output_distribution="normal")
                elif self.cont_scaler == "standard":
                    scaler = StandardScaler()
                processed_data[col_name] = scaler.fit_transform(
                    df[[col_name]].values
                ).flatten()
                self.scalers[i] = scaler
            elif i in self.var_types.get("cat", {}):
                encoder = LabelEncoder()
                processed_data[col_name] = encoder.fit_transform(df[col_name].values)
                self.scalers[i] = encoder

        # Store the processed data as a tensor
        self.data = torch.tensor(processed_data.values, dtype=torch.float32)
        print(self.data)

