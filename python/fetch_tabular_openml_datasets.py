"""Fetch and preprocess 8 tabular OpenML benchmarks -> real_data/tabular/df_*.csv."""
'''
All of the features are standardized, with the categorical features 
preprocessed as labelencoder.
'''
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

_PYROOT = os.path.dirname(os.path.abspath(__file__))
# os.chdir(_PYROOT)  # disabled: must not steal process cwd
TABULAR_DIR = os.environ.get("TABULAR_DATA_DIR", os.path.join("real_data", "tabular_rf_bench"))

DATASET_LIST = [
    "df_adult.csv",
    "df_house16H.csv",
    "df_elevators.csv",
    "df_pol.csv",
    "df_cpu.csv",
    "df_kin.csv",
    "df_Ail.csv",
    "df_bank32nh.csv",
]


@dataclass
class DatasetSpec:
    filename: str
    openml_name: str
    data_id: int
    task: str  # "regression" | "classification"
    target_from: str = "openml"  # "openml" | column name in X
    label_encode_cols: Sequence[str] = ()
    ordinal_encode_cols: Sequence[str] = ()
    drop_cols: Sequence[str] = ()
    drop_constant_features: bool = True


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "df_adult.csv": DatasetSpec(
        filename="df_adult.csv",
        openml_name="adult",
        data_id=1590,
        task="regression",
        target_from="fnlwgt",
        label_encode_cols=(
            "workclass",
            "sex",
            "relationship",
            "race",
            "native-country",
            "occupation",
            "marital-status",
        ),
        ordinal_encode_cols=("education",),
    ),
    "df_house16H.csv": DatasetSpec(
        filename="df_house16H.csv",
        openml_name="house_16H",
        data_id=574,
        task="regression",
    ),
    "df_elevators.csv": DatasetSpec(
        filename="df_elevators.csv",
        openml_name="elevators",
        data_id=1509,
        task="classification",
    ),
    "df_pol.csv": DatasetSpec(
        filename="df_pol.csv",
        openml_name="pol",
        data_id=722,
        task="classification",
        drop_constant_features=True,
    ),
    "df_cpu.csv": DatasetSpec(
        filename="df_cpu.csv",
        openml_name="cpu_act",
        data_id=562,
        task="regression",
    ),
    "df_kin.csv": DatasetSpec(
        filename="df_kin.csv",
        openml_name="kin8nm",
        data_id=189,
        task="regression",
    ),
    "df_Ail.csv": DatasetSpec(
        filename="df_Ail.csv",
        openml_name="Ailerons",
        data_id=712,
        task="regression",
    ),
    "df_bank32nh.csv": DatasetSpec(
        filename="df_bank32nh.csv",
        openml_name="bank32nh",
        data_id=573,
        task="regression",
    ),
}


def _is_categorical(series: pd.Series) -> bool:
    return (
        series.dtype == object
        or str(series.dtype) == "string"
        or str(series.dtype) == "category"
    )


def _encode_categorical_columns(
    frame: pd.DataFrame,
    label_encode_cols: Sequence[str],
    ordinal_encode_cols: Sequence[str],
) -> pd.DataFrame:
    out = frame.copy()
    manual_label = set(label_encode_cols)
    manual_ordinal = set(ordinal_encode_cols)

    for col in out.columns:
        if col in manual_label:
            out[col] = LabelEncoder().fit_transform(out[col].astype(str))
            continue
        if col in manual_ordinal:
            out[col] = OrdinalEncoder().fit_transform(out[[col]].astype(str))
            continue
        if not _is_categorical(out[col]):
            continue
        n_unique = out[col].nunique(dropna=False)
        if n_unique < 3:
            out[col] = LabelEncoder().fit_transform(out[col].astype(str))
        else:
            out[col] = OrdinalEncoder().fit_transform(out[[col]].astype(str))
    return out


def _scale_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        vals = pd.to_numeric(out[col], errors="coerce")
        vals = vals.fillna(vals.median()).fillna(0.0)
        out[col] = StandardScaler().fit_transform(vals.to_frame())
    return out


def _prepare_target(
    y: pd.Series,
    task: str,
) -> pd.Series:
    y = y.copy()
    if _is_categorical(y):
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="Y")
        return y
    y = pd.to_numeric(y, errors="coerce")
    y = y.fillna(y.median()).fillna(0.0)
    if task == "regression":
        y = StandardScaler().fit_transform(y.to_frame()).ravel()
    y = pd.Series(y, name="Y")
    return y


def build_dataset(spec: DatasetSpec) -> pd.DataFrame:
    bundle = fetch_openml(
        data_id=spec.data_id,
        as_frame=True,
        parser="pandas",
    )
    X = pd.DataFrame(bundle.data).copy()

    if spec.target_from == "openml":
        y = bundle.target
    else:
        if spec.target_from not in X.columns:
            raise KeyError(
                f"{spec.filename}: target column {spec.target_from!r} not in features"
            )
        y = X[spec.target_from]
        X = X.drop(columns=[spec.target_from])

    for col in spec.drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    if spec.drop_constant_features:
        const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if const_cols:
            X = X.drop(columns=const_cols)

    X = _encode_categorical_columns(
        X,
        label_encode_cols=spec.label_encode_cols,
        ordinal_encode_cols=spec.ordinal_encode_cols,
    )

    for col in X.columns:
        if not _is_categorical(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X = _scale_columns(X)

    y = _prepare_target(y, spec.task)
    out = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return out


def save_all(force: bool = False) -> None:
    os.makedirs(TABULAR_DIR, exist_ok=True)
    for filename in DATASET_LIST:
        spec = DATASET_SPECS[filename]
        path = os.path.join(TABULAR_DIR, spec.filename)
        if (
            not force
            and os.path.exists(path)
            and os.path.getsize(path) > 0
            and filename != "df_Ail.csv"
        ):
            print(f"[tabular] skip existing {path}", flush=True)
            continue
        print(
            f"[tabular] fetch {spec.openml_name} ({spec.data_id}) -> {path}",
            flush=True,
        )
        df = build_dataset(spec)
        df.to_csv(path, index=False)
        print(
            f"[tabular] saved {path} shape={df.shape} task={spec.task}",
            flush=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite all CSVs",
    )
    args = parser.parse_args()
    save_all(force=args.force)
