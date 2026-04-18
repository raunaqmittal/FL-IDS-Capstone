import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.logging.logger import logging
from src.exception.exception import FLIDSException

DROP_COLS = ["Src IP dec", "Dst IP dec", "Timestamp", "Attempted Category"]


def drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(axis=1, how="all")                              # Step 1: remove all-NaN cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[num_cols + ["Label"]]                                 # keep numeric + label only


def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)                     # Step 2a: Inf → NaN
    df = df.fillna(df.median(numeric_only=True))                   # Step 2b: NaN → median
    return df


def variance_filter(df: pd.DataFrame, feature_cols: list) -> list:
    return [c for c in feature_cols if df[c].var() > 0]            # Step 3: drop zero-variance


def correlation_filter(df: pd.DataFrame, feature_cols: list, threshold: float = 0.95) -> list:
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {c for c in upper.columns if (upper[c] > threshold).any()}
    return [c for c in feature_cols if c not in drop]              # Step 4: Pearson pruning


def encode_labels(df: pd.DataFrame):
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])                    # Step 5: LabelEncoder
    return df, le


def preprocess(df: pd.DataFrame):
    try:
        logging.info("Starting preprocessing")

        df = drop_unusable(df)
        df = impute(df)

        feature_cols = [c for c in df.columns if c != "Label"]
        feature_cols = variance_filter(df, feature_cols)
        feature_cols = correlation_filter(df, feature_cols)

        df, le = encode_labels(df)

        logging.info(f"Preprocessing done: {df.shape[0]} rows, {len(feature_cols)} features, {len(le.classes_)} classes")
        return df, feature_cols, le

    except Exception as e:
        raise FLIDSException(e, sys)