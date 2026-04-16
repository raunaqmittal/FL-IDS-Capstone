import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.logging.logger import logging
from src.exception.exception import FLIDSException

DROP_COLS = ["Src IP dec", "Dst IP dec", "Timestamp", "Attempted Category"]

def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def encode_labels(df: pd.DataFrame):
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])
    return df, le

def scale_features(df: pd.DataFrame, feature_cols: list):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def preprocess(df: pd.DataFrame):
    try:
        logging.info("Starting preprocessing")

        df = drop_irrelevant(df)
        df = clean(df)

        feature_cols = [c for c in df.columns if c != "Label"]

        df, le = encode_labels(df)
        df, scaler = scale_features(df, feature_cols)

        logging.info(f"Preprocessing done: {df.shape[0]} rows, {len(feature_cols)} features, {len(le.classes_)} classes")
        return df, feature_cols, le, scaler

    except Exception as e:
        raise FLIDSException(e, sys)