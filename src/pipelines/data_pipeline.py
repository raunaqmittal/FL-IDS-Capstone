import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.configs.paths import RAW_DIR, PREPROCESSED_DIR, ensure_dirs
from src.components.data.data_loader import load_cicids2017
from src.components.data.data_preprocessor import preprocess
from src.components.data.data_partitioner import run_partitioning


def run_data_pipeline():
    try:
        logging.info("Data pipeline started")
        ensure_dirs()

        # Load
        df_raw = load_cicids2017()

        # Save raw data
        raw_path = RAW_DIR / "cicids2017_raw.parquet"
        df_raw.to_parquet(raw_path, index=False)
        logging.info(f"Raw data saved → {raw_path}")

        # Step 1–5: preprocess
        df, feature_cols, le = preprocess(df_raw)

        # Save preprocessed data + metadata
        pre_path = PREPROCESSED_DIR / "cicids2017_preprocessed.parquet"
        df.to_parquet(pre_path, index=False)
        logging.info(f"Preprocessed data saved → {pre_path}")

        # Save label encoder and feature list
        with open(PREPROCESSED_DIR / "label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        with open(PREPROCESSED_DIR / "feature_cols.pkl", "wb") as f:
            pickle.dump(feature_cols, f)

        X = df[feature_cols].values.astype("float32")
        y = df["Label"].values.astype("int64")

        # Step 7: global 80/20 stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Step 6: fit scaler on train only (no data leakage)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save scaler + test set for server-side evaluation
        with open(PREPROCESSED_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        np.savez_compressed(PREPROCESSED_DIR / "test_set.npz", X=X_test, y=y_test)
        logging.info(f"Scaler and test set saved → {PREPROCESSED_DIR}")

        # Step 8: partition training data across clients
        run_partitioning(X_train, y_train)

        logging.info("Data pipeline completed")
        return feature_cols, le, scaler, X_test, y_test

    except Exception as e:
        raise FLIDSException(e, sys)