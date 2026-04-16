import sys
import pandas as pd
from datasets import load_dataset
from src.logging.logger import logging
from src.exception.exception import FLIDSException


def load_cicids2017() -> pd.DataFrame:
    try:
        logging.info("Loading CIC-IDS2017 from HuggingFace")
        ds = load_dataset("bvk/CICIDS-2017")
        df = ds["train"].to_pandas()
        logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise FLIDSException(e, sys)