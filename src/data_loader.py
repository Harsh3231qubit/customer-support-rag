"""
Loads the Bitext customer-support dataset and cleans it into a
document-ready DataFrame.
"""

import re
import pandas as pd
from datasets import load_dataset

from src.config import DATASET_NAME


def clean_text(text: str) -> str:
    """Strip template placeholders, stray HTML, and extra whitespace."""
    text = re.sub(r"\{\{.*?\}\}", "", text)   # remove {{Order Number}}-style placeholders
    text = re.sub(r"<.*?>", "", text)          # strip stray HTML tags
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def load_and_clean() -> pd.DataFrame:
    """
    Loads the Bitext dataset, keeps only the columns needed for the
    document store, cleans text, and removes empty/duplicate rows.
    """
    ds = load_dataset(DATASET_NAME)["train"]
    df = ds.to_pandas()[["response", "intent", "category"]]

    df["response"] = df["response"].apply(clean_text)
    df = df[df["response"].str.len() > 10]
    df = df.drop_duplicates(subset="response")
    df = df.reset_index(drop=True)

    return df
