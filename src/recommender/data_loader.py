"""Data loading and preprocessing for Online Retail II recommender.

Dataset: UCI Online Retail II — ~1M UK e-commerce transactions (2009-2011).
Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

Pipeline:
    download (cached) → clean → frequency filter → log(1+qty) matrix → LOO split
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parents[2] / "data" / "raw"
_CACHE_PATH = _DATA_DIR / "online_retail_ii.parquet"
_UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

_MIN_USER_INTERACTIONS = 5
_MIN_ITEM_INTERACTIONS = 5


def _download_and_cache() -> pd.DataFrame:
    if _CACHE_PATH.exists():
        logger.info("Loading from cache: %s", _CACHE_PATH)
        return pd.read_parquet(_CACHE_PATH)

    logger.info("Downloading Online Retail II from UCI (~50 MB)...")
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(_UCI_URL, timeout=180)
    r.raise_for_status()

    with ZipFile(BytesIO(r.content)) as zf:
        xlsx_name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
        logger.info("Reading %s from zip...", xlsx_name)
        with zf.open(xlsx_name) as f:
            raw_bytes = f.read()

    buf = BytesIO(raw_bytes)
    df1 = pd.read_excel(buf, sheet_name=0, engine="openpyxl")
    buf.seek(0)
    df2 = pd.read_excel(buf, sheet_name=1, engine="openpyxl")

    df = pd.concat([df1, df2], ignore_index=True)
    # Normalise mixed-type columns so pyarrow can serialise them
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str)
    df.to_parquet(_CACHE_PATH, index=False)
    logger.info("Cached %d rows → %s", len(df), _CACHE_PATH)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid UK purchase transactions."""
    df = df.rename(columns={"Customer ID": "customer_id"})
    df = df.dropna(subset=["customer_id", "Description"])
    df["customer_id"] = df["customer_id"].astype(int)
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["Country"] == "United Kingdom"]
    return df.reset_index(drop=True)


def _frequency_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively remove low-activity users and items until stable."""
    for _ in range(5):
        n_before = len(df)
        user_counts = df.groupby("customer_id")["StockCode"].count()
        df = df[df["customer_id"].isin(
            user_counts[user_counts >= _MIN_USER_INTERACTIONS].index
        )]
        item_counts = df.groupby("StockCode")["customer_id"].count()
        df = df[df["StockCode"].isin(
            item_counts[item_counts >= _MIN_ITEM_INTERACTIONS].index
        )]
        if len(df) == n_before:
            break
    return df.reset_index(drop=True)


def build_interaction_matrix(
    df: pd.DataFrame,
) -> tuple[sp.csr_matrix, list, list, pd.DataFrame]:
    """Build log(1+quantity) implicit user-item matrix.

    Returns:
        matrix  : CSR sparse matrix (n_users × n_items)
        users   : ordered list of customer_ids
        items   : ordered list of StockCodes
        item_meta: DataFrame with StockCode, Description, Price
    """
    agg = (
        df.groupby(["customer_id", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
    )
    agg["weight"] = np.log1p(agg["Quantity"])

    users = sorted(agg["customer_id"].unique().tolist())
    items = sorted(agg["StockCode"].unique().tolist())
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {it: i for i, it in enumerate(items)}

    row = agg["customer_id"].map(user_idx).values
    col = agg["StockCode"].map(item_idx).values
    data = agg["weight"].values.astype(np.float32)

    matrix = sp.csr_matrix((data, (row, col)), shape=(len(users), len(items)))

    item_meta = (
        df.groupby("StockCode")
        .agg(
            Description=("Description", lambda x: x.mode().iloc[0]),
            Price=("Price", "mean"),
        )
        .reset_index()
    )
    item_meta = item_meta[item_meta["StockCode"].isin(items)].reset_index(drop=True)

    logger.info(
        "Interaction matrix: %d users × %d items  density=%.3f%%",
        len(users), len(items),
        100.0 * matrix.nnz / (len(users) * len(items)),
    )
    return matrix, users, items, item_meta


def leave_one_out_split(
    df: pd.DataFrame,
    matrix: sp.csr_matrix,
    users: list,
    items: list,
) -> tuple[sp.csr_matrix, dict[int, int]]:
    """Hold out each user's last purchase (by InvoiceDate) as ground truth.

    Returns:
        train_matrix: CSR with the held-out entry zeroed
        test_items  : {user_row_idx → item_col_idx}
    """
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {it: i for i, it in enumerate(items)}

    last_per_user = (
        df.sort_values("InvoiceDate")
        .groupby("customer_id")[["StockCode"]]
        .last()
        .reset_index()
    )

    train = matrix.tolil()
    test_items: dict[int, int] = {}

    for _, row_data in last_per_user.iterrows():
        u, it = int(row_data["customer_id"]), str(row_data["StockCode"])
        if u not in user_idx or it not in item_idx:
            continue
        u_i, it_i = user_idx[u], item_idx[it]
        # Require ≥2 training interactions so the user isn't invisible at test time
        if matrix.getrow(u_i).nnz < 2:
            continue
        test_items[u_i] = it_i
        train[u_i, it_i] = 0.0

    train_csr = sp.csr_matrix(train)
    train_csr.eliminate_zeros()
    logger.info("LOO split: %d test users", len(test_items))
    return train_csr, test_items


def load_data() -> tuple[
    sp.csr_matrix, sp.csr_matrix, dict[int, int], list, list, pd.DataFrame
]:
    """Full pipeline: download → clean → filter → matrix → LOO split.

    Returns:
        train_matrix, full_matrix, test_items, users, items, item_meta
    """
    raw = _download_and_cache()
    df = _clean(raw)
    df = _frequency_filter(df)
    logger.info(
        "After filtering: %d rows | %d users | %d items",
        len(df), df["customer_id"].nunique(), df["StockCode"].nunique(),
    )
    matrix, users, items, item_meta = build_interaction_matrix(df)
    train_matrix, test_items = leave_one_out_split(df, matrix, users, items)
    return train_matrix, matrix, test_items, users, items, item_meta
