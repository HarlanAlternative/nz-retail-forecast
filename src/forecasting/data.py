"""Data fetching, parsing, validation, and caching for NZ retail forecasting.

Data sources:
  - Retail sales    : Stats NZ Retail Trade Survey quarterly CSV (RTTQ.SF11CA)
  - CPI (monthly)   : Stats NZ Aotearoa Data Explorer API (DF_CPI315601) → quarterly
  - Unemployment    : IMF DataMapper API — annual NZ LUR (1980+) → quarterly interpolation
  - Interest rate   : OECD MEI_FIN — NZ 3-month interbank rate, monthly → quarterly mean
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import requests
from pandera.pandas import Check, Column, DataFrameSchema

from forecasting.config import get_api_key, load_config, resolve_path

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0

# ---------------------------------------------------------------------------
# Pandera schemas
# ---------------------------------------------------------------------------

RETAIL_SCHEMA = DataFrameSchema(
    {
        "date": Column(
            pa.dtypes.DateTime,
            checks=Check(
                lambda s: s <= pd.Timestamp(datetime.now(UTC).replace(tzinfo=None)),
                element_wise=False,
                error="date column contains future dates",
            ),
            nullable=False,
        ),
        "retail_sales": Column(
            float,
            checks=[
                Check.greater_than(0, error="retail_sales must be positive"),
                Check.less_than(1e9, error="retail_sales implausibly large (>1B NZD)"),
            ],
            nullable=False,
        ),
    },
    coerce=True,
    strict=False,
)

MERGED_SCHEMA = DataFrameSchema(
    {
        "date": Column(pa.dtypes.DateTime, nullable=False),
        "retail_sales": Column(float, checks=Check.greater_than(0), nullable=False),
        "cpi": Column(float, checks=Check.greater_than(0), nullable=True),
        "unemployment_rate": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than(30)],
            nullable=True,
        ),
        "interest_rate_90d": Column(
            float,
            checks=[Check.greater_than_or_equal_to(0), Check.less_than(50)],
            nullable=True,
        ),
        "employment_count": Column(float, checks=Check.greater_than(0), nullable=True),
    },
    coerce=True,
    strict=False,
)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _http_get(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> requests.Response:
    """GET with retry/back-off. Raises RuntimeError after exhausting retries."""
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", _RETRY_BACKOFF * attempt))
                logger.warning("Rate limited — sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (requests.HTTPError, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                sleep = _RETRY_BACKOFF ** attempt
                logger.warning("Request failed (%d/%d): %s — retry in %.1fs",
                               attempt, _MAX_RETRIES, exc, sleep)
                time.sleep(sleep)
    raise RuntimeError(f"All {_MAX_RETRIES} attempts failed for {url}: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Stats NZ period parser  (format: YYYY.MM where MM = quarter-end month)
# ---------------------------------------------------------------------------

def _parse_statsnz_period(period: str) -> pd.Timestamp:
    """Convert Stats NZ period string 'YYYY.MM' to quarter-end date."""
    parts = str(period).split(".")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 12
    return pd.Timestamp(year=year, month=month, day=1)


# ---------------------------------------------------------------------------
# Stats NZ CSV client
# ---------------------------------------------------------------------------

class StatsNZCSVClient:
    """Downloads and parses Stats NZ quarterly CSV/zip releases.

    No API key required — all data is publicly downloadable.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        cfg = load_config()
        self._cache_dir: Path = cache_dir or resolve_path(cfg["data"]["raw_data_path"])
        self._cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "NZ-Economic-Forecasting/0.1"})

    # ------------------------------------------------------------------
    # Retail sales (quarterly CSV)
    # ------------------------------------------------------------------

    def fetch_retail_sales(self, start_date: str | None = None) -> pd.DataFrame:
        """Download and parse the Retail Trade Survey quarterly CSV.

        Series: RTTQ.SF11CA — Core retail industries, current prices,
        seasonally adjusted and forward-calculated.

        Returns:
            DataFrame [date, retail_sales] validated by RETAIL_SCHEMA.
        """
        cache_key = "retail_RTTQ_SF11CA"
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Retail sales loaded from cache")
            df = cached
        else:
            url = self._cfg["data"]["retail_csv_url"]
            logger.info("Fetching retail trade CSV …")
            resp = _http_get(self._session, url)
            raw = pd.read_csv(StringIO(resp.text))
            series_ref = self._cfg["data"]["retail_series_ref"]
            df = raw[raw["Series_reference"] == series_ref].copy()
            if df.empty:
                raise ValueError(
                    f"Series '{series_ref}' not found in retail CSV. "
                    f"Available: {raw['Series_reference'].unique()[:10]}"
                )
            df["date"] = df["Period"].apply(_parse_statsnz_period)
            df["retail_sales"] = pd.to_numeric(df["Data_value"], errors="coerce")
            df = df[["date", "retail_sales"]].dropna().sort_values("date").reset_index(drop=True)
            self._save_cache(df, cache_key)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        df = RETAIL_SCHEMA.validate(df)
        logger.info("Retail sales: %d quarters (%s → %s)",
                    len(df), df["date"].min().date(), df["date"].max().date())
        return df

    # ------------------------------------------------------------------
    # HLFS unemployment rate (quarterly zip CSV)
    # ------------------------------------------------------------------

    def fetch_unemployment(self, start_date: str | None = None) -> pd.DataFrame:
        """Download and parse HLFS unemployment rate.

        Series: HLFSQ.S2A — Unemployment rate (%), seasonally adjusted,
        quarterly. Data starts Q2 1986.

        The HLFS release is published as a zip containing one or more CSVs
        that follow the standard Stats NZ column layout (Series_reference,
        Period, Data_value).

        Returns:
            DataFrame [date, unemployment_rate].
        """
        cache_key = "hlfs_unemployment_S2A"
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Unemployment loaded from cache")
            df = cached
        else:
            url = self._cfg["data"]["hlfs_zip_url"]
            series_ref = self._cfg["data"]["hlfs_series_ref"]
            logger.info("Fetching HLFS zip …")
            resp = _http_get(self._session, url)
            df = self._parse_statsnz_zip(resp.content, series_ref, "unemployment_rate")
            self._save_cache(df, cache_key)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        logger.info("Unemployment rate: %d quarters (%s → %s)",
                    len(df), df["date"].min().date(), df["date"].max().date())
        return df

    # ------------------------------------------------------------------
    # Stats NZ zip CSV parser (shared for HLFS, Building Consents, etc.)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_statsnz_zip(
        content: bytes,
        series_ref: str,
        value_col: str,
    ) -> pd.DataFrame:
        """Extract and parse a Stats NZ quarterly zip release.

        Searches all CSV files inside the zip for the given series_reference,
        then parses Period → date and Data_value → value_col.
        """
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV files found inside the Stats NZ zip archive.")

            for name in csv_names:
                try:
                    raw = pd.read_csv(
                        io.StringIO(zf.read(name).decode("utf-8", errors="replace"))
                    )
                except Exception:
                    continue

                ref_col = next(
                    (c for c in raw.columns
                     if c.strip().lower().replace(" ", "_") == "series_reference"),
                    None,
                )
                if ref_col is None:
                    continue

                sub = raw[raw[ref_col].astype(str).str.strip() == series_ref]
                if sub.empty:
                    continue

                period_col = next(
                    (c for c in raw.columns if c.strip().lower() == "period"), None
                )
                value_col_src = next(
                    (c for c in raw.columns
                     if c.strip().lower() in ("data_value", "datavalue")),
                    None,
                )
                if period_col is None or value_col_src is None:
                    continue

                sub = sub.copy()
                sub["date"] = sub[period_col].apply(
                    lambda p: _parse_statsnz_period(str(p).strip())
                )
                sub[value_col] = pd.to_numeric(sub[value_col_src], errors="coerce")
                df = (
                    sub[["date", value_col]]
                    .dropna()
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                logger.debug("Parsed %d rows for series %s from %s", len(df), series_ref, name)
                return df

        raise ValueError(
            f"Series '{series_ref}' not found in any CSV inside the zip. "
            f"Files searched: {csv_names}"
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Cache read failed (%s): %s", path, exc)
        return None

    def _save_cache(self, df: pd.DataFrame, key: str) -> None:
        path = self._cache_path(key)
        try:
            df.to_parquet(path, index=False)
            logger.debug("Cached %d rows → %s", len(df), path)
        except Exception as exc:
            logger.warning("Cache write failed (%s): %s", path, exc)


# ---------------------------------------------------------------------------
# External indicators client — OECD (interest rates) + IMF (unemployment)
# ---------------------------------------------------------------------------

class ExternalIndicatorsClient:
    """Fetches open international economic data for NZ with no authentication.

    Sources:
      - OECD MEI_FIN SDMX-JSON API — NZ 3-month interbank rate, monthly → quarterly
      - IMF DataMapper REST API     — NZ annual unemployment rate → quarterly interpolation
    """

    _OECD_MEI_FIN = (
        "https://stats.oecd.org/sdmx-json/data/MEI_FIN"
        "/NZL.IR3TBB01.ST.Q/OECD?startTime=1987"
    )
    _IMF_LUR_NZL = "https://www.imf.org/external/datamapper/api/v1/LUR/NZL"

    def __init__(self, cache_dir: Path | None = None) -> None:
        cfg = load_config()
        self._cache_dir: Path = cache_dir or resolve_path(cfg["data"]["raw_data_path"])
        self._session = requests.Session()
        # Use a generic browser UA — IMF and some OECD endpoints reject custom bot strings
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------
    # Interest rate (OECD MEI_FIN — NZ 3-month interbank, monthly → Q)
    # ------------------------------------------------------------------

    def fetch_interest_rate(self, start_date: str | None = None) -> pd.DataFrame:
        """Fetch NZ 3-month interbank rate from OECD MEI_FIN; monthly → quarterly mean.

        Returns:
            DataFrame [date, interest_rate_90d] aligned to quarter-end months.
            Coverage: typically 2018-Q2 onward; earlier periods are absent (NaN
            in merged dataset, filled by imputer).
        """
        cache_key = "oecd_nz_interest_rate"
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Interest rate loaded from cache")
            df = cached
        else:
            logger.info("Fetching NZ interest rate from OECD MEI_FIN …")
            resp = _http_get(self._session, self._OECD_MEI_FIN, timeout=90)
            df = self._parse_oecd_mei_fin(resp.json())
            self._save_cache(df, cache_key)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        logger.info("Interest rate: %d quarterly obs (%s → %s)",
                    len(df), df["date"].min().date() if len(df) else "n/a",
                    df["date"].max().date() if len(df) else "n/a")
        return df

    @staticmethod
    def _parse_oecd_mei_fin(data: dict) -> pd.DataFrame:
        """Parse OECD SDMX-JSON MEI_FIN response for NZ IR3TIB monthly series.

        Aggregates monthly → quarterly by taking the mean of the three months
        in each quarter, then aligns to the quarter-end month (Mar/Jun/Sep/Dec).
        """
        structure = data["data"]["structures"][0]
        ref_areas = [v["id"] for v in structure["dimensions"]["series"][0]["values"]]
        freqs     = [v["id"] for v in structure["dimensions"]["series"][1]["values"]]
        measures  = [v["id"] for v in structure["dimensions"]["series"][2]["values"]]
        units     = [v["id"] for v in structure["dimensions"]["series"][3]["values"]]
        time_vals = [v["id"] for v in structure["dimensions"]["observation"][0]["values"]]

        # Locate the NZL / Monthly / IR3TIB (short-term interbank) / PA series key
        try:
            nzl_i = ref_areas.index("NZL")
            m_i   = freqs.index("M")
            ir_i  = measures.index("IR3TIB")
            pa_i  = units.index("PA")
        except ValueError as exc:
            raise ValueError(f"OECD MEI_FIN structure mismatch: {exc}") from exc

        # Series key has 9 dimension slots; remaining 5 default to 0
        key = f"{nzl_i}:{m_i}:{ir_i}:{pa_i}:0:0:0:0:0"

        records: list[tuple[str, float]] = []
        for ds in data["data"]["dataSets"]:
            sv = ds.get("series", {}).get(key)
            if sv:
                for tidx_str, obs in sv.get("observations", {}).items():
                    if obs and obs[0] is not None:
                        records.append((time_vals[int(tidx_str)], float(obs[0])))

        if not records:
            raise ValueError("No NZ interest rate observations found in OECD MEI_FIN response")

        df = pd.DataFrame(records, columns=["period", "interest_rate_90d"])
        df["date"] = pd.to_datetime(df["period"])
        df = df.drop(columns=["period"]).sort_values("date").reset_index(drop=True)

        # Aggregate monthly → quarterly mean, keep quarter-end months only
        df = df.set_index("date").resample("QE").mean().reset_index()
        # Align dates to the first of each quarter-end month (matching retail dates)
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        return df.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Unemployment rate (IMF DataMapper — annual → quarterly interpolation)
    # ------------------------------------------------------------------

    def fetch_unemployment(self, start_date: str | None = None) -> pd.DataFrame:
        """Fetch NZ annual unemployment rate from IMF DataMapper; interpolate to quarterly.

        The IMF LUR (Labour Unemployment Rate) series for NZ provides annual
        data from 1980, including IMF projections to 2031.  Annual values are
        linearly interpolated to quarterly frequency so every quarter has a value.

        Returns:
            DataFrame [date, unemployment_rate] at quarterly frequency.
        """
        cache_key = "imf_nz_unemployment"
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Unemployment loaded from cache")
            df = cached
        else:
            logger.info("Fetching NZ unemployment from IMF DataMapper …")
            # IMF DataMapper rejects custom Accept headers — use a plain session
            imf_session = requests.Session()
            resp = _http_get(imf_session, self._IMF_LUR_NZL, timeout=30)
            df = self._parse_imf_lur(resp.json())
            self._save_cache(df, cache_key)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        logger.info("Unemployment: %d quarterly obs (%s → %s)",
                    len(df), df["date"].min().date() if len(df) else "n/a",
                    df["date"].max().date() if len(df) else "n/a")
        return df

    @staticmethod
    def _parse_imf_lur(data: dict) -> pd.DataFrame:
        """Parse IMF DataMapper LUR response for NZL; expand annual → quarterly.

        Annual midpoint interpolation: the value for year Y is assumed to fall
        at the mid-year point (July 1).  Linear interpolation between midpoints
        gives smooth quarterly estimates.  The series is then resampled to
        quarter-end frequency aligned to retail dates (Mar/Jun/Sep/Dec).
        """
        annual = data.get("values", {}).get("LUR", {}).get("NZL", {})
        if not annual:
            raise ValueError("No NZL LUR data in IMF DataMapper response")

        # Build annual series at mid-year
        ann_rows = []
        for year_str, val in annual.items():
            if val is not None:
                try:
                    ann_rows.append((pd.Timestamp(f"{year_str}-07-01"), float(val)))
                except ValueError:
                    continue
        ann_rows.sort()
        ann_df = pd.DataFrame(ann_rows, columns=["date", "unemployment_rate"])
        ann_df = ann_df.set_index("date").sort_index()

        # Resample to monthly and linearly interpolate between mid-year anchors
        monthly = ann_df.resample("MS").interpolate(method="linear")
        # Resample to quarterly (quarter-end), keep only Mar/Jun/Sep/Dec
        quarterly = monthly.resample("QE").mean().reset_index()
        quarterly["date"] = quarterly["date"].dt.to_period("M").dt.to_timestamp()
        quarterly.columns = ["date", "unemployment_rate"]
        return quarterly.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Cache read failed (%s): %s", path, exc)
        return None

    def _save_cache(self, df: pd.DataFrame, key: str) -> None:
        path = self._cache_path(key)
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:
            logger.warning("Cache write failed (%s): %s", path, exc)


# ---------------------------------------------------------------------------
# ADE API client (new portal: api.data.stats.govt.nz)
# ---------------------------------------------------------------------------

class ADEClient:
    """Client for the Stats NZ Aotearoa Data Explorer SDMX-JSON API.

    Requires a subscription key from portal.apis.stats.govt.nz.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._api_key = api_key or get_api_key()
        cfg = load_config()
        self._base_url = cfg["data"]["ade_base_url"].rstrip("/") + "/"
        self._cache_dir: Path = cache_dir or resolve_path(cfg["data"]["raw_data_path"])
        self._cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({
            "Ocp-Apim-Subscription-Key": self._api_key,
            "Accept": "application/json",
        })

    def fetch_cpi_monthly(self, start_date: str | None = None) -> pd.DataFrame:
        """Fetch monthly CPI food price index from ADE (DF_CPI315601).

        Returns national average across all broad regions.
        """
        cache_key = "cpi_food_monthly_ade"
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("CPI (monthly) loaded from cache")
            df = cached
        else:
            dataflow = self._cfg["data"]["ade_cpi_dataflow"]
            url = f"{self._base_url}data/{dataflow}/all"
            params: dict[str, str] = {"format": "jsondata"}
            if start_date:
                params["startPeriod"] = start_date[:7]

            logger.info("Fetching CPI food monthly from ADE API …")
            resp = _http_get(self._session, url, params=params)
            data = resp.json()
            df = self._parse_sdmx_json(data, value_col="cpi")
            df = df.groupby("date")["cpi"].mean().reset_index()
            self._save_cache(df, cache_key)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        return df

    def fetch_employment(self, start_date: str | None = None) -> pd.DataFrame:
        """Fetch quarterly LEED employment count from ADE (LEED_AP1_001)."""
        cache_key = "employment_leed_ap1_001"
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Employment loaded from cache")
            df = cached
        else:
            dataflow = self._cfg["data"]["ade_leed_dataflow"]
            url = f"{self._base_url}data/{dataflow}/all"
            params = {"format": "jsondata"}
            logger.info("Fetching LEED employment from ADE API …")
            resp = _http_get(self._session, url, params=params)
            df = self._parse_leed(resp.json())
            self._save_cache(df, cache_key)

        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        logger.info("Employment: %d quarters", len(df))
        return df

    @staticmethod
    def _parse_sdmx_json(response: dict[str, Any], value_col: str) -> pd.DataFrame:
        try:
            structure = response["data"]["structures"][0]
            datasets = response["data"]["dataSets"]
        except (KeyError, IndexError) as exc:
            raise KeyError(f"Unexpected SDMX-JSON structure: {exc}") from exc

        obs_dim = structure["dimensions"]["observation"][0]
        time_values: list[str] = [v["id"] for v in obs_dim["values"]]

        records: list[dict[str, Any]] = []
        for dataset in datasets:
            for _sk, series_data in dataset.get("series", {}).items():
                for time_idx_str, obs_vals in series_data.get("observations", {}).items():
                    t_idx = int(time_idx_str)
                    if t_idx >= len(time_values) or not obs_vals or obs_vals[0] is None:
                        continue
                    records.append({"period": time_values[t_idx], value_col: float(obs_vals[0])})

        if not records:
            raise ValueError(f"No observations found in SDMX-JSON for '{value_col}'")

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["period"])
        return df.drop(columns=["period"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _parse_leed(response: dict[str, Any]) -> pd.DataFrame:
        try:
            structure = response["data"]["structures"][0]
            datasets = response["data"]["dataSets"]
        except (KeyError, IndexError) as exc:
            raise KeyError(f"Unexpected LEED SDMX-JSON structure: {exc}") from exc

        obs_dim = structure["dimensions"]["observation"][0]
        time_values: list[str] = [v["id"] for v in obs_dim["values"]]

        period_totals: dict[str, float] = {}
        for dataset in datasets:
            for _sk, series_data in dataset.get("series", {}).items():
                for time_idx_str, obs_vals in series_data.get("observations", {}).items():
                    t_idx = int(time_idx_str)
                    if t_idx >= len(time_values) or not obs_vals or obs_vals[0] is None:
                        continue
                    period = time_values[t_idx]
                    period_totals[period] = period_totals.get(period, 0.0) + float(obs_vals[0])

        if not period_totals:
            raise ValueError("No LEED observations found in SDMX-JSON response")

        df = pd.DataFrame(
            [{"date": pd.to_datetime(p), "employment_count": v}
             for p, v in period_totals.items()]
        )
        return df.sort_values("date").reset_index(drop=True)

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Cache read failed (%s): %s", path, exc)
        return None

    def _save_cache(self, df: pd.DataFrame, key: str) -> None:
        path = self._cache_path(key)
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:
            logger.warning("Cache write failed (%s): %s", path, exc)


# ---------------------------------------------------------------------------
# Merged dataset builder
# ---------------------------------------------------------------------------

def build_merged_dataset(start_date: str | None = None) -> pd.DataFrame:
    """Fetch and merge all series into one quarterly DataFrame.

    Sources merged on quarterly date (quarter-end month, day=1):
      - retail_sales      : Stats NZ RTTQ.SF11CA (anchor, 1995 Q3 onward)
      - cpi               : ADE monthly food CPI → quarterly (2000 Q1 onward)
      - unemployment_rate : IMF annual LUR → quarterly interpolation (1980 onward)
      - interest_rate_90d : OECD MEI_FIN NZ 3-month interbank, monthly → Q (2018 onward)

    All exogenous columns are optional — missing series fall back to NaN and
    are handled by the SimpleImputer in the training pipeline.

    Returns:
        Merged DataFrame validated by MERGED_SCHEMA.
    """
    cfg = load_config()
    if start_date is None:
        start_date = cfg["data"]["start_date"]

    csv_client = StatsNZCSVClient()
    ade_client = ADEClient()
    ext_client = ExternalIndicatorsClient()

    # --- Retail (anchor) ---
    retail = csv_client.fetch_retail_sales(start_date)

    # --- CPI: ADE monthly → quarterly quarter-end alignment ---
    try:
        cpi_monthly = ade_client.fetch_cpi_monthly(start_date)
        cpi_monthly["quarter_end"] = cpi_monthly["date"].dt.month.isin([3, 6, 9, 12])
        cpi = (
            cpi_monthly[cpi_monthly["quarter_end"]][["date", "cpi"]]
            .reset_index(drop=True)
        )
        logger.info("CPI (monthly→quarterly): %d quarters", len(cpi))
    except Exception as exc:
        logger.warning("CPI fetch failed (%s) — proceeding without CPI", exc)
        cpi = pd.DataFrame(columns=["date", "cpi"])

    # --- Unemployment rate (IMF annual → quarterly) ---
    try:
        unemployment = ext_client.fetch_unemployment(start_date)
    except Exception as exc:
        logger.warning("Unemployment fetch failed (%s) — proceeding without it", exc)
        unemployment = pd.DataFrame(columns=["date", "unemployment_rate"])

    # --- Short-term interest rate (OECD monthly → quarterly) ---
    try:
        interest = ext_client.fetch_interest_rate(start_date)
    except Exception as exc:
        logger.warning("Interest rate fetch failed (%s) — proceeding without it", exc)
        interest = pd.DataFrame(columns=["date", "interest_rate_90d"])

    # --- Merge all on date ---
    merged = retail.copy()
    for exog_df, col in [
        (cpi, "cpi"),
        (unemployment, "unemployment_rate"),
        (interest, "interest_rate_90d"),
    ]:
        if not exog_df.empty and col in exog_df.columns:
            merged = merged.merge(exog_df[["date", col]], on="date", how="left")
        else:
            merged[col] = float("nan")

    # employment_count: retained for schema compatibility, always NaN
    if "employment_count" not in merged.columns:
        merged["employment_count"] = float("nan")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = MERGED_SCHEMA.validate(merged)

    # Log coverage summary for exogenous features
    for col in ("cpi", "unemployment_rate", "interest_rate_90d"):
        n_valid = merged[col].notna().sum()
        logger.info("  %-22s %3d/%d quarters (%.0f%%)",
                    col, n_valid, len(merged), 100 * n_valid / len(merged))

    logger.info(
        "Merged dataset: %d quarters (%s → %s)",
        len(merged), merged["date"].min().date(), merged["date"].max().date(),
    )
    return merged


# ---------------------------------------------------------------------------
# CLI entry point: python -m forecasting.data
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch and cache all datasets, then save merged.parquet."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()

    try:
        merged = build_merged_dataset(cfg["data"]["start_date"])
    except EnvironmentError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    out = resolve_path(cfg["data"]["processed_data_path"]) / "merged.parquet"
    merged.to_parquet(out, index=False)
    logger.info("Saved merged dataset → %s  (%d rows)", out, len(merged))
    print(merged.tail(10).to_string())


if __name__ == "__main__":
    main()
