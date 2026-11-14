"""Shared helpers for the respiratory IDOBE pipeline."""
from __future__ import annotations

import datetime as dt
import io
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"

# Long-format interchange schema: one row per series per week.
SERIES_COLUMNS = [
    "series_id", "source", "disease", "location", "location_fips", "geo_level",
    "event", "target", "date", "value", "as_of",
]

STATES = {
    "AL": ("01", "ALABAMA"), "AK": ("02", "ALASKA"), "AZ": ("04", "ARIZONA"), "AR": ("05", "ARKANSAS"),
    "CA": ("06", "CALIFORNIA"), "CO": ("08", "COLORADO"), "CT": ("09", "CONNECTICUT"), "DE": ("10", "DELAWARE"),
    "DC": ("11", "DISTRICT OF COLUMBIA"), "FL": ("12", "FLORIDA"), "GA": ("13", "GEORGIA"), "HI": ("15", "HAWAII"),
    "ID": ("16", "IDAHO"), "IL": ("17", "ILLINOIS"), "IN": ("18", "INDIANA"), "IA": ("19", "IOWA"),
    "KS": ("20", "KANSAS"), "KY": ("21", "KENTUCKY"), "LA": ("22", "LOUISIANA"), "ME": ("23", "MAINE"),
    "MD": ("24", "MARYLAND"), "MA": ("25", "MASSACHUSETTS"), "MI": ("26", "MICHIGAN"), "MN": ("27", "MINNESOTA"),
    "MS": ("28", "MISSISSIPPI"), "MO": ("29", "MISSOURI"), "MT": ("30", "MONTANA"), "NE": ("31", "NEBRASKA"),
    "NV": ("32", "NEVADA"), "NH": ("33", "NEW HAMPSHIRE"), "NJ": ("34", "NEW JERSEY"), "NM": ("35", "NEW MEXICO"),
    "NY": ("36", "NEW YORK"), "NC": ("37", "NORTH CAROLINA"), "ND": ("38", "NORTH DAKOTA"), "OH": ("39", "OHIO"),
    "OK": ("40", "OKLAHOMA"), "OR": ("41", "OREGON"), "PA": ("42", "PENNSYLVANIA"), "RI": ("44", "RHODE ISLAND"),
    "SC": ("45", "SOUTH CAROLINA"), "SD": ("46", "SOUTH DAKOTA"), "TN": ("47", "TENNESSEE"), "TX": ("48", "TEXAS"),
    "UT": ("49", "UTAH"), "VT": ("50", "VERMONT"), "VA": ("51", "VIRGINIA"), "WA": ("53", "WASHINGTON"),
    "WV": ("54", "WEST VIRGINIA"), "WI": ("55", "WISCONSIN"), "WY": ("56", "WYOMING"),
    "AS": ("60", "AMERICAN SAMOA"), "GU": ("66", "GUAM"), "MP": ("69", "NORTHERN MARIANA ISLANDS"),
    "PR": ("72", "PUERTO RICO"), "VI": ("78", "VIRGIN ISLANDS"),
}
NAME_TO_FIPS = {name: fips for fips, name in STATES.values()}
NAME_TO_FIPS["NEW YORK CITY"] = "NYC"  # ILINet reports NYC separately; not a FIPS state


def today() -> str:
    return dt.date.today().isoformat()


def get(url: str, params: dict | None = None, tries: int = 5, timeout: int = 300) -> requests.Response:
    """GET with retries and exponential backoff."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:  # network error
            err = str(e)
        if i == tries - 1:
            raise RuntimeError(f"GET {url} failed: {err}")
        time.sleep(2 ** (i + 1))
    raise AssertionError


def socrata_csv(dataset_id: str, select: str = "*", where: str | None = None, page: int = 50000) -> pd.DataFrame:
    """Download a data.cdc.gov (Socrata) dataset as a DataFrame of strings, paging until exhausted."""
    url = f"https://data.cdc.gov/resource/{dataset_id}.csv"
    frames, offset = [], 0
    while True:
        params = {"$select": select, "$limit": page, "$offset": offset, "$order": ":id"}
        if where:
            params["$where"] = where
        df = pd.read_csv(io.StringIO(get(url, params).text), dtype=str)
        frames.append(df)
        if len(df) < page:
            break
        offset += page
    return pd.concat(frames, ignore_index=True)


def socrata_updated(dataset_id: str) -> str:
    """Date (UTC) the dataset's rows were last updated on data.cdc.gov."""
    meta = get(f"https://data.cdc.gov/api/views/{dataset_id}.json").json()
    return dt.datetime.fromtimestamp(meta["rowsUpdatedAt"], dt.timezone.utc).date().isoformat()


def save_raw(df: pd.DataFrame, source: str, name: str, as_of: str) -> Path:
    """Write a raw pull to data/raw/<source>/<as_of>/<name>.csv.gz. Old pulls are kept (vintages)."""
    out = RAW / source / as_of
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv.gz"
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"wrote {path.relative_to(ROOT)} ({len(df):,} rows)")
    return path


def latest_raw(source: str, name: str, as_of: str | None = None) -> tuple[pd.DataFrame, str]:
    """Read the newest (or the given) pull of a raw file. Returns (frame, as_of)."""
    base = RAW / source
    vintages = sorted(p.name for p in base.iterdir() if (p / f"{name}.csv.gz").exists()) if base.exists() else []
    if not vintages:
        raise FileNotFoundError(f"no raw pull for {source}/{name}; run the fetch script first")
    as_of = as_of or vintages[-1]
    return pd.read_csv(base / as_of / f"{name}.csv.gz", dtype=str), as_of


def make_series(df: pd.DataFrame, **const) -> pd.DataFrame:
    """Fill constants, build series_id and return the frame in SERIES_COLUMNS order."""
    df = df.copy()
    for k, v in const.items():
        df[k] = v
    if "target" not in df:
        df["target"] = ""
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = (df["source"] + "|" + df["disease"] + "|" + df["event"] + "|"
                       + df["geo_level"] + "|" + df["location_fips"].astype(str))
    return df[SERIES_COLUMNS]
