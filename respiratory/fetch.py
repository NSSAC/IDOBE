"""Download raw respiratory surveillance data. No processing happens here.

Usage (from the repository root):
    python -m respiratory.fetch                 # all sources
    python -m respiratory.fetch nhsn nssp       # selected sources

Each pull is written to respiratory/data/raw/<source>/<as_of>/ where <as_of> is the date the
source says it was last updated. Earlier pulls are never overwritten, so ground truth can be vintaged.
"""
from __future__ import annotations

import io
import sys

import pandas as pd

from .common import STATES, get, save_raw, socrata_csv, socrata_updated

JHU_BASE = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
            "csse_covid_19_data/csse_covid_19_time_series/")
JHU_AS_OF = "2023-03-10"  # the JHU CSSE repository stopped updating on this date

NHSN_ID = "ua7e-t2fy"     # Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction
NSSP_ID = "rdmq-nq56"     # NSSP ED Visit Trajectories by State and Sub State Regions
RESPNET_ID = "kvib-3txy"  # RESP-NET Rates and Clinical Data (COVID-NET, FluSurv-NET, RSV-NET)
DELPHI_FLUVIEW = "https://api.delphi.cmu.edu/epidata/fluview/"  # mirror of CDC ILINet (FluView)


def fetch_jhu() -> None:
    """County-level cumulative COVID-19 cases and deaths, daily, 2020-01-22 to 2023-03-09."""
    for kind in ("confirmed", "deaths"):
        text = get(JHU_BASE + f"time_series_covid19_{kind}_US.csv").text
        save_raw(pd.read_csv(io.StringIO(text), dtype=str), "jhu_csse", f"{kind}_US", JHU_AS_OF)


def fetch_nhsn() -> None:
    """Weekly new hospital admissions for COVID-19, influenza and RSV: states, HHS regions, national."""
    cols = "weekendingdate,jurisdiction,totalconfc19newadm,totalconfflunewadm,totalconfrsvnewadm"
    save_raw(socrata_csv(NHSN_ID, select=cols), "nhsn", "hrd_weekly", socrata_updated(NHSN_ID))


def fetch_nssp() -> None:
    """Weekly percent of emergency department visits: national, state and sub-state (HSA, listed by county)."""
    cols = ("week_end,geography,county,fips,hsa,hsa_nci_id,percent_visits_covid,"
            "percent_visits_influenza,percent_visits_rsv")
    save_raw(socrata_csv(NSSP_ID, select=cols), "nssp", "ed_visits", socrata_updated(NSSP_ID))


def fetch_respnet() -> None:
    """Weekly lab-confirmed hospitalization rates per 100,000, overall population, by network and site."""
    where = ("date_type='Week Ending Date' AND data_type='Weekly Rate' AND rate_type='Observed' "
             "AND age_category='Overall' AND sex='All' AND race='All'")
    save_raw(socrata_csv(RESPNET_ID, where=where), "respnet", "weekly_rates", socrata_updated(RESPNET_ID))


def fetch_ilinet() -> None:
    """Weekly outpatient ILI percentages from ILINet via the Delphi Epidata fluview endpoint."""
    regions = (["nat"] + [f"hhs{i}" for i in range(1, 11)] + [s.lower() for s in STATES]
               + ["jfk", "ny_minus_jfk"])
    cols = ["release_date", "region", "issue", "epiweek", "num_ili", "num_patients", "num_providers", "wili", "ili"]
    frames = []
    for region in regions:
        out = get(DELPHI_FLUVIEW, {"regions": region, "epiweeks": "199740-203053"}).json()
        if out.get("result") == 1:
            frames.append(pd.DataFrame(out["epidata"])[cols])
        elif out.get("result") != -2:  # -2 means "no results" (e.g. territories that do not report)
            raise RuntimeError(f"fluview {region}: {out.get('message')}")
    df = pd.concat(frames, ignore_index=True)
    save_raw(df, "ilinet", "fluview", str(df["release_date"].dropna().max()))


SOURCES = {"jhu": fetch_jhu, "nhsn": fetch_nhsn, "nssp": fetch_nssp, "respnet": fetch_respnet, "ilinet": fetch_ilinet}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SOURCES):
        print(f"== {name}")
        SOURCES[name]()
