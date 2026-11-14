"""Turn the newest raw pulls into one long-format file of full weekly series (no segmentation).

Usage (from the repository root):
    python -m respiratory.build_series

Outputs in respiratory/data/:
    series_long.csv.gz        one row per series per week (see SERIES_COLUMNS in common.py)
    series_index.csv          one row per series: coverage and basic counts
    nssp_hsa_county_crosswalk.csv   counties that make up each NSSP Health Service Area

All dates are week-ending Saturdays (MMWR weeks, Sunday to Saturday). Missing weeks are kept as
empty values, never filled. Nothing here depends on how outbreaks are later split.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .common import DATA, NAME_TO_FIPS, SERIES_COLUMNS, STATES, latest_raw, make_series

HUB_TARGET = {"COVID-19": "wk inc covid hosp", "INFLUENZA": "wk inc flu hosp", "RSV": "wk inc rsv hosp"}


def mmwr_week_end(epiweek: int) -> dt.date:
    """Saturday that ends MMWR week YYYYWW (week 1 is the Sunday-Saturday week containing Jan 4)."""
    year, week = divmod(int(epiweek), 100)
    jan4 = dt.date(year, 1, 4)
    start = jan4 - dt.timedelta(days=jan4.isoweekday() % 7)
    return start + dt.timedelta(days=6 + 7 * (week - 1))


def build_jhu() -> pd.DataFrame:
    """Daily cumulative counts -> weekly new counts at county, state and national level.

    Negative weekly counts (from downward corrections of the cumulative series) are set to 0.
    The partial first and last weeks are dropped.
    """
    out = []
    for kind, event in (("confirmed", "CASES"), ("deaths", "DEATHS")):
        raw, as_of = latest_raw("jhu_csse", f"{kind}_US")
        date_cols = [c for c in raw.columns if c[0].isdigit() and "/" in c]
        cum = raw[date_cols].astype(float)
        cum.columns = pd.to_datetime(date_cols, format="%m/%d/%y")
        meta = raw[["FIPS", "Admin2", "Province_State"]].copy()
        meta["state"] = meta["Province_State"].str.upper()
        keep = meta["state"].isin(NAME_TO_FIPS)  # drops cruise ships
        cum, meta = cum[keep], meta[keep]
        fips = pd.to_numeric(meta["FIPS"], errors="coerce")
        is_county = (meta["Admin2"].notna() & fips.notna() & (fips % 1000 > 0) & (fips % 1000 < 900)
                     & ~meta["Admin2"].fillna("").str.startswith(("Out of", "Unassigned")))

        def weekly(c: pd.DataFrame) -> pd.DataFrame:
            w = c.T.diff().iloc[1:].resample("W-SAT").sum().iloc[1:-1]  # full Sun-Sat weeks only
            return w.clip(lower=0).T

        levels = []
        cw = weekly(cum[is_county])
        cm = meta[is_county]
        cw.index = pd.MultiIndex.from_arrays([
            fips[is_county].astype(int).astype(str).str.zfill(5),
            cm["Admin2"].str.upper() + ", " + cm["state"]])
        levels.append(("county", cw))
        sw = weekly(cum.groupby(meta["state"]).sum())
        sw.index = pd.MultiIndex.from_arrays([sw.index.map(NAME_TO_FIPS), sw.index])
        levels.append(("state", sw))
        nw = weekly(cum.sum().to_frame().T)
        nw.index = pd.MultiIndex.from_tuples([("US", "US")])
        levels.append(("national", nw))
        for geo, w in levels:
            w.index.names = ["location_fips", "location"]
            w.columns.name = "date"
            long = w.stack().rename("value").reset_index()
            out.append(make_series(long, source="JHU_CSSE", disease="COVID-19", event=event,
                                   geo_level=geo, as_of=as_of))
    return pd.concat(out, ignore_index=True)


def build_nhsn() -> pd.DataFrame:
    raw, as_of = latest_raw("nhsn", "hrd_weekly")
    j = raw["jurisdiction"]
    raw["geo_level"] = np.select([j == "USA", j.str.startswith("Region")], ["national", "hhs_region"], "state")
    raw["location"] = j.map(lambda x: "US" if x == "USA" else x.upper().replace("REGION", "HHS REGION")
                            if x.startswith("Region") else STATES[x][1])
    raw["location_fips"] = j.map(lambda x: "US" if x == "USA" else "HHS" + x.split()[-1]
                                 if x.startswith("Region") else STATES[x][0])
    raw["date"] = raw["weekendingdate"].str[:10]
    out = []
    for col, disease in (("totalconfc19newadm", "COVID-19"), ("totalconfflunewadm", "INFLUENZA"),
                         ("totalconfrsvnewadm", "RSV")):
        d = raw.rename(columns={col: "value"})
        out.append(make_series(d, source="NHSN", disease=disease, event="HOSPITALIZATIONS",
                               target=HUB_TARGET[disease], as_of=as_of))
    return pd.concat(out, ignore_index=True)


def build_nssp() -> pd.DataFrame:
    """Percent of ED visits. Sub-state values are reported per Health Service Area (HSA): every county
    in an HSA carries the same numbers, so one series is kept per HSA and a county crosswalk is written."""
    raw, as_of = latest_raw("nssp", "ed_visits")
    raw["date"] = raw["week_end"].str[:10]
    raw["state"] = raw["geography"].str.upper()
    top = raw[raw["county"] == "All"].copy()
    top["geo_level"] = np.where(top["state"] == "UNITED STATES", "national", "state")
    top["location"] = top["state"].replace("UNITED STATES", "US")
    top["location_fips"] = top["location"].map(lambda x: "US" if x == "US" else NAME_TO_FIPS[x])
    sub = raw[raw["county"] != "All"].copy()
    sub["fips"] = sub["fips"].str.zfill(5)
    sub["hsa_id"] = "HSA" + sub["fips"].str[:2] + "-" + sub["hsa_nci_id"]
    xw = (sub[["hsa_id", "hsa", "hsa_nci_id", "geography", "county", "fips"]]
          .drop_duplicates().sort_values(["hsa_id", "fips"]))
    xw.to_csv(DATA / "nssp_hsa_county_crosswalk.csv", index=False)
    sub = sub.drop_duplicates(["hsa_id", "date"]).copy()
    sub["geo_level"] = "hsa"
    sub["location"] = sub["hsa"].str.upper()
    sub["location_fips"] = sub["hsa_id"]
    both = pd.concat([top, sub], ignore_index=True)
    out = []
    for col, disease in (("percent_visits_covid", "COVID-19"), ("percent_visits_influenza", "INFLUENZA"),
                         ("percent_visits_rsv", "RSV")):
        d = both.rename(columns={col: "value"})
        out.append(make_series(d, source="NSSP", disease=disease, event="PERCENT_ED_VISITS", as_of=as_of))
    return pd.concat(out, ignore_index=True)


def build_respnet() -> pd.DataFrame:
    """Weekly hospitalization rate per 100,000 (observed, all ages). 'Overall' is the whole network
    catchment, used here as the national-level series; state rows cover only that state's catchment."""
    raw, as_of = latest_raw("respnet", "weekly_rates")
    disease = {"COVID-NET": "COVID-19", "FluSurv-NET": "INFLUENZA", "RSV-NET": "RSV"}
    raw = raw[raw["surveillance_network"].isin(disease)].copy()
    raw["disease"] = raw["surveillance_network"].map(disease)
    raw["location"] = raw["state"].str.upper().replace("OVERALL", "US")
    raw["geo_level"] = np.where(raw["location"] == "US", "national", "state")
    raw["location_fips"] = raw["location"].map(lambda x: "US" if x == "US" else NAME_TO_FIPS[x])
    raw["date"] = raw["date"].str[:10]
    raw["value"] = raw["estimate"]
    return make_series(raw, source="RESP_NET", event="HOSPITALIZATION_RATE", as_of=as_of)


def build_ilinet() -> pd.DataFrame:
    """Percent of outpatient visits for ILI. Unweighted for every level; weighted also for national/HHS."""
    raw, as_of = latest_raw("ilinet", "fluview")
    r = raw["region"]
    raw["geo_level"] = np.select([r == "nat", r.str.startswith("hhs")], ["national", "hhs_region"], "state")
    special = {"nat": ("US", "US"), "jfk": ("NYC", "NEW YORK CITY"),
               "ny_minus_jfk": ("36-NYC", "NEW YORK (EXCLUDING NYC)")}

    def loc(x: str) -> tuple[str, str]:
        if x in special:
            return special[x]
        if x.startswith("hhs"):
            return "HHS" + x[3:], "HHS REGION " + x[3:]
        return STATES[x.upper()]

    raw["location_fips"] = r.map(lambda x: loc(x)[0])
    raw["location"] = r.map(lambda x: loc(x)[1])
    raw["date"] = raw["epiweek"].map(mmwr_week_end)
    out = [make_series(raw.rename(columns={"ili": "value"}), source="ILINET", disease="ILI",
                       event="PERCENT_UNWEIGHTED", as_of=as_of)]
    top = raw[raw["geo_level"] != "state"]
    out.append(make_series(top.rename(columns={"wili": "value"}), source="ILINET", disease="ILI",
                           event="PERCENT_WEIGHTED", as_of=as_of))
    return pd.concat(out, ignore_index=True)


BUILDERS = [build_jhu, build_nhsn, build_nssp, build_respnet, build_ilinet]


def regularize(df: pd.DataFrame) -> pd.DataFrame:
    """Give every series a gap-free weekly index from its first to its last observed week."""
    df = df.dropna(subset=["value"]).drop_duplicates(["series_id", "date"])
    meta = df.drop(columns=["date", "value"]).drop_duplicates("series_id").set_index("series_id")
    wide = df.pivot(index="series_id", columns="date", values="value")
    wide = wide.reindex(columns=pd.date_range(wide.columns.min(), wide.columns.max(), freq="W-SAT")
                        .strftime("%Y-%m-%d"))
    wide.columns.name = "date"
    long = wide.stack(future_stack=True).rename("value").reset_index()
    first = df.groupby("series_id")["date"].min()
    last = df.groupby("series_id")["date"].max()
    long = long[(long["date"] >= long["series_id"].map(first)) & (long["date"] <= long["series_id"].map(last))]
    return long.join(meta, on="series_id")[SERIES_COLUMNS]


def main() -> None:
    series = regularize(pd.concat([b() for b in BUILDERS], ignore_index=True))
    series = series.sort_values(["series_id", "date"]).reset_index(drop=True)
    series.to_csv(DATA / "series_long.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    index = (series.groupby(["series_id", "source", "disease", "event", "geo_level", "location",
                             "location_fips", "target", "as_of"], dropna=False)
             .agg(first_date=("date", "min"), last_date=("date", "max"), n_weeks=("date", "size"),
                  n_missing=("value", lambda v: int(v.isna().sum())), total=("value", "sum"))
             .reset_index())
    index.to_csv(DATA / "series_index.csv", index=False)
    print(index.groupby(["source", "disease", "event", "geo_level"])
          .agg(series=("series_id", "size"), first=("first_date", "min"), last=("last_date", "max")))
    print(f"{len(index):,} series, {len(series):,} weekly rows")


if __name__ == "__main__":
    main()
