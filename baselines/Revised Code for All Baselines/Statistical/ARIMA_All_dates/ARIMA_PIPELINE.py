import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA


class FixedARIMAProcessor:
    def __init__(self):
        self.forecasts = []
        self.eval_pairs = []
        self.unique_ids = []
        self.dates = []

        self.maes = []
        self.mses = []
        self.mapes = []
        self.nmses = []

        self.metrics_df = pd.DataFrame(columns=["Reference Date", "MAE", "MSE", "MAPE", "NMSE"])
        self.metrics_display_df = pd.DataFrame(
            columns=["Unique_id", "Reference Date", "Target End Date", "GT", "MAE", "MSE", "MAPE", "NMSE"])
        self.display_df = pd.DataFrame(
            columns=["Unique_id", "Reference Date", "Target End Date", "GT", "Quantile", "Prediction"])
        self.color_mapping = {}

    def create_fixed_model(self, df_long, h, window, freq="W-SAT", level=[80, 95]):

        df_fit = df_long.groupby("unique_id").apply(lambda g: g.iloc[0:window]).reset_index(drop=True)
        df_truth = df_long.groupby("unique_id").apply(lambda g: g.iloc[window:window + h]).reset_index(drop=True)

        start = time.time()
        self.sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq=freq, n_jobs=-1)
        self.sf.fit(df_fit)
        print(f"ARIMA fit time: {time.time() - start:.2f} sec")

        forecast = self.sf.predict(h=h, level=level)
        forecast.set_index(["unique_id", "ds"], inplace=True)
        df_truth.set_index(["unique_id", "ds"], inplace=True)

        print("Processing forecasts per series...")
        for uid in tqdm(df_fit["unique_id"].unique(), desc="Fitting per series"):
            f = forecast.loc[uid].copy()
            f["unique_id"] = uid
            t = df_truth.loc[uid]
            self.forecasts.append(f)
            self.eval_pairs.append((f, t))
            self.dates.append(df_fit[df_fit["unique_id"] == uid]["ds"].max().strftime("%Y-%m-%d"))
            self.unique_ids.append(uid)

    def calculate_metrics(self):
        for forecast_df, truth_df in self.eval_pairs:
            y_true = truth_df.iloc[:, 0]
            y_pred = forecast_df.iloc[:, 0]

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            nmse = mse / np.var(y_true)

            self.maes.append(mae)
            self.mses.append(mse)
            self.mapes.append(mape)
            self.nmses.append(nmse)

    def calculate_pointwise_metrics(self):
        records = []
        print("Calculating pointwise metrics...")

        for i in tqdm(range(len(self.eval_pairs)), desc="Computing pointwise errors"):
            forecast_df = self.forecasts[i]
            truth_df = self.eval_pairs[i][1]
            reference_date = self.dates[i]
            unique_id = self.unique_ids[i]

            pred_series = forecast_df["AutoARIMA-hi-50"]
            true_series = truth_df.iloc[:, 0]

            for target_date in pred_series.index:
                y_pred = pred_series.loc[target_date]
                y_true = true_series.loc[target_date]

                mae = abs(y_pred - y_true)
                mse = (y_pred - y_true) ** 2
                mape = abs((y_pred - y_true) / y_true) if y_true != 0 else np.nan
                nmse = mse / np.var(true_series) if np.var(true_series) != 0 else np.nan

                records.append({
                    "Unique_id": unique_id,
                    "Reference Date": reference_date,
                    "Target End Date": target_date,
                    "GT": y_true,
                    "MAE": mae,
                    "MSE": mse,
                    "MAPE": mape,
                    "NMSE": nmse
                })

        self.metrics_display_df = pd.DataFrame(records)

    def create_display_df(self):
        records = []
        print("Generating display DataFrame...")
        for i in tqdm(range(len(self.forecasts)), desc="Building display_df"):
            forecast_df = self.forecasts[i]
            reference_date = self.dates[i]
            unique_id = self.unique_ids[i]
            truth_series = self.eval_pairs[i][1].iloc[:, 0]

            for col in forecast_df.columns:
                if col == "unique_id":
                    continue
                if "lo" in col or "hi" in col:
                    number = int(col.split("-")[-1])
                    alpha = 1 - (number / 100)
                    quantile = 1 - (alpha / 2) if "hi" in col else alpha / 2
                elif col == "AutoARIMA":
                    quantile = 0.5
                else:
                    continue

                preds = forecast_df[col]
                for idx, pred in preds.items():
                    records.append({
                        "Unique_id": unique_id,
                        "Reference Date": reference_date,
                        "Target End Date": idx,
                        "GT": truth_series.get(idx, np.nan),
                        "Quantile": quantile,
                        "Prediction": pred
                    })

        self.display_df = pd.DataFrame(records).sort_values(
            by=["Unique_id", "Reference Date", "Target End Date", "GT", "Quantile"]
        ).reset_index(drop=True)

    def compute_wis(self):
        display_df = self.display_df.sort_values(by=["Unique_id", "Reference Date", "Target End Date", "Quantile"])
        records = []

        print("Computing WIS for each forecasted point...")
        grouped = display_df.groupby(["Unique_id", "Reference Date", "Target End Date"])

        for (uid, ref_date, tgt_date), group in tqdm(grouped, desc="Computing WIS"):
            gt = group["GT"].values[0]
            predictions = group.set_index("Quantile")["Prediction"]
            quantiles = sorted(predictions.index)

            pairs = list(zip(quantiles[:len(quantiles) // 2], quantiles[::-1][:len(quantiles) // 2]))
            wis_components = []

            for lo_q, hi_q in pairs:
                lo_pred = predictions.get(lo_q, np.nan)
                hi_pred = predictions.get(hi_q, np.nan)
                alpha = hi_q - lo_q

                interval_score = (
                        (hi_pred - lo_pred)
                        + (2 / alpha) * (lo_pred - gt) * (gt < lo_pred)
                        + (2 / alpha) * (gt - hi_pred) * (gt > hi_pred)
                )
                wis_components.append(interval_score)

            ae = abs(gt - predictions[0.5]) if 0.5 in predictions else np.nan
            wis = (ae + sum(wis_components)) / (1 + len(wis_components))

            records.append({
                "Unique_id": uid,
                "Reference Date": ref_date,
                "Target End Date": tgt_date,
                "GT": gt,
                "WIS": wis
            })

        return pd.DataFrame(records)


# Load wide-format data
df_wide = pd.read_csv("outbreaks_disease_location.csv")
value_cols = [str(i) for i in range(60)]
start_dates = pd.to_datetime(df_wide["start_date"])
series_values = df_wide[value_cols].astype(float).fillna(0)

# Convert to long format
records = []
for i, (start_date, row) in enumerate(zip(start_dates, series_values.values), start=1):
    adjusted_start = start_date - pd.Timedelta(weeks=4)
    dates = pd.date_range(start=adjusted_start, periods=60, freq="W-SAT")
    for t, value in enumerate(row):
        records.append({"unique_id": f"Y_{i}", "ds": dates[t], "y": value})

df_long = pd.DataFrame(records)

print(sys.argv[1])
window = int(sys.argv[1])

processor = FixedARIMAProcessor()
processor.create_fixed_model(df_long=df_long, h=4, window=window, freq="W-SAT", level=[10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95])
processor.calculate_pointwise_metrics()
processor.create_display_df()
wis_df = processor.compute_wis()
final_result = pd.merge(
    wis_df,
    processor.metrics_display_df,
    on=["Unique_id", "Reference Date", "Target End Date", "GT"],
    how="inner"
)

final_result.to_csv('final_result_ARIMA_{}.csv'.format(window))
