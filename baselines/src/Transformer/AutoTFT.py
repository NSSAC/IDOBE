#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# import cufflinks as cf
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# cf.go_offline()

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from neuralforecast.auto import AutoTFT
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.core import NeuralForecast

from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from neuralforecast.losses.pytorch import MQLoss
from tqdm import tqdm


# In[2]:


# Load and process the MEASLES_ARIZONA data
df = pd.read_csv("../../../raw_data/outbreaks_disease_location.csv")
value_columns = [str(i) for i in range(60)]
series_values = df[value_columns].fillna(0).astype(float)
start_dates = pd.to_datetime(df["start_date"])

# Shuffle and split
shuffled_indices = df.sample(frac=1, random_state=42).index
split_point = int(0.8 * len(df))
train_indices = shuffled_indices[:split_point]
test_indices = shuffled_indices[split_point:]


# In[3]:


train_records = []
for i, row in series_values.iloc[train_indices].iterrows():
    dates = pd.date_range(start="2000-01-01", periods=60, freq="W-SAT")
    for t, value in enumerate(row):
        train_records.append({"unique_id": f"Y{i+1}", "ds": dates[t], "y": value})
df_train = pd.DataFrame(train_records)

test_start_dates = start_dates.loc[test_indices] - pd.Timedelta(weeks=4)
df_test_all = []

for idx in test_indices:
    start_date = test_start_dates.loc[idx]
    row = series_values.loc[idx]
    dates = pd.date_range(start=start_date, periods=60, freq="W-SAT")
    for t, value in enumerate(row):
        df_test_all.append({"unique_id": f"Y_{idx}", "ds": dates[t], "y": value})

df_test_all = pd.DataFrame(df_test_all)


# In[4]:


df_train["date"] = pd.to_datetime(df_train["ds"])
df_train.set_index("date", inplace = True)

df_test_all["date"] = pd.to_datetime(df_test_all["ds"])
df_test_all.set_index("date", inplace = True)


# In[5]:


df_test_all


# In[15]:


class FixedModelTFTProcessor:
    def __init__(self, dates=[]):
        self.dates = dates
        self.forecasts = []
        self.eval_pairs = []
        self.input_size = None
        self.nf = None
        self.config = None
        self.testset = None
        self.maes = []
        self.mses = []
        self.mapes = []
        self.nmses = []
        self.reference_dates = {}
        self.metrics_df = pd.DataFrame(columns=["Unique_id","Reference Date", "MAE", "MSE", "MAPE", "NMSE"])
        self.display_df = pd.DataFrame(columns=["Unique_id","Reference Date", "Target End Date", "GT" , "Quantile", "Prediction"])
        self.model = None
        self.selected_input_size = None
        self.unique_ids = []

    def create_fixed_model(self, df_train, df_test, test_ids, h, freq, level=[]):
        input_length = df_train.groupby("unique_id").size().min()
        max_input_size = max(8, input_length - h - 1)
        self.testset = df_test

        def config(trial):
            return {
                "input_size": trial.suggest_int("input_size", 8, max_input_size),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "random_seed": trial.suggest_int("random_seed", 1, 99999),
                "accelerator": "gpu",
                "devices": 1,
                "strategy": "auto"
            }

        self.config = config

        if not level:
            nf = NeuralForecast(models=[AutoTFT(h=h, backend="optuna", config=self.config)], freq=freq)
        else:
            nf = NeuralForecast(models=[AutoTFT(h=h, backend="optuna", loss=MQLoss(level=level), config=self.config)], freq=freq)

        nf.fit(df=df_train)
        self.nf = nf
        self.model = nf.models[0].model
        self.selected_input_size = self.model.hparams['input_size']

        for uid in tqdm(test_ids, desc="Forecasting with AutoTFT"):
            series_df = df_test[df_test["unique_id"] == uid]
            if len(series_df) <= self.selected_input_size:
                continue
            reference_date = series_df["ds"].iloc[-4]
            self.reference_dates[uid] = reference_date
            df = series_df[series_df["ds"] < reference_date]
            forecast = nf.predict(df=df).set_index("ds")
            forecast_horizon = forecast.index
            ground_truth = series_df[series_df["ds"].isin(forecast_horizon)].set_index("ds")
            self.forecasts.append(forecast)
            self.eval_pairs.append((forecast, ground_truth))
            self.dates.append(reference_date.strftime("%Y-%m-%d"))
            self.unique_ids.append(uid)

    def calculate_metrics(self):
        for forecast_df, truth_df in self.eval_pairs:
            y_true = truth_df.iloc[:, 1]
            y_pred = forecast_df.iloc[:, 0]
            self.maes.append(mean_absolute_error(y_true, y_pred))
            self.mses.append(mean_squared_error(y_true, y_pred))
            self.mapes.append(mean_absolute_percentage_error(y_true, y_pred))
            self.nmses.append(self.mses[-1] / np.var(y_true))

    def create_metrics_df(self):
        for i in range(len(self.dates)):
            self.metrics_df.loc[len(self.metrics_df)] = [
                self.unique_ids[i],
                self.dates[i],
                self.maes[i],
                self.mses[i],
                self.mapes[i],
                self.nmses[i]
            ]

    def create_display_df(self):
        records = []
        testset_indexed = self.testset.set_index(["unique_id", "ds"])

        for i, forecast_df in enumerate(tqdm(self.forecasts, desc="Building display_df")):
            uid = forecast_df["unique_id"].iloc[0]
            reference_date = self.dates[i]

            try:
                gt_series = testset_indexed.loc[uid]["y"]
            except KeyError:
                gt_series = pd.Series()

            for col in forecast_df.columns:
                if col == "unique_id":
                    continue

                if "lo" in col:
                    number = int(col.split("-")[-1])
                    alpha = 1 - (number / 100)
                    quantile = alpha / 2
                elif "hi" in col:
                    number = int(col.split("-")[-1])
                    alpha = 1 - (number / 100)
                    quantile = 1 - (alpha / 2)
                elif col in ["AutoTFT", "AutoTFT-median"]:
                    quantile = 0.5
                else:
                    continue

                df_col = forecast_df[[col]].copy()
                df_col.rename(columns={col: "Prediction"}, inplace=True)
                df_col["Unique_id"] = uid
                df_col["Reference Date"] = reference_date
                df_col["Target End Date"] = df_col.index
                df_col["GT"] = df_col.index.map(gt_series.get)
                df_col["Quantile"] = quantile
                records.append(df_col)

        self.display_df = pd.concat(records)[
            ["Unique_id", "Reference Date", "Target End Date", "GT", "Quantile", "Prediction"]
        ].sort_values(by=["Unique_id", "Reference Date", "Target End Date", "Quantile"]).reset_index(drop=True)

    def efficient_compute_wis(self):
        df = self.display_df.sort_values(by=["Unique_id", "Reference Date", "Target End Date", "Quantile"])
        results = []
        grouped = df.groupby(["Unique_id", "Reference Date", "Target End Date"])
        for (uid, ref_date, tgt_date), group in tqdm(grouped, desc="Computing WIS"):
            gt = group["GT"].iloc[0]
            if 0.5 not in group["Quantile"].values:
                continue
            median_pred = group[group["Quantile"] == 0.5]["Prediction"].iloc[0]
            ae = abs(median_pred - gt)
            wis = ae
            results.append({
                "Unique_id": uid,
                "Reference Date": ref_date,
                "Target End Date": tgt_date,
                "GT": gt,
                "WIS": wis
            })
        return pd.DataFrame(results)


# In[16]:


test_ids = [f"Y_{i}" for i in test_indices]


# In[17]:


processor = FixedModelTFTProcessor()


# In[18]:


processor.create_fixed_model(df_train, df_test_all, test_ids, h=4, freq="W-SAT", level = [10,20,30,40,50,60,70,80, 85,90,95])


# In[19]:


display_df = processor.create_display_df()


# In[21]:


processor.display_df.to_csv('../../output/forecasts_AutoTFT.csv',index=None)


# In[22]:


wis_table = processor.efficient_compute_wis()


# In[23]:


wis_table


# In[24]:


wis_table.to_csv('../../eval/WIS_AutoTST.csv',index=None)


# In[17]:


mean_wis = np.mean(wis_table['WIS'].values)


# In[18]:


mean_wis


# In[19]:


wis_dfs = [wis_table.iloc[i::4].reset_index(drop=True) for i in range(4)]


# In[23]:


np.mean(wis_dfs[3]['WIS'].values)


# In[ ]:




