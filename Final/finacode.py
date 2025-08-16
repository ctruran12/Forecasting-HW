import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

random.seed(946)
np.random.seed(1)

config = {
    "retailCsv": r"...\MonthlyRetail(Millions).csv",
    "cpiCsv":    r"...\CPIAUCSL.csv",
    "cutoff": "2025-01-31",
    "start":  "2022-01-31",
    "end":    "2025-01-31",
    "rf": {"n_estimators": 300, "max_depth": 12, "max_features": None,
           "random_state": 21, "n_jobs": -1}
}
def loadSeries(filePath, dateCol="DATE"):
    df = pd.read_csv(filePath)
    if dateCol not in df.columns:
        dateCol = df.columns[0]
    valueCol = [c for c in df.columns if c != dateCol][0]
    df[dateCol] = pd.to_datetime(df[dateCol])
    df = df[[dateCol, valueCol]].rename(columns={dateCol: "date", valueCol: "value"})
    df = df.sort_values("date").drop_duplicates("date")
    # https://stackoverflow.com/questions/18233107/pandas-convert-datetime-to-end-of-month
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").ffill()
    firstValid = df["value"].first_valid_index()
    if df["value"].isna().any() and firstValid is not None:
        df.loc[:firstValid, "value"] = df.loc[firstValid, "value"]
    return df.dropna()

def cutTo(df, cutoffDate):
    return df[df["date"] <= pd.to_datetime(cutoffDate)].copy()

def joinCpi(retailDf, cpiDf):
    cpiDf = cpiDf.rename(columns={"value": "CPI"})
    out = pd.merge(retailDf.rename(columns={"value": "y"}), cpiDf, on="date", how="inner")
    out["CPI_YoY"] = out["CPI"].pct_change(12)
    return out

def addFeatures(df):
    d = df.copy()
    for lag in [1, 3, 6, 12]:
        d[f"yLag{lag}"] = d["y"].shift(lag)
    for w in [3, 6, 12]:
        d[f"yRmean{w}"] = d["y"].shift(1).rolling(w).mean()
        d[f"yRstd{w}"]  = d["y"].shift(1).rolling(w).std()
    d["month"] = d["date"].dt.month
    d["holiday"] = d["month"].isin([11, 12]).astype(int)
    d["cpiLag1"] = d["CPI"].shift(1)
    d["cpiYoYLag1"] = d["CPI_YoY"].shift(1)
    return d

def metrics(yTrue, yPred):
    mae = mean_absolute_error(yTrue, yPred)
    rmse = math.sqrt(mean_squared_error(yTrue, yPred))
    denom = np.where(yTrue == 0.0, 1e-9, yTrue)
    mape = np.mean(np.abs((yTrue - yPred) / denom)) * 100.0
    return mae, rmse, mape

def forecastEts(ySeries, horizon):
    model = ExponentialSmoothing(
        ySeries, trend="mul", seasonal="mul", seasonal_periods=12,
        initialization_method="estimated"
    )
    fit = model.fit(optimized=True, use_brute=True)
    return np.asarray(fit.forecast(horizon)), np.asarray(fit.fittedvalues)

def forecastArima(ySeries, horizon): #https://stackoverflow.com/questions/68852064/determine-the-parameter-range-for-grid-search-in-arima-p-d-q
    bestFit = None
    bestAic = np.inf
    for p in [0, 1, 2]:
        for q in [0, 1, 2]:
            try:
                fit = ARIMA(ySeries, order=(p, 1, q), trend="t").fit()
                if fit.aic < bestAic:
                    bestAic = fit.aic
                    bestFit = fit
            except Exception:
                pass
    if bestFit is None:
        return np.repeat(ySeries.iloc[-1], horizon)
    return np.asarray(bestFit.forecast(horizon))

def main():
    retail = cutTo(loadSeries(config["retailCsv"]), config["cutoff"])
    cpi    = cutTo(loadSeries(config["cpiCsv"]), config["cutoff"])
    base   = joinCpi(retail, cpi).dropna().reset_index(drop=True)

    start = pd.to_datetime(config["start"]).to_period("M").to_timestamp("M")
    end   = pd.to_datetime(config["end"]).to_period("M").to_timestamp("M")
    months = pd.date_range(start, min(end, pd.to_datetime(config["cutoff"])), freq="M")
    horizon = len(months)

    train = base[base["date"] < start].copy()
    if len(train) < 36:
        raise RuntimeError("Need at least 36 training months.")

    #ets arima forecast
    etsForecast, etsFitted = forecastEts(train["y"], horizon)
    arimaForecast = forecastArima(train["y"], horizon)

    
    feats = addFeatures(train).dropna().reset_index(drop=True)
    featureCols = [c for c in feats.columns if c not in ["date", "y", "CPI", "CPI_YoY"]]
    yLog = np.log(np.clip(train["y"].values, 1e-9, None))
    etsFitLog = np.log(np.clip(etsFitted, 1e-9, None))
    resLog = yLog - etsFitLog
    resLog = resLog[-len(feats):]

    rf = RandomForestRegressor(**config["rf"])

    rf.fit(feats[featureCols].values, resLog)

    #forecast loop https://stackoverflow.com/questions/53064545/statsmodels-implementing-a-direct-and-recursive-multi-step-forecasting-strategy
    history = train.copy()
    rfForecast = []
    for i, dt in enumerate(months):
        futureRow = base[base["date"] == dt][["date", "y", "CPI", "CPI_YoY"]]
        tmp = pd.concat([history, futureRow], ignore_index=True)
        tmp.loc[tmp.index[-1], "y"] = np.nan
        tmpFeats = addFeatures(tmp)
        tmpFeats[featureCols] = tmpFeats[featureCols].ffill()
        rowF = tmpFeats.loc[tmpFeats["date"] == dt, featureCols]
        if rowF.empty:
            rowF = tmpFeats[featureCols].tail(1)
        rhatLog = float(rf.predict(rowF.values)[0])
        rfFc = float(np.exp(rhatLog) * etsForecast[i])
        rfForecast.append(max(1e-6, rfFc))
        history = pd.concat([
            history,
            pd.DataFrame({
                "date": [dt],
                "y": [etsForecast[i]],
                "CPI": [base.loc[base["date"] == dt, "CPI"].values[0]],
                "CPI_YoY": [base.loc[base["date"] == dt, "CPI_YoY"].values[0]]
            })
        ], ignore_index=True)

    rfForecast = np.array(rfForecast)
    ensemble = np.median(np.vstack([etsForecast, arimaForecast, rfForecast]), axis=0)

    out = pd.DataFrame({
        "date": months,
        "ETS": etsForecast,
        "ARIMA": arimaForecast,
        "RF": rfForecast,
        "Ensemble": ensemble
    })

    merged = pd.merge(out, base[["date", "y"]], on="date", how="left")
    have = merged.dropna(subset=["y"])

    if len(have) > 0:
        print("Final Metrics (Jan 2022 to Jan 2025)")

        for model in ["ETS", "ARIMA", "RF", "Ensemble"]:
            mae, rmse, mape = metrics(have["y"], have[model])
            print(f"{model:9s}  MAE {mae:.4f}   RMSE {rmse:.4f}   MAPE {mape:.2f}%")


    # STL decomp
    STL(base.set_index("date")["y"], period=12, robust=True).fit().plot()
    plt.suptitle("STL Retail and Food Services Sales")
    plt.show()
    #plots
    plt.plot(base["date"], base["y"], label="Actual", color="black", linewidth=1.3)
    for k in ["ETS", "ARIMA", "RF", "Ensemble"]:
        plt.plot(out["date"], out[k], label=k)
    plt.axvline(x=start, color="k", linestyle=":", alpha=0.7)
    plt.title("Monthly Retail and Food Services Sales (Jan 2022 - Jan 2025)")
    plt.xlabel("Date")
    plt.ylabel("Millions USD")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(base["date"], base["y"], linewidth=1.6)
    plt.title("U.S. Retail & Food Services Sales (Millions USD)")
    plt.xlabel("Date")
    plt.ylabel("Millions USD")
    plt.show()

    s_yoy = base["y"].pct_change(12) * 100
    cpi_yoy = base["CPI"].pct_change(12) * 100
    plt.figure()
    plt.plot(base["date"], s_yoy, label="Sales YoY (%)", linewidth=1.4)
    plt.plot(base["date"], cpi_yoy, label="CPI YoY (%)", alpha=0.85)
    plt.axhline(0, color="k", linestyle=":", alpha=0.6)
    plt.title("Sales vs CPI ")
    plt.xlabel("Date") 
    plt.ylabel("Percent")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
