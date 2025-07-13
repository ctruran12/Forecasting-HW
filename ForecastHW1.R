# Load libraries
library(forecast)
library(tidyverse)
library(lubridate)
library(tsibble)
library(Metrics)
library(scales)

# Load the data
df <- read_csv("C:/Users/Colin/Desktop/Boston College/Summer/Forecasting/airlineMiles.csv")

autoplot(df_ts) +
  ggtitle("Airline Passenger Miles") +
  xlab("Year") +
  ylab("Passenger Miles") +
  scale_y_continuous(labels = label_comma())


# Convert to time series object
df_ts <- ts(df$miles, start = c(2020, 1), frequency = 12)

# Train-test split
train_ts <- window(df_ts, end = c(2023, 12))
test_ts <- window(df_ts, start = c(2024, 1))

#Naive Forecast
naive_model <- naive(train_ts, h = 13)

# Drift 
drift_model <- rwf(train_ts, h = 13, drift = TRUE)

# Seasonal Naive 
snaive_model <- snaive(train_ts, h = 13)

# Ts
ts_lm_model <- tslm(train_ts ~ trend + season)
ts_lm_forecast <- forecast(ts_lm_model, h = 13)

#plot
autoplot(df_ts) +
  autolayer(naive_model$mean, series = "Naive") +
  autolayer(drift_model$mean, series = "Drift") +
  autolayer(snaive_model$mean, series = "Seasonal Naive") +
  autolayer(ts_lm_forecast$mean, series = "TS Regression") +
  ggtitle("Forecasts vs Actuals") +
  xlab("Year") +
  ylab("Passenger Miles") +
  guides(colour = guide_legend(title = "Model"))

#Accuracies 
acc_df <- data.frame(
  Model = c("Naive", "Drift", "SNaive", "TS Regression"),
  RMSE = c(rmse(test_ts, naive_model$mean),
           rmse(test_ts, drift_model$mean),
           rmse(test_ts, snaive_model$mean),
           rmse(test_ts, ts_lm_forecast$mean)),
  MAE = c(mae(test_ts, naive_model$mean),
          mae(test_ts, drift_model$mean),
          mae(test_ts, snaive_model$mean),
          mae(test_ts, ts_lm_forecast$mean)),
  MAPE = c(mape(test_ts, naive_model$mean),
           mape(test_ts, drift_model$mean),
           mape(test_ts, snaive_model$mean),
           mape(test_ts, ts_lm_forecast$mean))
)

#TheilsU
naive_rmse <- rmse(test_ts, naive_model$mean)
acc_df$Theils_U <- acc_df$RMSE / naive_rmse

print(acc_df)
