# ========= Libraries =========
library(tidyverse)
library(lubridate)
library(fpp3)   # tsibble + fable + feasts (includes ETS/ARIMA/TSLM)

# ========= Load & shape =========
df <- read_csv("C:/Users/Colin/Desktop/Boston College/Summer/Forecasting/dataf.csv")

data_full <- df %>%
  pivot_longer(-Year, names_to = "Month", values_to = "Sales",
               values_ptypes = list(Sales = double())) %>%
  mutate(Month = str_to_title(Month),
         Date  = yearmonth(paste(Year, Month))) %>%
  arrange(Date) %>%
  select(Date, Sales) %>%
  as_tsibble(index = Date) %>%
  fill_gaps()

data  <- data_full %>% filter(!is.na(Sales))

# ========= Train/Test =========
train <- data %>% filter_index(~ "2023 Dec")
test  <- data %>% filter_index("2024 Jan" ~ "2024 Dec")
h <- nrow(test)


stl_cmp <- train %>%
  model(STL = STL(Sales ~ trend(window = 13) + season(window = "periodic"))) %>%
  components()

autoplot(stl_cmp) + labs(title = "STL decomposition (training)")

# ========= FORCE MODELS (not Naive) =========
# ETS: additive error + additive trend (DAMPED) + additive season
tfit_ets <- train %>%
  model(
    ETS = ETS(Sales ~ error("A") +
                trend("A", phi = 0.92) +   # damping keeps it from being flat
                season("A"))               # monthly seasonality from data index
  )

# ARIMA: include AR & MA + an intercept (constant) so it’s not a random walk
tfit_arima <- train %>%
  model(
    ARIMA = ARIMA(Sales ~ 1 + pdq(1,1,1))   # non-seasonal (1,1,1) with constant
    # If you want seasonal too, try: ARIMA(Sales ~ 1 + pdq(1,1,1) + PDQ(0,1,1))
  )

# DynReg stays as requested
tfit_dynreg <- train %>%
  model(DynReg = TSLM(Sales ~ trend() + season()))

# Naive only for comparison / sanity
tfit_naive <- train %>% model(Naive = NAIVE(Sales))

# ========= Forecast =========
fc_ets    <- forecast(tfit_ets,    h = h)
fc_arima  <- forecast(tfit_arima,  h = h)
fc_dynreg <- forecast(tfit_dynreg, h = h)
fc_naive  <- forecast(tfit_naive,  h = h)

# ========= Simple Average of ETS/ARIMA/DynReg =========
combined_fc <- bind_rows(
  fc_ets    %>% mutate(.model = "ETS"),
  fc_arima  %>% mutate(.model = "ARIMA"),
  fc_dynreg %>% mutate(.model = "DynReg")
) %>%
  as_tibble() %>%
  select(Date, .model, .mean) %>%
  pivot_wider(names_from = .model, values_from = .mean) %>%
  mutate(.mean = rowMeans(across(c(ETS, ARIMA, DynReg)), na.rm = TRUE),
         .model = "Average") %>%
  select(Date, .mean, .model) %>%
  as_tsibble(index = Date, key = .model)

# ========= Build one table for plotting =========
all_fc <- bind_rows(
  fc_ets     %>% mutate(.model = "ETS"),
  fc_arima   %>% mutate(.model = "ARIMA"),
  fc_dynreg  %>% mutate(.model = "DynReg"),
  fc_naive   %>% mutate(.model = "Naive"),
  combined_fc                          # .model = "Average"
) %>% as_tibble()

test_start <- as.Date(min(test$Date))

# ========= Plot: Actuals vs each model =========
ggplot() +
  geom_line(data = data_full %>% as_tibble(),
            aes(x = as.Date(Date), y = Sales), color = "black") +
  geom_vline(xintercept = test_start, linetype = "dashed") +
  geom_line(data = all_fc %>% filter(.model %in% c("ETS","ARIMA","DynReg","Naive","Average")),
            aes(x = as.Date(Date), y = .mean, color = .model), linewidth = 1) +
  # Draw ARIMA points so it can’t hide even if it overlaps another line anywhere
  geom_point(data = all_fc %>% filter(.model == "ARIMA"),
             aes(x = as.Date(Date), y = .mean), size = 1.6) +
  scale_color_manual(values = c(
    "ETS" = "#1f77b4",
    "ARIMA" = "#d62728",
    "DynReg" = "#2ca02c",
    "Naive" = "#7f7f7f",
    "Average" = "#9467bd"
  )) +
  labs(title = "Actual vs Forecasts (ETS damped seasonal, ARIMA(1,1,1)+const, DynReg, Naive, Avg)",
       x = NULL, y = "Sales", color = "Model") +
  theme_minimal()

# ========= Sanity: prove they’re not flat/identical =========
print(tfit_ets %>% report())
print(tfit_arima %>% report())
cat("ARIMA equals Naive? ", isTRUE(all.equal(fc_arima$.mean, fc_naive$.mean)), "\n")
cat("ETS   equals Naive? ", isTRUE(all.equal(fc_ets$.mean,   fc_naive$.mean)), "\n")



# =========================
# Accuracy (RMSE/MAE/MAPE) + Theil's U per model
# =========================

# Align forecasts to the test set
test_tbl <- test %>% as_tibble() %>% select(Date, Sales)

pred_tbls <- list(
  ETS      = fc_ets      %>% as_tibble() %>% select(Date, .mean),
  ARIMA    = fc_arima    %>% as_tibble() %>% select(Date, .mean),
  DynReg   = fc_dynreg   %>% as_tibble() %>% select(Date, .mean),
  Naive    = fc_naive    %>% as_tibble() %>% select(Date, .mean),
  Average  = combined_fc %>% as_tibble() %>% select(Date, .mean)
)

# Metric helper (joined on Date, test period only)
acc_manual <- function(pred, truth_tbl, model_name){
  j <- truth_tbl %>% inner_join(pred, by = "Date") %>% drop_na(Sales, .mean)
  d <- j$Sales - j$.mean
  tibble(
    .model = model_name,
    .type  = "Test",
    RMSE   = sqrt(mean(d^2)),
    MAE    = mean(abs(d)),
    MAPE   = mean(abs(d / j$Sales)) * 100
  )
}

# Build accuracy table for all models
accuracy_table <- bind_rows(
  acc_manual(pred_tbls$ETS,     test_tbl, "ETS"),
  acc_manual(pred_tbls$ARIMA,   test_tbl, "ARIMA"),
  acc_manual(pred_tbls$DynReg,  test_tbl, "DynReg"),
  acc_manual(pred_tbls$Naive,   test_tbl, "Naive"),
  acc_manual(pred_tbls$Average, test_tbl, "Average")
)

print(accuracy_table %>% select(.model, .type, RMSE, MAE, MAPE))

# Theil's U for each model vs Naive (computed on the same aligned dates)
j_naive <- test_tbl %>% inner_join(pred_tbls$Naive, by="Date") %>% rename(Naive = .mean)
rmse_naive <- with(j_naive, sqrt(mean((Sales - Naive)^2)))

theils_u <- map_dfr(names(pred_tbls), function(m){
  j <- test_tbl %>% inner_join(pred_tbls[[m]], by = "Date") %>% drop_na(Sales, .mean)
  tibble(
    .model = m,
    Theils_U = sqrt(mean((j$Sales - j$.mean)^2)) / rmse_naive
  )
})

print(theils_u)




