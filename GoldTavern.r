# DISCLAIMER: This code is created by a hobbyist for educational and research purposes only.
# It is NOT intended to provide financial advice. The predictions and analyses generated
# by this system should not be used as the basis for any investment decisions.
# I am not a financial advisor, and this tool should be used at your own risk.
# Past performance is not indicative of future results.

# Install missing packages first
if (!require("fredr")) install.packages("fredr")
if (!require("prophet")) install.packages("prophet")
if (!require("rugarch")) install.packages("rugarch")
if (!require("keras")) install.packages("keras")
if (!require("reshape2")) install.packages("reshape2")
if (!require("plotly")) install.packages("plotly")
if (!require("flexdashboard")) install.packages("flexdashboard")

# Then load libraries
library(quantmod)    # Yahoo Finance data
library(fredr)       # FRED API
library(prophet)     # Prophet model
library(rugarch)     # GARCH
library(keras)       # LSTM
library(doParallel)  # Parallel processing
library(tidyverse)   # Data manipulation
library(plotly)      # Interactive plots
library(flexdashboard) # Dashboard
library(logger)      # Logging
library(sentimentr)  # Sentiment analysis
library(Metrics)     # Model evaluation metrics


# Logging setup
log_appender(appender_file("goldtavern.log"))
log_info("Starting GoldTavern execution on {Sys.Date()}")
# Setting up parralel processing
cl <- makeCluster(detectCores() - 1)
# It Reserve one core for system operations
registerDoParallel(cl)
log_info("Parallel processing initialized with {getDoParWorkers()} cores")


# --- Data Collection ---
# 1. Gold Prices (Hourly from Yahoo Finance)
tryCatch({
  getSymbols("GC=F", src = "yahoo", from = "2010-01-01", to = "2023-01-01")
# gold_prices is likely a time series object (xts/zoo class) from the quantmod package
# XTS/ZOO classes are specialized R data structures for time series
# eXtensible Time Series (Time-based Indexing) / Z's Ordered Observations (irregular time series)
  gold_prices <- Cl(GC=F)
  df <- data.frame(ds = index(gold_prices), y = as.numeric(gold_prices) %>% na.omit())
# Uses index(gold_prices) to extract the timestamps
# Timestamps are specific points in time
# Likely stored as POSIXct or Date Objects
# POSIXct A date-time class that stores time
# as seconds since January 1, 1970 (Unix epoch)
  log_info("Gold prices loaded successfully: {nrow(df)} observations")
  }, error = function(e) {
  log_error("Error loading gold prices: {conditionMessage(e)}")
  stop("Gold prices loading failed")
})

# Data Collection Visualization
ggplot(df, aes(x = ds, y = y)) +
  geom_line() +
  labs(title = "Gold Prices Over Time", x = "Date", y = "Price")


# 2. Simulated X sentiment (replace with real analysis if keywords provided)
set.seed(42)
sentiment <- rnorm(nrow(df), mean = 0, sd = 1)  # Simulated: -1 (bearish) to 1 (bullish)
df$sentiment <- sentiment# 2. Simulated X sentiment (replace with real analysis if keywords provided)
set.seed(42)
sentiment <- rnorm(nrow(df), mean = 0, sd = 1)  # Simulated: -1 (bearish) to 1 (bullish)
df$sentiment <- sentiment # Add sentiment column to the dataframe


# After Sentiment Visualization
ggplot(df, aes(x = ds, y = sentiment)) +
  geom_line(color = "blue") +
  labs(title = "X Sentiment Over Time", x = "Date", y = "Sentiment Score")


# 3. Fed rates from FRED API (replace "your_api_key" with actual key)
# The selected code retrieves and integrates Federal Reserve interest rate data
fredr_set_key("your_api_key")  # Get key from fred.stlouisfed.org
# Fetches the Daily Federal Funds Rate (DFF) from March 1 to April 7, 2025
fed_rates <- fredr(series_id = "DFF", observation_start = as.Date("2025-03-01"),
                   observation_end = as.Date("2025-04-07"))
# Processes the data by:
# Renaming columns (date → ds, value → fed_rates)
# Converting dates to POSIXct format to match the timestamp format in the main dataframe
fed_rates <- fed_rates %>%
  select(ds = date, fed_rates = value) %>%
  mutate(ds = as.POSIXct(ds))  # Match timestamp format
# Merges the Fed rates with the main dataframe:
df <- merge(df, fed_rates, by = "ds", all.x = TRUE) %>%
  fill(fed_rates, .direction = "down")  # Forward-fill missing Fed rates

# After Fed Rates Visualization
ggplot(df, aes(x = ds)) +
  geom_line(aes(y = y, color = "Gold Price")) +
  geom_line(aes(y = fed_rates * 100, color = "Fed Rate (scaled)")) +
  scale_color_manual(values = c("Gold Price" = "gold", "Fed Rate (scaled)" = "red")) +
  labs(title = "Gold Price vs Fed Rate", x = "Date", y = "Value")

# Correlation heatmap
library(reshape2)
correlation_matrix <- cor(df[, c("y", "sentiment", "fed_rates")], use = "complete.obs")
ggplot(melt(correlation_matrix), aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap")



# --- Prophet Model ---
# This creates a 24-hour gold price forecast
# that accounts for historical patterns, sentiment, and Fed rates.
prophet_data <- df %>% select(ds, y, sentiment, fed_rates) %>% na.omit()
# Creates a Prophet model that accounts for daily and weekly patterns in gold prices
prophet_model <- prophet(daily.seasonality = TRUE, weekly.seasonality = TRUE)
# Incorporates X sentiment and Federal Reserve rates as additional predictive factors
prophet_model <- add_regressor(prophet_model, "sentiment")
prophet_model <- add_regressor(prophet_model, "fed_rates")
# Trains the model on historical data
prophet_model <- fit.prophet(prophet_model, prophet_data)
# Generates a dataframe for the next 24 hours to make predictions
future <- make_future_dataframe(prophet_model, periods = 24, freq = "hour")
# Assumes neutral sentiment and unchanged Fed rates for the forecast period
# Set the "sentiment" column in the "future" dataframe to 0
future$sentiment <- 0  # Neutral future sentiment
future$fed_rates <- tail(prophet_data$fed_rates, 1)  # Last known rate
# Makes predictions and extracts just the forecasted values for the next 24 hours
prophet_forecast <- predict(prophet_model, future)$yhat
# yhat statistical notation for a predicted value, "y" value, "hat" estimate
# Extracts just the prediction
prophet_forecast <- tail(prophet_forecast, 24)  # Last 24 hours



# --- LSTM Model ---
# Prepare data (normalize and reshape)
# Creates a function that normalizes values to range [0,1]
# which helps neural networks train better
scale_data <- function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
scaled_data <- apply(df[, c("y", "sentiment", "fed_rates")], 2, scale_data)
# Defines how many previous time steps to use for prediction (10 hours of historical data)
lookback <- 10
X <- array(NA, dim = c(nrow(scaled_data) - lookback, lookback, 3))
y <- numeric(nrow(scaled_data) - lookback)
# For each time point, take 10 consecutive hours of data
# Store these sequences in X
# Store the gold price from the 11th hour in y (what we want to predict)
for (i in 1:(nrow(scaled_data) - lookback)) {
  X[i,,] <- scaled_data[i:(i + lookback - 1), ]
  y[i] <- scaled_data[i + lookback, 1]  # Predict gold price
}
# This creates a sliding window approach where each input is 10 hours of data,
# and each output is the gold price in the 11th hour.


# Define and train LSTM
# Initializes a sequential Keras model where layers are stacked linearly
lstm_model <- keras_model_sequential() %>%
  # LSTM layer with 50 units that returns sequences.
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(lookback, 3)) %>%
  # LSTM layer with 50 units that returns only the final output
  layer_lstm(units = 50) %>%
  # Dense output layer with 1 unit (single price prediction)
  layer_dense(units = 1)
lstm_model %>% compile(optimizer = "adam", loss = "mse")
# Update weights after seeing 32 samples
lstm_model %>% fit(X, y, epochs = 5, batch_size = 32, verbose = 1)


# Interactive time series plots
library(plotly)
p <- ggplot(df, aes(x = ds, y = y)) +
  geom_line() +
  labs(title = "Gold Prices Over Time", x = "Date", y = "Price")
ggplotly(p)

# Interactive dashboard
library(flexdashboard)
# Create an Rmd file with flexdashboard layout

# Forecast (1-step, repeated)
last_sequence <- scaled_data[(nrow(scaled_data) - lookback + 1):nrow(scaled_data), ]
last_sequence <- array(last_sequence, dim = c(1, lookback, 3))
lstm_forecast_raw <- predict(lstm_model, last_sequence)
lstm_forecast <- lstm_forecast_raw * (max(df$y, na.rm = TRUE) - min(df$y, na.rm = TRUE)) + min(df$y, na.rm = TRUE)
lstm_forecast_full <- rep(lstm_forecast, 24)  # Repeat for 24 hours

# --- GARCH Model ---
# Calculate log returns of gold price in percentages, a commmon transformation for financial time series.
returns <- diff(log(df$y)) * 100  # Log returns in percent
# Specifies a standard GARCH(1,1) model
garch_spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                         mean.model = list(armaOrder = c(0, 0)))
# Fit the GARCH model to the returns data, excluding NA values
garch_fit <- ugarchfit(spec = garch_spec, data = returns[!is.na(returns)])
garch_vol <- sigma(garch_fit)
# Gets the most recent gold price.
last_price <- tail(df$y, 1)
# Creates a one-step forecast by applying the predicted volatility to the last price.
garch_forecast <- last_price * (1 + tail(garch_vol, 1) / 100)  # 1-step
# Repeats this single forecast value 24 times to match the 24-hour forecast 
# horizon used by the other models.
garch_forecast_full <- rep(garch_forecast, 24)  # Repeat for 24 hours

# --- Ensemble ---
ensemble_forecast <- 0.4 * lstm_forecast_full + 0.3 * prophet_forecast + 0.3 * garch_forecast_full

# --- Output ---
cat("Gold Price Forecast for April 8, 2025 (GMT, first 5 hours):\n")
print(head(ensemble_forecast, 5))

# Plot
future_dates <- seq(from = as.POSIXct("2025-04-08 00:00:00", tz = "GMT"),
                    by = "hour", length.out = 24)
plot(future_dates, ensemble_forecast, type = "l", col = "blue",
     main = "Gold Price Forecast (April 8, 2025)", xlab = "Time (GMT)", ylab = "Price (USD)")
grid()

# Stop cluster
stopCluster(cl)