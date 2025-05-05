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
# Data Fetching
library(quantmod)    # Yahoo Finance data, quantmod fetches financial data
library(fredr)       # FRED API, fredr accesses economic indicators
# Modeling
library(prophet)     # Prophet model
library(rugarch)     # GARCH
library(keras)       # LSTM, keras integrates TensorFlow for LSTM
library(doParallel)  # Parallel processing
library(tidyverse)   # Data manipulation
# Visualization
library(plotly)      # Interactive plots
library(flexdashboard) # Dashboard
# Utilities
library(logger)      # Logging
library(sentimentr)  # Sentiment analysis
library(Metrics)     # Model evaluation metrics


# Logging setup
log_appender(appender_file("goldtavern.log"))
log_info("Starting GoldTavern execution on {Sys.Date()}")
# Setting up parralel processing
cl <- makeCluster(detectCores() - 1)
# It Reserve one core for system operations
# Math Insight: Parallelism reduces runtime for independent tasks but adds overhead for small datasets.
registerDoParallel(cl)
log_info("Parallel processing initialized with {getDoParWorkers()} cores")


# --- Data Collection ---
# 1. Gold Prices (Hourly from Yahoo Finance)
# Quantmod fetches Yahoo Finance data as an xts (eXtensible Time Series) object, indexed by POSIXct timestamps (seconds since 1970). Cl() extracts closing prices.
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
# Combines predictions from LSTM, Prophet, and GARCH models with different weights
ensemble_forecast <- 0.4 * lstm_forecast_full + 0.3 * prophet_forecast + 0.3 * garch_forecast_full

# --- Ray Dalio Economic Principles Implementation ---
# Install required packages
if (!require("quantmod")) install.packages("quantmod")
if (!require("TTR")) install.packages("TTR")

# Load libraries
library(quantmod)
library(TTR)

# 1. Debt Cycle Analysis
# Dalio emphasizes long-term and short-term debt cycles
log_info("Adding Ray Dalio's debt cycle analysis")

# Get debt-to-GDP ratio from FRED
tryCatch({
  fredr_set_key("your_api_key")
  debt_to_gdp <- fredr(series_id = "GFDEGDQ188S", # Federal Debt to GDP
                      observation_start = as.Date("2010-01-01"))

  # Calculate rate of change in debt
  debt_to_gdp$roc <- ROC(debt_to_gdp$value, n = 4) # Quarterly change

  # Identify debt cycle phase
  debt_cycle_threshold <- 2.5 # Percentage point threshold
  debt_to_gdp$cycle_phase <- ifelse(debt_to_gdp$roc > debt_cycle_threshold,
                                   "Expansion", "Contraction")

  # Get latest debt cycle phase
  current_debt_phase <- tail(debt_to_gdp$cycle_phase, 1)
  log_info("Current debt cycle phase: {current_debt_phase}")

  # 2. Productivity Growth
  productivity <- fredr(series_id = "OPHNFB", # Output per hour
                       observation_start = as.Date("2010-01-01"))
  productivity$roc <- ROC(productivity$value, n = 4)
  current_productivity_growth <- tail(productivity$roc, 1)

  # 3. Create Dalio's "Reflation/Deflation" indicator
  inflation <- fredr(series_id = "CPIAUCSL",
                    observation_start = as.Date("2010-01-01"))
  inflation$roc <- ROC(inflation$value, n = 12) # Annual inflation
  current_inflation <- tail(inflation$roc, 1)

  # Combine into Dalio's quadrant framework
  # 1: Rising Growth, Rising Inflation (Gold positive)
  # 2: Rising Growth, Falling Inflation (Gold neutral)
  # 3: Falling Growth, Falling Inflation (Gold negative)
  # 4: Falling Growth, Rising Inflation (Gold very positive)

  growth_rising <- tail(productivity$roc, 1) > 0
  inflation_rising <- tail(inflation$roc, 1) > tail(inflation$roc, 2)[1]

  dalio_quadrant <- case_when(
    growth_rising & inflation_rising ~ 1,
    growth_rising & !inflation_rising ~ 2,
    !growth_rising & !inflation_rising ~ 3,
    !growth_rising & inflation_rising ~ 4
  )

  # Adjust ensemble weights based on Dalio's principles
  dalio_gold_bias <- case_when(
    dalio_quadrant == 1 ~ 0.2,  # Positive
    dalio_quadrant == 2 ~ 0,    # Neutral
    dalio_quadrant == 3 ~ -0.1, # Negative
    dalio_quadrant == 4 ~ 0.3   # Very positive
  )

  # Apply Dalio bias to forecast
  ensemble_forecast <- ensemble_forecast * (1 + dalio_gold_bias)

  log_info("Applied Ray Dalio economic principles: Quadrant {dalio_quadrant}")

}, error = function(e) {
  log_error("Error implementing Dalio principles: {conditionMessage(e)}")
})

# Add visualization of Dalio's framework
dalio_plot <- ggplot() +
  geom_point(aes(x = tail(productivity$roc, 20),
                y = tail(inflation$roc, 20),
                color = seq_along(tail(productivity$roc, 20)))) +
  geom_hline(yintercept = 0) +
  geom_vline(xintercept = 0) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Dalio's Economic Framework",
       x = "Productivity Growth",
       y = "Inflation",
       color = "Time") +
  annotate("text", x = 1, y = 1, label = "Q1: Gold +") +
  annotate("text", x = 1, y = -1, label = "Q2: Gold =") +
  annotate("text", x = -1, y = -1, label = "Q3: Gold -") +
  annotate("text", x = -1, y = 1, label = "Q4: Gold ++")

# --- Output ---
# Prints the first 5 hours of predictions for April 8, 2025
cat("Gold Price Forecast for April 8, 2025 (GMT, first 5 hours):\n")
# Shows the forecast values in the console
print(head(ensemble_forecast, 5))

# Plot
future_dates <- seq(from = as.POSIXct("2025-04-08 00:00:00", tz = "GMT"),
                    by = "hour", length.out = 24)
plot(future_dates, ensemble_forecast, type = "l", col = "blue",
     main = "Gold Price Forecast (April 8, 2025)", xlab = "Time (GMT)", ylab = "Price (USD)")
grid()

# Stop cluster
# Properly releases system resources when the script finishes
stopCluster(cl)

# --- Real-time Data Integration ---
# Install and load required packages
if (!require("IBrokers")) install.packages("IBrokers")
library(IBrokers)  # For Interactive Brokers API

# Connect to data provider (Interactive Brokers example)
tws <- twsConnect(clientId = 1)
log_info("Connected to Interactive Brokers TWS")

# Function to fetch real-time gold prices
get_realtime_gold <- function() {
  contract <- twsFuture("GC", exch = "NYMEX")
  realtime_data <- reqMktData(tws, contract)
  return(realtime_data$LAST)
}

# Schedule regular data updates
realtime_update <- function() {
  current_price <- get_realtime_gold()
  # Update your dataframe with new data
  df <- rbind(df, data.frame(ds = Sys.time(), y = current_price))
  # Re-run models with updated data
  # ...
}

# --- Backtesting Framework ---
if (!require("quantstrat")) install.packages("quantstrat")
library(quantstrat)

# Initialize backtesting environment
initDate <- "2010-01-01"
currency("USD")
stock("GOLD", currency = "USD")

# Create strategy
strategy <- "GoldTavernStrategy"
strategy.st <- portfolio.st <- account.st <- strategy
rm.strat(strategy.st)
initPortf(portfolio.st, symbols = "GOLD", initDate = initDate)
initAcct(account.st, portfolios = portfolio.st, initDate = initDate)
initOrders(portfolio.st, initDate = initDate)
strategy(strategy.st, store = TRUE)

# Add signals based on ensemble forecast
add.signal(strategy.st, name = "sigThreshold",
           arguments = list(column = "ensemble_forecast",
                           threshold = 0.5,
                           relationship = "gt",
                           cross = TRUE),
           label = "long")

# Add rules
add.rule(strategy.st, name = "ruleSignal",
         arguments = list(sigcol = "long",
                         sigval = TRUE,
                         orderqty = 100,
                         ordertype = "market",
                         orderside = "long"),
         type = "enter")

# Run backtest
applyStrategy(strategy.st, portfolios = portfolio.st)
updatePortf(portfolio.st)
chart.Posn(portfolio.st, "GOLD")

# --- Risk Management ---
# Position sizing based on volatility
calculate_position_size <- function(capital, risk_percent, current_volatility) {
  risk_amount <- capital * (risk_percent / 100)
  position_size <- risk_amount / current_volatility
  return(floor(position_size))
}

# Stop loss and take profit calculation
calculate_stops <- function(entry_price, volatility, risk_reward_ratio = 2) {
  atr_multiple <- 2  # Use 2x ATR for stop loss
  stop_loss <- entry_price - (atr_multiple * volatility)
  take_profit <- entry_price + (atr_multiple * volatility * risk_reward_ratio)
  return(list(stop_loss = stop_loss, take_profit = take_profit))
}

# Kelly criterion for optimal position sizing
kelly_criterion <- function(win_rate, win_loss_ratio) {
  kelly_percentage <- win_rate - ((1 - win_rate) / win_loss_ratio)
  return(max(0, kelly_percentage * 0.5))  # Half-Kelly for safety
}

# --- Trade Execution ---
# Function to place orders via broker API
place_order <- function(order_type, quantity, price = NULL) {
  contract <- twsFuture("GC", exch = "NYMEX")

  if (order_type == "MARKET") {
    order <- twsOrder(orderId = reqIds(tws),
                     action = "BUY",
                     totalQuantity = quantity,
                     orderType = "MKT")
  } else if (order_type == "LIMIT") {
    order <- twsOrder(orderId = reqIds(tws),
                     action = "BUY",
                     totalQuantity = quantity,
                     orderType = "LMT",
                     lmtPrice = price)
  }

  placeOrder(tws, contract, order)
  log_info("Order placed: {order_type}, Quantity: {quantity}")
}

# Trading logic
execute_trading_strategy <- function() {
  # Get current position
  current_position <- 0  # Replace with actual position query

  # Get forecast and current price
  current_price <- get_realtime_gold()

  # Decision logic based on ensemble forecast
  if (tail(ensemble_forecast, 1) > current_price * 1.01 && current_position == 0) {
    # Bullish signal with 1% expected gain
    position_size <- calculate_position_size(100000, 2, tail(garch_vol, 1))
    place_order("MARKET", position_size)

    # Set stop loss and take profit
    stops <- calculate_stops(current_price, tail(garch_vol, 1))
    # Place stop loss and take profit orders
  }
}

# --- Performance Monitoring ---
if (!require("PerformanceAnalytics")) install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)

# Track trading performance
track_performance <- function() {
  # Calculate returns
  returns <- ROC(portfolio_value)

  # Performance metrics
  sharpe <- SharpeRatio(returns, Rf = 0.02/252)
  drawdown <- maxDrawdown(returns)
  cagr <- Return.annualized(returns)

  # Create performance dashboard
  performance_data <- data.frame(
    Metric = c("Sharpe Ratio", "Max Drawdown", "CAGR", "Win Rate"),
    Value = c(sharpe, drawdown, cagr, win_count/total_trades)
  )

  # Log performance
  log_info("Performance update: Sharpe={sharpe}, Drawdown={drawdown}")

  # Create performance charts
  charts.PerformanceSummary(returns)
}

# Schedule regular performance updates
schedule_performance_update <- function() {
  track_performance()
  # Schedule next update
}


