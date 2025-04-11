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

# Setting up parralel processing
cl <- makeCluster(detectCores())
registerDoParallel(cl)  

# --- Data Collection ---
# 1. Gold Prices (Hourly from Yahoo Finance)
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



# Interactive time series plots
library(plotly)
p <- ggplot(df, aes(x = ds, y = y)) +
  geom_line() +
  labs(title = "Gold Prices Over Time", x = "Date", y = "Price")
ggplotly(p)

# Interactive dashboard
library(flexdashboard)
# Create an Rmd file with flexdashboard layout
