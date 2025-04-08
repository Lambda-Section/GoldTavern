library(quantmod)    # Yahoo Finance data
library(fredr)      # FRED API
library(prophet)    # Prophet model
library(rugarch)    # GARCH
library(keras)      # LSTM
library(doParallel) # Parallel processing
library(tidyverse)  # Data manipulation

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