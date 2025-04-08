# GoldTavern

## Overview
GoldTavern is a gold price forecasting system that combines multiple predictive models and data sources to provide high-quality price predictions.

## Data Sources
- **Gold Prices**: High-frequency data (1-second ticks via Alpaca API)
- **Market Sentiment**: Analysis of social media posts from X (Twitter)
- **Economic Indicators**: Federal Reserve rates and other metrics via FRED API
- **Historical Data**: Yahoo Finance historical gold prices

## Modeling Approach
Our ensemble approach combines multiple forecasting techniques:

1. **LSTM (Long Short-Term Memory)**: Deep learning model for capturing long-term patterns and dependencies in time series data
2. **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**: Statistical model for volatility forecasting
3. **Prophet**: Facebook's time series forecasting tool for identifying trends and seasonality
4. **Ensemble Method**: Weighted average of individual model predictions for improved accuracy

## Technical Implementation
- **Language**: R
- **Key Libraries**: quantmod, fredr, prophet, rugarch, keras, tidyverse
- **Data Structures**: XTS/Zoo time series objects for efficient time-based operations
- **Performance Optimization**: Parallel processing via doParallel, potential GPU acceleration via TensorFlow

## License
MIT License - See LICENSE file for details
