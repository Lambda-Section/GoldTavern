# GoldTavern Project: Week 1 Update
Date: May 26, 2024

## DISCLAIMER
This update is part of an educational project and is NOT financial advice. All predictions are experimental and should not be used for investment decisions.

## This Week's Highlights

### Project Launch
I'm excited to share the first update on my GoldTavern project - an experimental gold price forecasting system built in R. This project combines multiple forecasting models with economic indicators to predict gold price movements.

### Current Implementation
The system currently includes:

- **Data Collection**: Historical gold prices from Yahoo Finance (2010-2023)
- **Multi-Model Approach**:
  - LSTM Neural Network with 10-hour lookback window
  - GARCH(1,1) for volatility forecasting
  - Prophet for trend and seasonality detection
- **Ensemble Method**: Weighted combination (40% LSTM, 30% Prophet, 30% GARCH)
- **Economic Framework**: Ray Dalio-inspired quadrant analysis
- **Visualization**: Interactive plots using Plotly

The LSTM component uses a sequential model with two LSTM layers (50 units each), while the GARCH model provides volatility forecasts that help with risk management.

### Initial Results
The ensemble model shows promising initial results when trained on historical data:

- Successfully captures both trend and volatility patterns
- Generates 24-hour forecasts with hourly granularity
- Incorporates sentiment data (currently simulated)
- Adjusts forecasts based on economic conditions

### Ray Dalio Economic Framework
One of the most interesting aspects is the implementation of Ray Dalio's economic principles:

- Analyzes debt cycles using Federal Debt-to-GDP ratio
- Tracks productivity growth via Output-per-Hour metrics
- Monitors inflation/deflation indicators from CPI data
- Classifies economic environments into four quadrants:
  - Q1: Rising Growth, Rising Inflation (Gold positive)
  - Q2: Rising Growth, Falling Inflation (Gold neutral)
  - Q3: Falling Growth, Falling Inflation (Gold negative)
  - Q4: Falling Growth, Rising Inflation (Gold very positive)

### Current Market Context
Gold has been showing significant strength in 2024, recently trading around $2,330 per ounce. The current economic environment appears to favor gold as a store of value.

### Next Steps
For the coming week, I'll be working on:

- Replacing simulated sentiment data with real X (Twitter) analysis
- Enhancing the backtesting framework using quantstrat
- Creating an interactive dashboard with flexdashboard
- Implementing proper risk management with position sizing

## Connect & Discuss
I'd love to hear your thoughts on this project! Comment below or connect with me to discuss gold price forecasting, machine learning in finance, or R programming.

#GoldPriceForecasting #MachineLearning #DataScience #RStats #FinancialModeling
