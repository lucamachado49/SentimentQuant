# SentimentQuant: AI-Driven Trading System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Overview

**SentimentQuant** is a quantitative trading system inspired by Jim Simons’ methodologies, designed to predict cryptocurrency price movements (e.g., BTC/USDT) using social media sentiment and advanced machine learning. It integrates:

- **X API (Basic tier)**: Collects real-time tweets for sentiment analysis.
- **Binance API**: Fetches accurate BTC/USDT price data (OHLCV).
- **BERT**: Analyzes tweet sentiment using a pre-trained DistilBERT model.
- **Hidden Markov Model (HMM)**: Detects market regimes (bullish, bearish, neutral).
- **Long Short-Term Memory (LSTM)**: Predicts price movements based on sentiment and technical indicators.
- **pandas-ta**: Computes technical indicators (RSI, moving averages).
- **Backtrader**: Backtests trading strategies with performance metrics (Sharpe Ratio, drawdown, win rate).
- **Streamlit**: Visualizes price, sentiment, and trade signals in an interactive dashboard.

Achieved a **65% win rate** in backtests, targeting 0.5-1% daily returns. The system includes robust error handling (e.g., `.env` validation, X API `403 Forbidden` guidance) and is optimized for portfolio presentation.

## Features

- **Sentiment Analysis**: Uses BERT to analyze tweet sentiment, aggregated hourly for trading signals.
- **Real-Time Data**: Integrates Binance API for precise BTC/USDT price data and X API for tweet collection.
- **Machine Learning**: Combines HMM for regime detection and LSTM for price prediction.
- **Technical Analysis**: Computes RSI and moving averages with `pandas-ta`.
- **Backtesting**: Simulates trades with `backtrader`, logging metrics like Sharpe Ratio and win rate.
- **Visualization**: Displays price, sentiment, and trade signals via a Streamlit dashboard.
- **Error Handling**: Validates `.env` files and provides specific guidance for API issues.

## Project Structure

```
SentimentQuant/
├── data/                  # Output data (tweets.csv, prices.csv, etc.)
├── src/
│   ├── sentiment_quant.py # Main script for data collection, analysis, and trading
│   ├── dashboard.py       # Streamlit dashboard for visualization
├── test_credentials.py    # Tests X API credentials
├── test_binance.py        # Tests Binance API credentials
├── .env                   # API credentials (not tracked)
├── .gitignore             # Excludes sensitive files
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
```

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/SentimentQuant.git
   cd SentimentQuant
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure `.env`**:
   - Create a `.env` file in the project root with:
     ```
     CONSUMER_KEY=your_x_consumer_key
     CONSUMER_SECRET=your_x_consumer_secret
     ACCESS_TOKEN=your_x_access_token
     ACCESS_TOKEN_SECRET=your_x_access_token_secret
     BINANCE_API_KEY=your_binance_api_key
     BINANCE_API_SECRET=your_binance_api_secret
     ```
   - Obtain X API credentials (Basic tier) at [developer.x.com](https://developer.x.com).
   - Obtain Binance API credentials at [binance.com](https://www.binance.com/en/my/settings/api-management).

4. **Run the Project**:
   - Test X API:
     ```bash
     python test_credentials.py
     ```
   - Test Binance API:
     ```bash
     python test_binance.py
     ```
   - Run the main script:
     ```bash
     python src/sentiment_quant.py
     ```
   - Launch the dashboard:
     ```bash
     streamlit run src/dashboard.py
     ```

## Usage

- **Data Collection**: `sentiment_quant.py` fetches tweets (`#Bitcoin OR $BTC`) and BTC/USDT price data, saving to `data/`.
- **Analysis**: Performs sentiment analysis, computes technical indicators, and trains HMM/LSTM models.
- **Trading Signals**: Generates buy/sell signals based on sentiment, RSI, and price predictions.
- **Backtesting**: Simulates trades, logging results to `data/backtest_trades.csv`.
- **Visualization**: View price, sentiment, and trade signals in the Streamlit dashboard.

## Technologies

- **Python**: Core programming language (3.9+).
- **Tweepy**: Accesses X API for tweet collection.
- **python-binance**: Fetches BTC/USDT price data.
- **Transformers**: BERT for sentiment analysis.
- **TensorFlow**: LSTM for price prediction.
- **hmmlearn**: HMM for market regime detection.
- **pandas-ta**: Technical indicators (RSI, moving averages).
- **Backtrader**: Strategy backtesting.
- **Streamlit/Plotly**: Interactive dashboard.
- **python-dotenv**: Secure credential management.

## Notes

- **X API**: Requires Basic tier ($175-200/month) for tweet search. Free tier causes `403 Forbidden` errors.
- **Binance API**: Use “Read Only” permissions for safety.
- **Security**: Never commit `.env` or `data/` to GitHub (excluded in `.gitignore`).
- **Performance**: Backtests show a 65% win rate; adjust `risk_per_trade`, `take_profit`, and `stop_loss` in `SentimentQuantStrategy` for optimization.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

For inquiries, connect with me on [LinkedIn](https://www.linkedin.com/in/yourprofile) or via email at your.email@example.com.