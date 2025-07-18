import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable Hugging Face symlink warning
import tweepy
import pandas as pd
import numpy as np
from binance.client import Client
import yfinance as yf
from transformers import pipeline
from hmmlearn.hmm import GaussianHMM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import backtrader as bt
import plotly.express as px
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define .env file path
ENV_PATH = './.env'

# Log current working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Check if .env file exists, check for .env.txt, fallback to current directory
if not os.path.isfile(ENV_PATH):
    if os.path.isfile(ENV_PATH + '.txt'):
        logger.error(
            f"Found {ENV_PATH}.txt instead of {ENV_PATH}. "
            f"Rename to {ENV_PATH} (remove .txt extension) using File Explorer. "
            "Ensure the file is in the project root with UTF-8 encoding."
        )
        exit(1)
    logger.error(
        f".env file not found at {ENV_PATH}. "
        "Create '.env' in the project root with:\n"
        "CONSUMER_KEY=your_x_consumer_key\n"
        "CONSUMER_SECRET=your_x_consumer_secret\n"
        "ACCESS_TOKEN=your_x_access_token\n"
        "ACCESS_TOKEN_SECRET=your_x_access_token_secret\n"
        "BINANCE_API_KEY=your_binance_api_key\n"
        "BINANCE_API_SECRET=your_binance_api_secret\n"
        "Obtain X API credentials (Basic tier) at https://developer.x.com and Binance credentials at https://www.binance.com/en/my/settings/api-management"
    )
    exit(1)

# Load environment variables
logger.info(f"Loading .env file from {ENV_PATH}")
load_dotenv(ENV_PATH)

# Validate X and Binance API credentials
credentials = {
    'CONSUMER_KEY': os.getenv('CONSUMER_KEY'),
    'CONSUMER_SECRET': os.getenv('CONSUMER_SECRET'),
    'ACCESS_TOKEN': os.getenv('ACCESS_TOKEN'),
    'ACCESS_TOKEN_SECRET': os.getenv('ACCESS_TOKEN_SECRET'),
    'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY'),
    'BINANCE_API_SECRET': os.getenv('BINANCE_API_SECRET')
}
missing_credentials = [key for key, value in credentials.items() if not value]
if missing_credentials:
    logger.error(
        f"Missing API credentials: {', '.join(missing_credentials)}. "
        f"Check {ENV_PATH} for:\n"
        "CONSUMER_KEY=your_x_consumer_key\n"
        "CONSUMER_SECRET=your_x_consumer_secret\n"
        "ACCESS_TOKEN=your_x_access_token\n"
        "ACCESS_TOKEN_SECRET=your_x_access_token_secret\n"
        "BINANCE_API_KEY=your_binance_api_key\n"
        "BINANCE_API_SECRET=your_binance_api_secret\n"
        "Ensure no extra spaces or quotes and UTF-8 encoding. "
        "Obtain X API credentials (Basic tier) at https://developer.x.com and Binance credentials at https://www.binance.com/en/my/settings/api-management"
    )
    exit(1)

# Function 1: Collect X posts
def collect_tweets(query, count=100, lang='en'):
    """Collect tweets from X API."""
    try:
        auth = tweepy.OAuthHandler(os.getenv('CONSUMER_KEY'), os.getenv('CONSUMER_SECRET'))
        auth.set_access_token(os.getenv('ACCESS_TOKEN'), os.getenv('ACCESS_TOKEN_SECRET'))
        api = tweepy.API(auth, wait_on_rate_limit=True)
        tweets = api.search_tweets(q=f"{query} -is:retweet -is:reply", lang=lang, count=count)
        data = [
            {'text': tweet.text, 'created_at': tweet.created_at, 'followers': tweet.user.followers_count}
            for tweet in tweets if tweet.user.followers_count > 1000
        ]
        df_tweets = pd.DataFrame(data)
        df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'])
        if df_tweets.empty:
            logger.warning("No tweets collected. Check query or API rate limits.")
            return pd.DataFrame()
        df_tweets.to_csv('data/tweets.csv', index=False)
        logger.info(f"Collected {len(df_tweets)} tweets for {query}")
        return df_tweets
    except tweepy.TweepError as e:
        if '403' in str(e):
            logger.error(
                "403 Forbidden: Your X API access level (likely Free tier) does not support the search_tweets endpoint. "
                "Upgrade to Basic tier ($175-200/month) at https://developer.x.com/en/portal/products. "
                "Ensure your app has 'Read and Write' permissions under 'User authentication settings'."
            )
        else:
            logger.error(f"Error collecting tweets: {e}")
        return pd.DataFrame()

# Function 2: Collect price data
def collect_price_data(asset='BTCUSDT', timeframe='1h', limit=720, is_crypto=True):
    """Collect OHLCV price data from Binance API (crypto) or yfinance (stocks/WDO)."""
    try:
        if is_crypto:
            client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
            klines = client.get_klines(symbol=asset, interval=Client.KLINE_INTERVAL_1HOUR, limit=limit)
            df_prices = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df_prices = df_prices[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
            df_prices[['open', 'high', 'low', 'close', 'volume']] = df_prices[['open', 'high', 'low', 'close', 'volume']].astype(float)
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            df_prices = yf.download(asset, start=start_date, end=end_date, interval=timeframe)
            df_prices = df_prices.reset_index().rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        if df_prices.empty:
            logger.warning("No price data collected")
            return pd.DataFrame()
        df_prices.to_csv('data/prices.csv', index=False)
        logger.info(f"Collected {len(df_prices)} price points for {asset}")
        return df_prices
    except Exception as e:
        logger.error(f"Error collecting price data: {e}")
        return pd.DataFrame()

# Function 3: Sentiment analysis
def analyze_sentiment(df_tweets):
    """Perform sentiment analysis on tweets using BERT."""
    try:
        if df_tweets.empty:
            logger.warning("Empty tweet DataFrame; skipping sentiment analysis")
            return pd.DataFrame()
        sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f')
        df_tweets['sentiment'] = df_tweets['text'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
        df_tweets['score'] = df_tweets['text'].apply(lambda x: sentiment_analyzer(x)[0]['score'])
        df_tweets['sentiment_score'] = df_tweets.apply(
            lambda x: x['score'] if x['sentiment'] == 'POSITIVE' else -x['score'], axis=1
        )
        sentiment_hourly = df_tweets.groupby(df_tweets['created_at'].dt.floor('H'))['sentiment_score'].mean().reset_index()
        sentiment_hourly.to_csv('data/sentiment.csv', index=False)
        logger.info(f"Computed sentiment scores for {len(sentiment_hourly)} hours")
        return sentiment_hourly
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return pd.DataFrame()

# Function 4: Merge datasets
def merge_data(df_prices, df_sentiment):
    """Merge price and sentiment data by timestamp."""
    try:
        if df_prices.empty or df_sentiment.empty:
            logger.error("Empty DataFrame detected; cannot merge data")
            return pd.DataFrame()
        df_merged = pd.merge_asof(
            df_prices.sort_values('timestamp'),
            df_sentiment.rename(columns={'created_at': 'timestamp'}),
            on='timestamp',
            direction='nearest'
        )
        df_merged['sentiment_score'] = df_merged['sentiment_score'].fillna(0)
        df_merged['RSI'] = ta.rsi(df_merged['close'], length=14)
        df_merged['MA10'] = df_merged['close'].rolling(window=10).mean()
        df_merged['MA30'] = df_merged['close'].rolling(window=30).mean()
        df_merged.to_csv('data/merged_data.csv', index=False)
        logger.info(f"Merged {len(df_merged)} data points")
        return df_merged
    except Exception as e:
        logger.error(f"Error merging data: {e}")
        return pd.DataFrame()

# Function 5: HMM for regime detection
def train_hmm(df_merged):
    """Train HMM to detect market regimes."""
    try:
        if df_merged.empty:
            logger.error("Empty merged DataFrame; cannot train HMM")
            return None, df_merged
        returns = np.diff(np.log(df_merged['close'])).reshape(-1, 1)
        sentiment = df_merged['sentiment_score'].iloc[1:].values.reshape(-1, 1)
        X_hmm = np.concatenate([returns, sentiment], axis=1)
        hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
        hmm.fit(X_hmm)
        states = hmm.predict(X_hmm)
        df_merged['Regime'] = np.pad(states, (1, 0), mode='edge')
        df_merged.to_csv('data/merged_data.csv', index=False)
        logger.info("Trained HMM model")
        return hmm, df_merged
    except Exception as e:
        logger.error(f"Error training HMM: {e}")
        return None, df_merged

# Function 6: LSTM for price prediction
def train_lstm(df_merged, timesteps=10):
    """Train LSTM to predict price movements."""
    try:
        if df_merged.empty:
            logger.error("Empty merged DataFrame; cannot train LSTM")
            return None, None, None
        features = ['close', 'sentiment_score', 'RSI', 'MA10', 'MA30']
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_merged[features].dropna())
        X_lstm = np.array([scaled_data[i:i+timesteps] for i in range(len(scaled_data)-timesteps)])
        y_lstm = scaled_data[timesteps:, 0]
        model = Sequential([
            LSTM(50, input_shape=(timesteps, len(features)), return_sequences=True),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
        predictions = model.predict(X_lstm[-10:])
        predicted_prices = scaler.inverse_transform(
            np.concatenate([predictions, np.zeros((predictions.shape[0], len(features)-1))], axis=1)
        )[:, 0]
        logger.info("Trained LSTM model")
        return model, scaler, predicted_prices
    except Exception as e:
        logger.error(f"Error training LSTM: {e}")
        return None, None, None

# Function 7: Generate trading signals
def generate_signals(df_merged, predicted_prices, timesteps=10):
    """Generate buy/sell signals and log simulated trades."""
    try:
        if df_merged.empty:
            logger.error("Empty merged DataFrame; cannot generate signals")
            return df_merged
        df_merged['Signal'] = 0
        bullish_regime = df_merged['Regime'].iloc[-10:].mode()[0]
        trades = []
        if bullish_regime == 0:
            for i in range(-10, 0):
                timestamp = df_merged['timestamp'].iloc[i]
                close_price = df_merged['close'].iloc[i]
                predicted_price = predicted_prices[i + 10]
                rsi = df_merged['RSI'].iloc[i]
                prev_close = df_merged['close'].iloc[i - 1] if i > -10 else close_price
                if predicted_price > close_price and rsi < 40:
                    df_merged['Signal'].iloc[i] = 1  # Buy
                    trades.append({'timestamp': timestamp, 'action': 'buy', 'price': close_price})
                elif predicted_price < close_price or close_price > prev_close * 1.02:
                    df_merged['Signal'].iloc[i] = -1  # Sell
                    trades.append({'timestamp': timestamp, 'action': 'sell', 'price': close_price})
        df_merged.to_csv('data/merged_data.csv', index=False)
        if trades:
            pd.DataFrame(trades).to_csv('data/trades.csv', index=False)
            logger.info(f"Logged {len(trades)} simulated trades to data/trades.csv")
        logger.info("Generated trading signals")
        return df_merged
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return df_merged

# Function 8: Backtesting
class SentimentQuantStrategy(bt.Strategy):
    params = (('risk_per_trade', 0.01), ('take_profit', 1.02), ('stop_loss', 0.99))
    
    def __init__(self):
        self.regime = self.datas[0].Regime
        self.rsi = self.datas[0].RSI
        self.close = self.datas[0].close
        self.order = None
        self.trades = []
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'timestamp': self.datas[0].datetime.datetime(),
                'action': 'buy' if trade.size > 0 else 'sell',
                'price': trade.price,
                'pnl': trade.pnl
            })
    
    def next(self):
        if self.order:
            return
        if self.regime[0] == 0 and self.rsi[0] < 40:
            size = self.broker.getcash() * self.params.risk_per_trade / (self.close[0] * (1 - self.params.stop_loss))
            self.order = self.buy(size=size)
            self.sell(exectype=bt.Order.Stop, price=self.close[0] * self.params.stop_loss, parent=self.order)
            self.sell(exectype=bt.Order.Limit, price=self.close[0] * self.params.take_profit, parent=self.order)
        elif self.regime[0] != 0 or self.close[0] > self.close[-1] * self.params.take_profit:
            self.order = self.sell(size=self.position.size)
    
    def stop(self):
        if self.trades:
            pd.DataFrame(self.trades).to_csv('data/backtest_trades.csv', index=False)
            logger.info(f"Logged {len(self.trades)} backtest trades to data/backtest_trades.csv")

def run_backtest(df_merged):
    """Backtest the trading strategy with enhanced metrics."""
    try:
        if df_merged.empty:
            logger.error("Empty merged DataFrame; cannot backtest")
            return None
        cerebro = bt.Cerebro()
        data_feed = bt.feeds.PandasData(
            dataname=df_merged,
            datetime='timestamp',
            close='close',
            open='open',
            high='high',
            low='low',
            volume='volume',
            RSI='RSI',
            Regime='Regime'
        )
        cerebro.adddata(data_feed)
        cerebro.addstrategy(SentimentQuantStrategy)
        cerebro.broker.setcash(1000)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        results = cerebro.run()
        sharpe = results[0].analyzers.sharpe.get_analysis()
        drawdown = results[0].analyzers.drawdown.get_analysis()
        trades = results[0].analyzers.trades.get_analysis()
        returns = results[0].analyzers.returns.get_analysis()
        logger.info(f"Sharpe Ratio: {sharpe.get('sharperatio', 0):.2f}")
        logger.info(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
        logger.info(f"Total Trades: {trades.get('total', {}).get('total', 0)}")
        logger.info(f"Win Rate: {trades.get('pnl', {}).get('gross', {}).get('won', 0) / max(1, trades.get('total', {}).get('total', 1)):.2%}")
        logger.info(f"Total Return: {returns.get('rtot', 0):.2%}")
        return results
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        return None

# Main execution
if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df_tweets = collect_tweets('#Bitcoin OR $BTC', count=100)  # Reduced count to avoid rate limits
    df_prices = collect_price_data('BTCUSDT', '1h', 720, is_crypto=True)
    df_sentiment = analyze_sentiment(df_tweets)
    df_merged = merge_data(df_prices, df_sentiment)
    hmm, df_merged = train_hmm(df_merged)
    lstm_model, scaler, predicted_prices = train_lstm(df_merged)
    df_merged = generate_signals(df_merged, predicted_prices if predicted_prices is not None else [0])
    results = run_backtest(df_merged)
    # Run dashboard separately: streamlit run dashboard.py