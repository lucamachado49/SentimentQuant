import streamlit as st
import pandas as pd
import plotly.express as px

def run_dashboard(df_merged, predicted_prices):
    """Display real-time dashboard with Streamlit."""
    if df_merged.empty:
        st.error("No data available to display. Check data collection and merging.")
        return
    st.title("SentimentQuant: AI-Driven Trading System")
    # Price and Sentiment Plot
    fig = px.line(df_merged, x='timestamp', y=['close', 'sentiment_score'], title='Price vs. Sentiment')
    fig.add_scatter(x=df_merged['timestamp'], y=df_merged['Regime'], mode='lines', name='Market Regime')
    st.plotly_chart(fig)
    # Trade Signals Table
    st.write("Recent Trade Signals:")
    st.dataframe(df_merged[['timestamp', 'close', 'sentiment_score', 'RSI', 'Regime', 'Signal']].tail(10))
    # Predicted Prices
    st.write(f"LSTM Predicted Prices: {predicted_prices}")
    # Backtest Trades
    try:
        trades_df = pd.read_csv('data/backtest_trades.csv')
        st.write("Backtest Trade History:")
        st.dataframe(trades_df)
    except FileNotFoundError:
        st.warning("No backtest trades available.")

if __name__ == "__main__":
    try:
        df_merged = pd.read_csv('data/merged_data.csv')
        predicted_prices = [0]  # Placeholder; load actual predictions if needed
        run_dashboard(df_merged, predicted_prices)
    except FileNotFoundError:
        st.error("Merged data not found. Run sentiment_quant.py first.")