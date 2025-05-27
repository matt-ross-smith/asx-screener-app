import yfinance as yf
import pandas as pd
import time
import streamlit as st
import requests
from io import StringIO
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re

ASX_CSV_URL = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"

# Get all ASX stock tickers and industry info from ASX's official CSV
def get_all_asx_tickers_with_industries():
    try:
        response = requests.get(ASX_CSV_URL)
        response.encoding = 'utf-8'
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, skiprows=1)
        df = df[['ASX code', 'Industry Group']].dropna()
        df.columns = ['Ticker', 'Industry']
        df['Ticker'] = df['Ticker'].astype(str).str.strip()
        df['Industry'] = df['Industry'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Failed to download ticker list: {e}")
        return pd.DataFrame(columns=['Ticker', 'Industry'])

# Get news sentiment (simple keyword-based approach from Yahoo News)
def get_news_sentiment(ticker):
    try:
        search_url = f"https://finance.yahoo.com/quote/{ticker}.AX/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('h3')

        news_titles = [h.get_text() for h in headlines]
        positive_words = ['gain', 'surge', 'profit', 'beat', 'strong', 'record']
        negative_words = ['fall', 'drop', 'loss', 'miss', 'weak', 'cut']

        sentiment_score = 0
        for title in news_titles:
            title_lower = title.lower()
            sentiment_score += sum(1 for w in positive_words if w in title_lower)
            sentiment_score -= sum(1 for w in negative_words if w in title_lower)

        if sentiment_score > 1:
            sentiment = 'Positive'
        elif sentiment_score < -1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment
    except:
        return 'Unavailable'

# Get stock data from Yahoo Finance
def get_stock_details_yahoo(ticker):
    stock_data = {'Ticker': ticker}
    try:
        yf_ticker = f"{ticker}.AX"
        stock = yf.Ticker(yf_ticker)
        info = stock.info

        stock_data['Last Price'] = info.get('regularMarketPrice', 'N/A')
        stock_data['Price Change'] = info.get('regularMarketChange', 'N/A')
        stock_data['Percent Change'] = info.get('regularMarketChangePercent', 'N/A')

        hist = stock.history(period="6mo")
        if not hist.empty:
            avg_price = hist['Close'].mean()
            stock_data['6mo Avg Price'] = round(avg_price, 2)
            stock_data['Undervalued'] = stock_data['Last Price'] < avg_price
            stock_data['RSI'] = calculate_rsi(hist['Close'])
            stock_data['200MA'] = round(hist['Close'].rolling(window=200).mean().iloc[-1], 2)
            stock_data['Above 200MA'] = stock_data['Last Price'] > stock_data['200MA']
        else:
            stock_data['6mo Avg Price'] = 'N/A'
            stock_data['Undervalued'] = False
            stock_data['RSI'] = 'N/A'
            stock_data['200MA'] = 'N/A'
            stock_data['Above 200MA'] = 'N/A'

        stock_data['Sentiment'] = get_news_sentiment(ticker)
    except Exception as e:
        stock_data['Error'] = str(e)

    return stock_data

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2) if not rsi.empty else 'N/A'

# Streamlit UI
def main():
    st.title("ASX Share Screener with Filters, Charts, Value Picks & Sentiment")

    df_tickers = get_all_asx_tickers_with_industries()
    industries = sorted(df_tickers['Industry'].unique())
    selected_industry = st.selectbox("Select Industry to Filter", ["All"] + industries)

    if st.button("Fetch and Analyze ASX Stocks"):
        if selected_industry != "All":
            df_tickers = df_tickers[df_tickers['Industry'] == selected_industry]

        tickers = df_tickers['Ticker'].tolist()
        all_data = []
        progress = st.progress(0)

        for i, ticker in enumerate(tickers):
            stock_data = get_stock_details_yahoo(ticker)
            stock_data['Industry'] = df_tickers[df_tickers['Ticker'] == ticker]['Industry'].values[0]
            all_data.append(stock_data)
            progress.progress((i + 1) / len(tickers))
            time.sleep(0.2)

        df = pd.DataFrame(all_data)

        st.subheader("All Data")
        st.dataframe(df)

required_cols = {'Undervalued', 'RSI'}
if required_cols.issubset(df.columns):
    filtered_df = df[df['Undervalued'] == True]
    filtered_df = filtered_df[filtered_df['RSI'] != 'N/A']
    filtered_df = filtered_df[filtered_df['RSI'] < 40]
    st.subheader("Recommended Value Buys (Below 6mo Avg, RSI < 40)")
    st.dataframe(filtered_df)
else:
    st.warning("Some key data columns are missing. Try refreshing or checking data sources.")


        st.download_button("Download All Data", df.to_csv(index=False), "asx_all_data.csv")

        st.subheader("Visualize Stock Prices vs 6mo Avg and 200MA")
        selected_chart_tickers = st.multiselect("Choose tickers to chart", df['Ticker'].tolist())
        for ticker in selected_chart_tickers:
            try:
                yf_ticker = f"{ticker}.AX"
                hist = yf.Ticker(yf_ticker).history(period="6mo")
                if not hist.empty:
                    ma200 = hist['Close'].rolling(window=200).mean()
                    plt.figure(figsize=(10, 4))
                    plt.plot(hist.index, hist['Close'], label='Close Price')
                    plt.axhline(hist['Close'].mean(), color='red', linestyle='--', label='6mo Avg')
                    plt.plot(hist.index, ma200, color='orange', linestyle='--', label='200MA')
                    plt.title(f"{ticker} Price Trend")
                    plt.legend()
                    st.pyplot(plt)
            except:
                continue

if __name__ == "__main__":
    main()
