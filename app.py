import yfinance as yf
import pandas as pd
import time
import streamlit as st
import requests
from io import StringIO
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

ASX_CSV_URL = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"

@st.cache_data(show_spinner=False)
def get_asx200_tickers():
    url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
    try:
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "wikitable"})

        df_list = pd.read_html(str(table))
        if df_list:
            df_asx200 = df_list[0]
            possible_cols = ['ASX code', 'Ticker symbol', 'Symbol', 'Code', 'Company']

            for col in possible_cols:
                if col in df_asx200.columns:
                    df_asx200['Ticker'] = df_asx200[col]
                    break
            else:
                raise ValueError("ASX 200 table does not have a recognizable Ticker column.")

            df_asx200['Ticker'] = df_asx200['Ticker'].astype(str).str.upper().str.strip()
            df_asx200['Ticker'] = df_asx200['Ticker'].str.replace(".AX", "", regex=False)
            return df_asx200[['Ticker']]
    except Exception as e:
        st.warning(f"Failed to fetch ASX 200 list: {e}")
        return pd.DataFrame(columns=['Ticker'])


@st.cache_data(show_spinner=False)
def get_asx200_tickers():
    url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
    try:
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "wikitable"})

        df_list = pd.read_html(str(table))
        if df_list:
            df_asx200 = df_list[0]

            if 'ASX code' in df_asx200.columns:
                df_asx200['Ticker'] = df_asx200['ASX code']
            elif 'Ticker symbol' in df_asx200.columns:
                df_asx200['Ticker'] = df_asx200['Ticker symbol']
            elif 'Symbol' in df_asx200.columns:
                df_asx200['Ticker'] = df_asx200['Symbol']
            else:
                raise ValueError("ASX 200 table does not have a recognizable Ticker column.")

            df_asx200['Ticker'] = df_asx200['Ticker'].astype(str).str.upper().str.strip()
            df_asx200['Ticker'] = df_asx200['Ticker'].str.replace(".AX", "", regex=False)
            return df_asx200[['Ticker']]
    except Exception as e:
        st.warning(f"Failed to fetch ASX 200 list: {e}")
        return pd.DataFrame(columns=['Ticker'])

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
            try:
                last_price = float(stock_data['Last Price'])
                stock_data['Undervalued'] = last_price < avg_price
            except:
                stock_data['Undervalued'] = False
            stock_data['RSI'] = calculate_rsi(hist['Close'])
            stock_data['200MA'] = round(hist['Close'].rolling(window=200).mean().iloc[-1], 2)
            try:
                stock_data['Above 200MA'] = float(stock_data['Last Price']) > stock_data['200MA']
            except:
                stock_data['Above 200MA'] = 'N/A'
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

def fetch_all_data_concurrently(tickers, max_threads=10):
    results = []
    progress_bar = st.empty()
    progress = 0
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_ticker = {executor.submit(get_stock_details_yahoo, ticker): ticker for ticker in tickers}
        for i, future in enumerate(as_completed(future_to_ticker)):
            data = future.result()
            results.append(data)
            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
    return results

def main():
    st.title("ASX Share Screener with Filters, Charts, Value Picks & Sentiment")

    df_tickers = get_all_asx_tickers_with_industries()

    ticker_source = st.radio("Run analysis on:", ["ASX 200", "All ASX Shares"])
    if ticker_source == "ASX 200":
        asx200_df = get_asx200_tickers()
        df_tickers = df_tickers[df_tickers['Ticker'].isin(asx200_df['Ticker'])]

    industries = sorted(df_tickers['Industry'].unique())
    selected_industry = st.selectbox("Select Industry to Filter", ["All"] + industries)

    if st.button("Fetch and Analyze ASX Stocks"):
        if selected_industry != "All":
            df_tickers = df_tickers[df_tickers['Industry'] == selected_industry]

        tickers = df_tickers['Ticker'].tolist()
        all_data = fetch_all_data_concurrently(tickers)

        df = pd.DataFrame(all_data)

        st.subheader("All Data")
        if 'Ticker' in df.columns:
            st.dataframe(df)

            df['RSI_numeric'] = pd.to_numeric(df['RSI'], errors='coerce')
            undervalued_df = df[(df['Undervalued'] == True) & (df['RSI_numeric'] < 40)]

            st.subheader("Recommended Value Buys (Below 6mo Avg, RSI < 40)")
            st.dataframe(undervalued_df.drop(columns=['RSI_numeric']))

            st.download_button("Download All Data", df.to_csv(index=False), "asx_all_data.csv")

            st.subheader("Visualize Stock Prices vs 6mo Avg and 200MA")
            selected_chart_tickers = st.multiselect("Choose tickers to chart", df['Ticker'].dropna().unique().tolist())
            for ticker in selected_chart_tickers:
                try:
                    yf_ticker = f"{ticker}.AX"
                    hist = yf.Ticker(yf_ticker).history(period="6mo")
                    if not hist.empty:
                        hist = hist[['Close']].dropna()
                        if hist.empty:
                            st.warning(f"No valid price data for {ticker}")
                            continue
                        ma200 = hist['Close'].rolling(window=200).mean()
                        avg_price = hist['Close'].mean()
                        plt.figure(figsize=(10, 4))
                        plt.plot(hist.index, hist['Close'], label='Close Price')
                        plt.axhline(avg_price, color='red', linestyle='--', label='6mo Avg')
                        if not ma200.isna().all():
                            plt.plot(hist.index, ma200, color='orange', linestyle='--', label='200MA')
                        plt.title(f"{ticker} Price Trend")
                        plt.legend()
                        st.pyplot(plt)
                except Exception as e:
                    st.warning(f"Error plotting {ticker}: {e}")
        else:
            st.error("No Ticker column found in data.")

if __name__ == "__main__":
    main()
