import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stock-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📈 Financial Dashboard")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Market Overview", "Stock Analysis", "Crypto Tracker", "Portfolio"]
)

@st.cache_data(ttl=3600)
def load_market_data():
    """Load market data for major indices."""
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^FTSE": "FTSE 100"
    }
    
    data = {}
    for symbol, name in indices.items():
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            prev_close = hist['Open'].iloc[0]
            change = ((current - prev_close) / prev_close) * 100
            data[name] = {
                "price": current,
                "change": change
            }
    return data

def display_market_overview():
    """Display market overview page."""
    st.title("Market Overview")
    
    # Load market data
    with st.spinner("Loading market data..."):
        market_data = load_market_data()
    
    # Display market summary
    cols = st.columns(len(market_data))
    for idx, (index_name, data) in enumerate(market_data.items()):
        with cols[idx]:
            st.metric(
                index_name,
                f"${data['price']:,.2f}",
                f"{data['change']:+.2f}%"
            )
    
    # Add more sections here as we develop them
    st.subheader("Market Trends")
    st.info("Market trends visualization will be added here")
    
    st.subheader("Top Movers")
    st.info("Top gaining and losing stocks will be displayed here")
    
    st.subheader("Market News")
    st.info("Latest market news will be shown here")

def main():
    """Main application logic."""
    if analysis_type == "Market Overview":
        display_market_overview()
    elif analysis_type == "Stock Analysis":
        st.title("Stock Analysis")
        st.info("Stock analysis features will be added here")
    elif analysis_type == "Crypto Tracker":
        st.title("Cryptocurrency Tracker")
        st.info("Cryptocurrency tracking features will be added here")
    else:  # Portfolio
        st.title("Portfolio Management")
        st.info("Portfolio management features will be added here")

if __name__ == "__main__":
    main()
