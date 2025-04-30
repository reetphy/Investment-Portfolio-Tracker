import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objs as go
from streamlit_option_menu import option_menu
import plotly.colors as pc


# ------------------------
# Set Streamlit Page Config
# ------------------------
st.set_page_config(page_title="Stock Portfolio Analyzer", layout="wide")

# ------------------------
# Sidebar Navigation
# ------------------------
with st.sidebar:
    st.title("Investment Portfolio Tracker")

    selected_page = option_menu(
        menu_title="Navigation",  
        options=[
            "Portfolio Setup",
            "Portfolio Overview",
            "Performance Analysis",
            "Benchmark Comparison"
        ],
        icons=["gear", "bar-chart", "graph-up", "activity"],
        menu_icon="layers",  
        default_index=0
    )

# ------------------------
# Initialize session state
# ------------------------
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'ticker_weights' not in st.session_state:
    st.session_state.ticker_weights = {}

# ------------------------
# Utility Functions
# ------------------------
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']
    return data

def calculate_metrics(prices, weights, risk_free_rate):
    normalized_prices = prices / prices.iloc[0]
    portfolio = (prices * weights).sum(axis=1)

    initial_value = portfolio.iloc[0]
    final_value = portfolio.iloc[-1]
    days = (portfolio.index[-1] - portfolio.index[0]).days
    years = days / 365.25
    cagr = ((final_value / initial_value) ** (1 / years)) - 1

    daily_returns = portfolio.pct_change().dropna()
    annual_risk = daily_returns.std() * np.sqrt(252)
    annual_return = daily_returns.mean() * 252
    sharpe = (annual_return - risk_free_rate / 100) / annual_risk if annual_risk != 0 else np.nan

    roll_max = portfolio.cummax()
    drawdown = (portfolio - roll_max) / roll_max
    max_drawdown = drawdown.min()

    return {
        'portfolio': portfolio,
        'returns': daily_returns,
        'CAGR': cagr,
        'Volatility': annual_risk,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown
    }

# ------------------------
# Page 1: Portfolio Setup
# ------------------------
if selected_page == "Portfolio Setup":
    st.title("Portfolio Setup")
    tickers = st.multiselect("Choose Stocks", options=[
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "LT.NS", "AXISBANK.NS", "ITC.NS", "BHARTIARTL.NS"
    ])

    if tickers:
        st.subheader("Assign Weights (Total should be 100%)")
        st.session_state.ticker_weights = {}
        for ticker in tickers:
            weight = st.number_input(f"Weight for {ticker}", 0.0, 100.0, 10.0, key=f"weight_{ticker}")
            st.session_state.ticker_weights[ticker] = weight

    capital = st.number_input("Investment Amount (₹)", 100.0, 1e7, 100000.0)
    start_date = st.date_input("Start Date", datetime(2019, 1, 1))
    end_date = st.date_input("End Date", datetime.today(), max_value=datetime.today())
    risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 7.0)

    if st.button("Run Analysis"):
        total_weight = sum(st.session_state.ticker_weights.values())
        if round(total_weight, 2) != 100.0:
            st.error("Total weights must sum up to 100")
        else:
            selected_tickers = list(st.session_state.ticker_weights.keys())
            weights_arr = np.array([st.session_state.ticker_weights[t] for t in selected_tickers]) / 100

            prices = download_data(selected_tickers, start_date, end_date)
            st.session_state.portfolio_data = {
                'prices': prices,
                'weights': weights_arr,
                'tickers': selected_tickers
            }
            st.session_state.metrics = calculate_metrics(prices, weights_arr, risk_free)
            st.session_state.capital = capital 
            st.success("Analysis complete. Check other pages.")

# ------------------------
# Page 2: Portfolio Overview
# ------------------------
elif selected_page == "Portfolio Overview":
    st.title("Portfolio Overview")
    if not st.session_state.portfolio_data:
        st.warning("Please complete portfolio setup.")
    else:
        data = st.session_state.portfolio_data
        norm_prices = data['prices'] / data['prices'].iloc[0]
        weighted_prices = (data['prices'] * data['weights'])
        portfolio_value = weighted_prices.sum(axis=1)

        initial_investment = st.session_state.capital
        final_value = initial_investment * (portfolio_value.iloc[-1] / portfolio_value.iloc[0])
        percentage_change = ((final_value - initial_investment) / initial_investment) * 100

        st.metric(
            label=f"Final Value of ₹{initial_investment:,.0f} Investment", 
            value=f"₹{final_value:,.2f}",  
            delta=f"{percentage_change:,.2f}%" if percentage_change >= 0 else f"-{abs(percentage_change):,.2f}%"
        )

        st.subheader("Normalized Close Prices Over Time")
        tickers = norm_prices.columns.tolist()
        colors = pc.qualitative.Plotly 
        color_map = {ticker: colors[i % len(colors)] for i, ticker in enumerate(tickers)}

        fig = go.Figure()

        for ticker in tickers:
            fig.add_trace(go.Scatter(
                x=norm_prices.index,
                y=norm_prices[ticker],
                mode='lines',
                name=ticker,
                line=dict(color=color_map[ticker])
            ))

        norm_portfolio = (norm_prices * data['weights']).sum(axis=1)
        fig.add_trace(go.Scatter(
            x=norm_portfolio.index,
            y=norm_portfolio,
            mode='lines',
            name='Portfolio',
            line=dict(color='black', width=2)
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Portfolio Allocation")
        pie_data = pd.Series([st.session_state.ticker_weights[t] for t in data['tickers']], index=data['tickers'])
        pie_colors = [color_map[ticker] for ticker in pie_data.index]

        pie_fig = go.Figure(data=[go.Pie(
            labels=pie_data.index,
            values=pie_data,
            hole=0.3,
            marker=dict(colors=pie_colors)
        )])
        st.plotly_chart(pie_fig, use_container_width=True)

# ------------------------
# Page 3: Performance Analysis
# ------------------------
elif selected_page == "Performance Analysis":
    st.title("Performance Analysis")
    if not st.session_state.metrics:
        st.warning("Please complete portfolio setup.")
    else:
        metrics = st.session_state.metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
        col2.metric("Volatility (Annual)", f"{metrics['Volatility']*100:.2f}%")
        col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        col4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")

        cumulative_returns = (1 + metrics['returns']).cumprod() - 1

        st.subheader("Cumulative Return of Portfolio")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Return',
            line=dict(color='blue')
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Page 4: Benchmark Comparison
# ------------------------
elif selected_page == "Benchmark Comparison":
    st.title("Benchmark Comparison")

    st.markdown("""
        **NIFTY 50**: Represents the top 50 companies listed on NSE.  
        **SENSEX**: Represents 30 well-established companies on BSE.
    """)

    if not st.session_state.portfolio_data:
        st.warning("Please complete portfolio setup.")
    else:
        portfolio = st.session_state.metrics['portfolio']
        
        nifty_data = yf.download("^NSEI", start=portfolio.index[0], end=portfolio.index[-1])
        sensex_data = yf.download("^BSESN", start=portfolio.index[0], end=portfolio.index[-1])

        nifty_norm = nifty_data['Close'] / nifty_data['Close'].iloc[0]
        sensex_norm = sensex_data['Close'] / sensex_data['Close'].iloc[0]
        portfolio_norm = portfolio / portfolio.iloc[0]

        compare_df = pd.DataFrame({
            'Portfolio': portfolio_norm,
            'NIFTY 50': nifty_norm,
            'SENSEX': sensex_norm
        })

        fig = go.Figure()
        for column in compare_df.columns:
            fig.add_trace(go.Scatter(
                x=compare_df.index,
                y=compare_df[column],
                mode='lines',
                name=column
            ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Normalized Value",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
