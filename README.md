# Investment Portfolio Tracker

## Overview

The **Investment Portfolio Tracker** is a comprehensive web application built using **Streamlit** that helps users manage, analyze, and visualize their investment portfolios. The application allows users to set up a portfolio by selecting stocks, assigning weights to each stock, and tracking the portfolio’s performance over time. Key features include:

- Portfolio setup with stock selection and weight assignments
- Portfolio overview and performance analysis
- Benchmark comparison with major indices like NIFTY 50 and SENSEX
- Calculation of various financial metrics such as CAGR, Sharpe Ratio, and Maximum Drawdown

The tool uses **Yahoo Finance** to fetch historical stock data and visualizes it using **Plotly** and **Matplotlib**.

## Features

### 1. **Portfolio Setup**
   - Select stocks from a predefined list (e.g., **RELIANCE.NS**, **TCS.NS**, etc.).
   - Assign portfolio weights, with the total summing up to 100%.
   - Input the investment amount and time period for analysis.
   - Set a risk-free rate to calculate the Sharpe Ratio.

### 2. **Portfolio Overview**
   - View a summary of portfolio performance, including:
     - Final portfolio value.
     - Percentage change (gain/loss).
   - Visualize portfolio performance with:
     - Normalized close prices over time for each stock.
     - A pie chart displaying portfolio allocation.

### 3. **Performance Analysis**
   - View key financial metrics, including:
     - Compound Annual Growth Rate (CAGR).
     - Volatility (Annual).
     - Sharpe Ratio.
     - Maximum Drawdown.
   - Visualize cumulative returns of the portfolio over time.

### 4. **Benchmark Comparison**
   - Compare portfolio performance with major market indices:
     - **NIFTY 50**
     - **SENSEX**
   - Visualize performance of the portfolio, NIFTY 50, and SENSEX in a single graph.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/investment-portfolio-tracker.git
cd investment-portfolio-tracker
```

### 2. Run the app

After installing the dependencies, run the Streamlit app:

```bash
streamlit run portfolio_tracker.py
```
