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
git clone https://github.com/reetphy/investment-portfolio-tracker.git
cd investment-portfolio-tracker
```

### 2. Install dependencies
You need to install the required libraries before running the app. You can do this using the requirements.txt file provided in the repository.

```bash
pip install -r requirements.txt
```

The requirements.txt file includes the following dependencies:

streamlit: Framework to create the interactive web application.

yfinance: Library to download stock data from Yahoo Finance.

pandas: Library for data manipulation and analysis.

numpy: Library for numerical computing.

matplotlib: Plotting library for creating static, animated, and interactive visualizations.

plotly: Interactive plotting library.

streamlit-option-menu: Package for sidebar navigation.

### 3. Run the app

After installing the dependencies, run the Streamlit app:

```bash
streamlit run portfolio_tracker.py
```

This will open a new tab in your web browser with the Investment Portfolio Tracker app.

## Usage
### 1. Portfolio Setup
On the "Portfolio Setup" page, choose the stocks from the available list and assign weights to each stock. Ensure that the total weight adds up to 100%.

Enter the amount you wish to invest in INR and select the start and end dates for analysis.

Set a risk-free rate (for example, a government bond yield) to calculate the Sharpe Ratio.

### 2. Portfolio Overview
After setting up your portfolio, navigate to the "Portfolio Overview" page.

View the initial investment amount, final portfolio value, and the percentage change (gain/loss) over the selected period.

Visualize the normalized close prices of the selected stocks and the overall portfolio.

### 3. Performance Analysis
On the "Performance Analysis" page, view key financial metrics such as CAGR, Volatility, Sharpe Ratio, and Maximum Drawdown.

Visualize the cumulative return of your portfolio over time.

### 4. Benchmark Comparison
On the "Benchmark Comparison" page, compare the performance of your portfolio with major indices such as NIFTY 50 and SENSEX.

See how your portfolio's performance stacks up against the market.

## Financial Metrics Explained
CAGR (Compound Annual Growth Rate): The average annual growth rate of an investment over a specified period of time, assuming the investment grows at a constant rate.

Volatility (Annual): A statistical measure of the dispersion of returns, indicating how much the returns on an investment can deviate from its average over time. A higher volatility implies a higher risk.

Sharpe Ratio: A measure of risk-adjusted return, which compares the return of an investment to its risk (volatility). A higher Sharpe ratio indicates better risk-adjusted performance.

Max Drawdown: The largest peak-to-trough decline in portfolio value, indicating the maximum loss during a specific period.

## Technologies Used
Streamlit: An open-source Python library used for building interactive web applications.

Yahoo Finance API: Used for downloading historical stock data.

Pandas: Used for data manipulation and analysis.

Plotly: A powerful library for creating interactive visualizations.

Matplotlib: Used for static visualizations (e.g., charts, graphs).

Numpy: Used for numerical operations and calculations.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
If you'd like to contribute to this project, feel free to fork the repository, create a pull request with your changes, and submit it for review. Contributions are welcome to improve features, fix bugs, or enhance the app's usability.

## Contact
If you have any questions or suggestions, feel free to contact me:

Email: reetphy@gmail.com

[LinkedIn](https://www.linkedin.com/in/reet-chandra/) 

[GitHub](https://github.com/reetphy)

**Thank you for using the Investment Portfolio Tracker!**





