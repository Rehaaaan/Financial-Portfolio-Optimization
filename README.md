# Financial Portfolio Optimization Project
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/FinancialCover.jpeg' />


# Financial Portfolio Optimization Project

## Table of Contents

- [Project Overview](#project-overview)
- [Project Approach](#project-approach)
  - [Phase 1](#phase-1)
  - [Phase 2](#phase-2)
- [Visuals](#visuals)
- [Results](#results)
- [Directory Structure](#directory-structure)
- [Contact Information](#contact-information)

## Project Overview

This project focuses on financial portfolio optimization using historical stock price data. By leveraging various data analysis and machine learning techniques, we aim to construct an optimal portfolio that maximizes returns while minimizing risk. The tools and libraries used in this project include Python (pandas, numpy, matplotlib, cvxopt, yfinance), Excel, and Tableau.

## Project Approach

### Phase 1

1. **Data Collection:**
   - Collect historical stock prices data from Yahoo Finance using Python's `yfinance` library.

2. **Data Cleaning:**
   - Handle missing values and ensure proper date formatting.

3. **Data Analysis:**
   - Calculate key financial metrics such as returns, volatility, and correlations using Python.
   - Example code:
     ```python
     import pandas as pd
     import numpy as np

     # Calculate daily returns
     returns = pf_data.pct_change().dropna()

     # Calculate average returns and volatility
     avg_returns = returns.mean() * 250
     volatility = returns.std() * np.sqrt(250)
     ```

4. **Optimization Modeling:**
   - Implement Markowitz Mean-Variance Optimization using Python libraries (`numpy`, `cvxopt`).
   - Example code:
     ```python
     from cvxopt import matrix, solvers

     def portfolio_optimization(returns, cov_matrix):
         n = len(returns)
         P = matrix(cov_matrix)
         q = matrix(np.zeros(n))
         G = matrix(np.diag(np.ones(n) * -1))
         h = matrix(np.zeros(n))
         A = matrix(np.ones((1, n)))
         b = matrix(1.0)

         sol = solvers.qp(P, q, G, h, A, b)
         weights = sol['x']
         return weights
     ```

5. **Data Visualization:**
   - Use Excel and Tableau to create visualizations of portfolio performance and risk-return analysis.
   - Develop an interactive dashboard in Tableau displaying the optimized portfolio and risk-return trade-offs.

### Phase 2

1. **Stock Price Prediction using LSTM:**
   - Use Long Short-Term Memory (LSTM) neural networks to predict future stock prices.
   - Example code:
     ```python
     from sklearn.preprocessing import MinMaxScaler
     from keras.models import Sequential
     from keras.layers import Dense, LSTM

     def create_lstm_model(x_train, y_train):
         model = Sequential()
         model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
         model.add(LSTM(64, return_sequences=False))
         model.add(Dense(25))
         model.add(Dense(1))
         model.compile(optimizer='adam', loss='mean_squared_error')
         model.fit(x_train, y_train, batch_size=1, epochs=1)
         return model
     ```

2. **Prediction Visualization:**
   - Visualize the predicted stock prices alongside actual prices using Matplotlib and Plotly.
   - Example code:
     ```python
     import plotly.graph_objects as go

     def plot_predictions(predictions, actual_prices, title):
         fig = go.Figure()
         fig.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices['Close'], name='Actual Prices'))
         fig.add_trace(go.Scatter(x=actual_prices.index, y=predictions, name='Predicted Prices'))
         fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
         fig.show()
     ```

3. **Portfolio Optimization with LSTM Predictions:**
   - Re-evaluate the optimized portfolio using predicted prices.
   - Generate new optimal portfolios and visualize the results.


## Visuals

1. **Portfolio Returns vs. Volatility:**
   - Scatter plot of 5000 random portfolios showing the relationship between expected returns and volatility.

2. **Optimal Portfolios:**
   - Highlighting portfolios with maximum return and minimum volatility.

3. **Correlation Matrix:**
   - Heatmap displaying the correlations between different assets in the portfolio.

4. **LSTM Stock Price Predictions:**
   - Plot showing actual and predicted stock prices using LSTM.

## Results

- **Maximum Return Portfolio:**
  - Return: xx%
  - Volatility: xx%
  - Asset Weights: [weights]

- **Minimum Volatility Portfolio:**
  - Return: xx%
  - Volatility: xx%
  - Asset Weights: [weights]

- **Correlation Matrix:**
  - Visual representation of asset correlations.

- **LSTM Prediction Performance:**
  - Root Mean Squared Error (RMSE): xx

## Directory Structure

- `data/`: Raw and cleaned datasets.
- `notebook/`: Jupyter notebooks with detailed analysis.
- `reports/`: Business intelligence reports and dashboards.
- `visualizations/`: All generated visualizations.

## Contact Information

For any questions or further information, please contact:

[![**Email:**](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mohammedrehan2342@gmail.com)
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohammed-rehan-483943231/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/rehah_ahan/)

---

Thank you for your interest in the Financial Portfolio Optimization project. We hope this project provides valuable insights and tools for effective portfolio management.


