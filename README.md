# Financial Portfolio Optimization
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/FinancialCover.jpeg' />

## Table of Contents

- [Directory Structure](#directory-structure)
- [Project Overview](#project-overview)
- [Project Approach](#project-approach)
  - [Phase 1](#phase-1)
  - [Phase 2](#phase-2)
- [Visuals](#visuals)
- [Results](#results)
- [Conclusion](#Conclusion)
- [Contact Information](#contact-information)

## Directory Structure

- `notebook/`: Jupyter notebooks with detailed analysis.
- `reports/`: Business intelligence reports and dashboards.
- `visualizations/`: All generated visualizations.

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
   - Use Python and Tableau to create visualizations of portfolio performance and risk-return analysis.
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

### Phase 1: Financial Portfolio Optimization

1. **Portfolio Returns vs. Volatility Scatter Plot:**
   - Scatter plot showing the relationship between expected returns and volatility for 5000 randomly generated portfolios using Markowitz Mean-Variance Optimization.
   - **Purpose:** Visualizes the trade-off between risk (volatility) and reward (returns) when constructing an investment portfolio.
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/Return%20vs%20Volatility.png' align = 'center'/>

2. **Portfolio Optimization with the MPT:**
   - Scatter plot of 5000 random portfolios showing the relationship between expected returns and volatility, highlighting portfolios with maximum return and minimum volatility.
   - **Purpose:** Demonstrates the application of Modern Portfolio Theory (MPT) to optimize portfolios based on historical data analysis.
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/Portfolio%20Optimisation%20with%20the%20MPT.png' align = 'center'/>

3. **Correlation Matrix Heatmap:**
   - Heatmap displaying the correlations between different assets in the portfolio.
   - **Purpose:** Provides insights into how different assets move in relation to each other, aiding in diversification strategies and risk management.
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/heatmap.png' align = 'center'/>

### Phase 2: Stock Price Prediction using LSTM

1. **Actual vs. Predicted Stock Prices Plot:**
   - Line plot comparing the actual closing prices of stocks with their predicted prices generated using the LSTM model.
   - **Purpose:** Evaluates the performance of the LSTM model in predicting stock prices and visually illustrates the accuracy of the predictions over time.
   
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot.png' />
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot%20(1).png' />
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot%20(2).png' />
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot%20(3).png' />
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot%20(4).png' />
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot%20(5).png' />
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot%20(6).png' />
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot%20(7).png' />
<img src='https://github.com/Rehaaaan/Financial-Portfolio-Optimization/blob/main/visualizations/newplot%20(8).png' />

## Results

- **Maximum Return Portfolio:**
  - Return: 34.803%
  - Volatility: 40.526%
  - Asset Weights:

|index|Returns|Volatility|COALINDIA\.NS|HEROMOTOCO\.NS|ITC\.NS|LT\.NS|NCC\.NS|
|---|---|---|---|---|---|---|---|
|maximum Return|34.803%|40.526%|21.586%|7.557%|2.741%|1.994%|66.122%|


- **Minimum Volatility Portfolio:**
  - Return: 20.574%
  - Volatility: 	20.714%
  - Asset Weights:

|index|Returns|Volatility|COALINDIA\.NS|HEROMOTOCO\.NS|ITC\.NS|LT\.NS|NCC\.NS|
|---|---|---|---|---|---|---|---|
|minimun risk|20.574%|20.714%|13.628%|24.792%|41.702%|19.833%|0.045%|

- **LSTM Prediction Performance:**

| Asset         | RMSE    |
|---------------|---------|
| COALINDIA.NS  | 15.899  |
| HEROMOTOCO.NS | 157.722 |
| ITC.NS        | 15.966  |
| LT.NS         | 119.960 |
| NCC.NS        | 19.992  |
| ONGC.NS       | 39.011  |
| RELIANCE.NS   | 51.337  |
| SBIN.NS       | 32.389  |
| WIPRO.NS      | 33.126  |

## Conclusion

Based on the analysis and results obtained from the financial portfolio optimization project, here are the key conclusions:

1. **Portfolio Optimization Insights:**
   - The project utilized Modern Portfolio Theory (MPT) to optimize portfolios based on historical data of selected assets.
   - Through Markowitz Mean-Variance Optimization, portfolios were constructed to balance risk (volatility) and reward (returns).

2. **Performance Metrics:**
   - **Maximum Return Portfolio:** Achieved a return of 34.803% with a corresponding volatility of 40.526%. The asset weights for this portfolio were predominantly allocated to \(COALINDIA.NS\) (21.586%), \(HEROMOTOCO.NS\) (7.557%), \(ITC.NS\) (2.741%), \(LT.NS\) (1.994%), and \(NCC.NS\) (66.122%).
   - **Minimum Risk Portfolio:** Minimized volatility to 20.714%, with assets primarily allocated to \(ITC.NS\) (13.628%), \(LT.NS\) (24.792%), and \(NCC.NS\) (41.702%), while \(HEROMOTOCO.NS\), \(COALINDIA.NS\), and \(SBIN.NS\) had smaller weights.

3. **Stock Price Prediction:**
   - Implemented LSTM models to predict stock prices for individual assets.
   - Evaluated model performance using Root Mean Squared Error (RMSE), with varying levels of accuracy observed across different assets (e.g., \(HEROMOTOCO.NS\) with RMSE of 157.722 and \(ITC.NS\) with RMSE of 15.966).

4. **Visual Analysis:**
   - Visualized the relationships between returns and volatility using scatter plots for portfolio optimization.
   - Utilized correlation matrices to understand asset interdependencies and diversification opportunities.

5. **Next Steps:**
   - **Refinement:** Further refine predictive models and optimization strategies to improve accuracy and robustness.
   - **Implementation:** Implement optimized portfolios in real-world investment scenarios, considering market conditions and investor preferences.
   - **Monitoring:** Continuously monitor and adjust portfolios based on updated data and market trends to maintain optimal performance.

Overall, the project demonstrated effective use of quantitative methods and machine learning techniques to optimize financial portfolios, offering actionable insights for portfolio managers and investors aiming to achieve desired financial objectives while managing risks effectively.

## Contact Information

For any questions or further information, please contact:

[![**Email:**](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mohammedrehan2342@gmail.com)
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohammed-rehan-483943231/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/rehah_ahan/)

---

Thank you for your interest in the Financial Portfolio Optimization project. We hope this project provides valuable insights and tools for effective portfolio management.


