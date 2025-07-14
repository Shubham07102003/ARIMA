# ARIMA Model: A Comprehensive Overview

The **ARIMA (AutoRegressive Integrated Moving Average)** model is a statistical analysis technique commonly used for time series forecasting. It is effective for analyzing data that shows patterns over time, such as stock prices, sales, or economic indicators.

---

## Components of ARIMA

1. **AR (AutoRegressive):**
   - The current value of the series depends on its own past values.
   - Represented by parameter `p` (the number of lag terms).
   - Example:  
     ```
     X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t
     ```

2. **I (Integrated):**
   - Refers to differencing the time series to make it stationary.
   - Represented by parameter `d` (the number of differencing steps required to remove trends).
   - Example:  
     ```
     Y_t = X_t - X_{t-1}  (for d=1)
     ```

3. **MA (Moving Average):**
   - The current value of the series depends on past forecast errors.
   - Represented by parameter `q` (the number of lagged error terms).
   - Example:  
     ```
     X_t = c + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}
     ```

---

## The ARIMA Model

The ARIMA model combines these three components into a single framework:
```
X_t = c + φ₁X_{t-1} + ... + φₚX_{t-p} + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
```

It is defined as **ARIMA(p, d, q)** where:
- `p`: Number of autoregressive terms.
- `d`: Number of differencing steps.
- `q`: Number of moving average terms.

---

## Steps to Build an ARIMA Model

1. **Stationarity Check:**
   - Use tools like the **Augmented Dickey-Fuller (ADF) test** to check if the series is stationary.
   - If not stationary, apply differencing until stationarity is achieved.

2. **Identify Parameters:**
   - Use **ACF (Autocorrelation Function)** to determine `q`.
   - Use **PACF (Partial Autocorrelation Function)** to determine `p`.

3. **Model Fitting:**
   - Fit the ARIMA model using historical data.

4. **Diagnostics:**
   - Check the residuals for randomness (e.g., using ACF plots or statistical tests like the Ljung-Box test).

5. **Forecasting:**
   - Use the fitted model to forecast future values.

---

## Strengths of ARIMA

1. Handles both trend and seasonality when extended to **SARIMA (Seasonal ARIMA)**.
2. Flexible in modeling a wide range of time series data.
3. Suitable for short- to medium-term forecasting.

---

## Limitations of ARIMA

1. Assumes linear relationships; struggles with non-linear data.
2. Requires the time series to be stationary (or transformed into stationarity).
3. Performance decreases for very long-term forecasts.
4. Requires manual parameter tuning for `p`, `d`, `q` values.

---

## Extensions of ARIMA

1. **SARIMA (Seasonal ARIMA):**
   - Adds seasonal components `P, D, Q, s` to handle seasonality.
   - Example: **SARIMA(p, d, q)(P, D, Q, s)**.

2. **ARIMAX:**
   - Incorporates external variables (exogenous inputs) into the ARIMA framework.

3. **VARIMA (Vector ARIMA):**
   - Extends ARIMA to multivariate time series.

---

## Applications of ARIMA

- **Financial Forecasting:** Stock prices, returns, and volatility.
- **Sales Forecasting:** Monthly or seasonal sales predictions.
- **Economic Indicators:** GDP, inflation, or unemployment rates.
- **Energy Demand:** Forecasting electricity or gas consumption.

---

## ARIMA Model Implementation in Python

This section demonstrates how to implement the ARIMA model in Python step-by-step. Each step includes a brief description, followed by a code block where you can add the corresponding Python code.

---

### Step 1: Import Necessary Libraries
First, import all the required libraries for working with time series data and building an ARIMA model.

```python
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf_yw
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for better readability
```
---
### Step 2: Load and Visualize the Dataset
Load your time series dataset and visualize it to understand its structure and trends.
```python
dfs = pd.read_excel("/content/financialdata.xlsx")
```
```python
for l in list_dfs:
  # for l in list_dfs:
  df = (dfs[l])
  print(df.tail(5))
  # Plotting Volatility, Returns and Log Returns
  plt.figure(figsize=(14, 7))

  plt.plot(df.index, df["returns"], '-r', label=" Returns ")

  plt.plot(df.index, df["volatility"], '-r', label="Volatility (Returns)")
  plt.plot(df.index, df["log_volatility"], '.g', label="Volatility (Log Returns)")
  plt.title(f"Volatility for {l}")
  plt.xlabel("Date")
  plt.ylabel("Value")
  plt.legend()
  plt.grid(True)
  plt.show()

  plt.figure(figsize=(14, 7))
  plt.plot(df.index, df["returns"], '-r', label=" Returns ")
  plt.plot(df.index, df["SMA_90"], '.k', label="SMA 90 Days (Returns)")
  plt.plot(df.index, df["SMA_2"], '-r', label="SMA 2 years (Returns)")
  plt.plot(df.index, df["log_SMA_90"], '--g', label="SMA 90 Days (Log Returns)")
  plt.plot(df.index, df["log_SMA_2"], '-.b', label="SMA 2 years (Log Returns)")
  plt.title(f"Simple Moving Average for {l}")
  plt.xlabel("Date")
  plt.ylabel("Value")
  plt.legend()
  plt.grid(True)
  plt.show()


  plt.figure(figsize=(14, 7))
  plt.plot(df.index, df["returns"], '-r', label=" Returns ")
  plt.plot(df.index, df["EWMA_90"], '-r', label="EWMA 0.90 (Returns)")
  plt.plot(df.index, df["EWMA_95"], '--g', label="EWMA 0.95 (Returns)")
  plt.plot(df.index, df["EWMA_99"], '.k',  label="EWMA 0.99 (Returns)")


  plt.plot(df.index, df["log_EWMA_99"], '--g', label="EWMA 0.99 (Log Returns)")
  plt.plot(df.index, df["log_EWMA_95"], '-.r', label="EWMA 0.95 (Log Returns)")
  plt.plot(df.index, df["log_EWMA_90"], '-.b', label="EWMA 0.90 (Log Returns)")
  plt.title(f"EWMA for {l}")
  plt.xlabel("Date")
  plt.ylabel("Value")
  plt.legend()
  plt.grid(True)
  plt.show()
```
![image](https://github.com/user-attachments/assets/096c02ab-b85e-41bf-8127-c6bd5e5f90af)
---
### Step 3: Check for Stationarity
- Use statistical tests like the Augmented Dickey-Fuller (ADF) test to check if the time series is stationary.
- If the series is not stationary, apply differencing until stationarity is achieved.
```python
def is_stationary(series, significance_level=0.05):
    adf_result = adfuller(series.dropna())
    p_value = adf_result[1]
    return p_value <= significance_level, p_value
```
---
### Step 4: Identify ARIMA Parameters (p, d, q)
Use Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to determine the values of p and q.
#### PACF:
```python
for name, df in dfs.items():
    target_column = df.columns[1]
    data = df[target_column].dropna()
    # Plot PACF and store results
    plt.figure(figsize=(12, 6))
    plot_pacf(smoothed_data.dropna(), lags=min(22, len(smoothed_data) - 1), method='ywm')
    plt.title(f"PACF Plot for {name}")
    plt.show()

    # Store PACF results
    pacf_values = pacf_yw(smoothed_data.dropna(), nlags=min(22, len(smoothed_data) - 1), method='adjusted')
    significant_lags = [(lag, value) for lag, value in enumerate(pacf_values) if abs(value) > 1.96 / (len(smoothed_data) ** 0.5)]

    for lag, value in significant_lags:
        new_row = pd.DataFrame({"DataFrame": [name], "Lag": [lag], "PACF_Value": [value]})
        pacf_results = pd.concat([pacf_results, new_row], ignore_index=True)
```
![image](https://github.com/user-attachments/assets/035c1744-1ec6-40f5-afc1-af9529d707f1)
#### ACF:
```python
for name, df in dfs.items():
    target_column = "log_returns"
    data = df[target_column].dropna()
    # Calculate ACF values
    acf_values = acf(data, nlags=40, fft=False)  # Limit to 40 lags

    # Store ACF values in DataFrame
    for lag, value in enumerate(acf_values):
        new_row = pd.DataFrame({"DataFrame": [name], "Lag": [lag], "ACF_Value": [value]})
        acf_values_df = pd.concat([acf_values_df, new_row], ignore_index=True, sort=False)

        # Identify significant lags
        if abs(value) > fixed_threshold:
            acf_significant_lags_df = pd.concat([acf_significant_lags_df, new_row], ignore_index=True, sort=False)

    # Plot ACF with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(acf_values)), acf_values, basefmt=" ", markerfmt="o", linefmt="b-")
    plt.axhline(y=fixed_threshold, color="r", linestyle="--", label="Upper Threshold")
    plt.axhline(y=-fixed_threshold, color="r", linestyle="--", label="Lower Threshold")
    plt.title(f"ACF Plot for {name}")
    plt.xlabel("Lags")
    plt.ylabel("ACF")
    plt.legend()
    plt.show()
```
![image](https://github.com/user-attachments/assets/2775d3de-79dd-4673-9b52-95b875f8e6bc)

---
### Step 5: Fit the ARIMA Model
Fit an ARIMA model using the identified parameters.
```python
for name, df in dfs.items():
    try:
        # Ensure datetime index is set and handle index-based DATE
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Skipping {name}: Index is not a datetime index.")
            continue

        # Filter last 2 years of data
        df = df.last("2YE")

        # Handle frequency and ensure no missing values in index
        df = df.asfreq('D')  # Adjust frequency as needed
        df = df.ffill()  # Fill missing values

        # Ensure the target column exists
        target_column = "returns"
        if target_column not in df.columns:
            print(f"Skipping {name}: Target column '{target_column}' not found.")
            continue

        # Prepare and smooth data for ARIMA
        data = df[target_column].dropna()

        # Apply smoothing using the provided function
        window_size = 5  # Adjust the window size as needed
        data = smooth_with_moving_average(data, window=window_size).dropna()

        # Determine d using ADF test
        adf_stationary, adf_p_value = is_stationary(data)
        d = 0
        if not adf_stationary:
            d = 1  # Assuming one differencing is sufficient; adjust if needed
            data = data.diff().dropna()

        # Determine p (PACF) and q (ACF) values
        pacf_significant_lags = pacf_results[pacf_results["DataFrame"] == name]["Lag"].tolist()
        acf_significant_lags = acf_significant_lags_df[acf_significant_lags_df["DataFrame"] == name]["Lag"].tolist()

        p = max(pacf_significant_lags) if pacf_significant_lags else 0
        q = max(acf_significant_lags) if acf_significant_lags else 0

        # Fit ARIMA model
        model = ARIMA(data, order=(p, d, q))
        model_fit = model.fit()

        # Compute residuals and RMSE
        residuals = model_fit.resid
        mse = mean_squared_error(data[d:], residuals[d:])  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error

        # Append results
        new_row = pd.DataFrame({
            "DataFrame": [name],
            "p": [p],
            "d": [d],
            "q": [q],
            "AIC": [model_fit.aic],
            "BIC": [model_fit.bic],
            "RMSE": [rmse]
        })
        arima_results = pd.concat([arima_results, new_row], ignore_index=True, sort=False)

        # Plot diagnostics
        model_fit.plot_diagnostics(figsize=(12, 8))
        plt.suptitle(f"Diagnostics for {name} (ARIMA({p}, {d}, {q}))")
        plt.show()

        # Forecast for the next 2 months (60 days)
        forecast_steps = 10
        forecast = model_fit.forecast(steps=forecast_steps)

        # Create a datetime index for the forecast
        forecast_index = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

        # Create a forecast DataFrame
        forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_index)

        # Plot historical data and forecast
        plt.figure(figsize=(10, 6))
        plt.plot(data, label="Original Data")
        plt.plot(forecast_df, label="Forecast", color="red")
        plt.title(f"Forecast for {name} (ARIMA({p}, {d}, {q}))")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"ARIMA model failed for {name} with error: {e}")
```
![image](https://github.com/user-attachments/assets/8544530b-cf09-427a-aa42-5bc97704d6d1)
![image](https://github.com/user-attachments/assets/d0df0499-060e-47de-a3b2-159ffcccdd31)

## Conclusion

- In this repository, we explored the ARIMA model for time series forecasting, covering its theoretical foundation, step-by-step implementation in Python, and practical applications. ARIMA is a powerful tool for short- to medium-term forecasting, but it requires careful parameter tuning and data preprocessing. This project serves as a foundation for anyone looking to get started with time series modeling using ARIMA.
- For advanced applications, consider exploring extensions like SARIMA or integrating ARIMA into machine learning pipelines.
## Future Work

Here are some potential extensions to this project:
- Implement **SARIMA (Seasonal ARIMA)** for datasets with strong seasonality.
- Compare ARIMA predictions with results from machine learning models like LSTMs or Facebook Prophet.
- Automate the process of parameter tuning (`p`, `d`, `q`) using grid search or other optimization techniques.
- Explore multivariate time series forecasting using ARIMAX or VARIMA.
- Create a web-based dashboard for interactive time series forecasting.


## References

Here are some resources for further learning:
- [ARIMA Documentation in Statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [ARIMA-Investopedia](https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp)
- [ARIMA for Time Series Forecasting: A Complete Guide](https://www.datacamp.com/tutorial/arima)
