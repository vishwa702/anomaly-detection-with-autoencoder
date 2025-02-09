# NYC Taxi Demand Anomaly Detection using LSTM Autoencoder

## Overview
This project aims to detect anomalies in NYC taxi demand using an **LSTM Autoencoder**. The dataset consists of taxi demand time series data, and we apply **time series decomposition** to separate trend, seasonality, and residuals before detecting anomalies.

## Dataset
The dataset used in this project is from the **Numenta Anomaly Benchmark (NAB)** and can be accessed at:
[NYC Taxi Dataset](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv)

The dataset contains:
- `timestamp`: The date and time of the record.
- `value`: The number of taxi rides at that time.

## Project Workflow
1. **Load & Preprocess Data**
   - Read the dataset and convert timestamps to datetime format.
   - Resample the data to hourly frequency.
   - Fill missing values using forward fill.
   - Normalize the data using `MinMaxScaler`.

2. **Time Series Decomposition**
   - Decompose the time series into **Trend, Seasonality, and Residuals** using `statsmodels.tsa.seasonal_decompose()`.
   - Use the trend component instead of the raw data to reduce noise.

3. **Anomaly Detection using LSTM Autoencoder**
   - Train an **LSTM Autoencoder** on normal taxi demand patterns.
   - Calculate reconstruction errors and define a threshold for anomalies.
   - Flag anomalies where reconstruction error exceeds the threshold.

4. **Visualization**
   - Plot the **trend component** with detected anomalies.
   - Mark anomalies on **residuals** and **trend + residuals**.
   - Provide insights on detected anomalies (e.g., holiday spikes, unusual demand drops).

## Installation & Dependencies
Ensure you have the following dependencies installed:

```sh
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn statsmodels
```

## Usage
1. **Open the Jupyter Notebook**:

```sh
jupyter notebook "Anomaly Detection with Autoencoder.ipynb"
```

2. **Run the cells sequentially** to execute the anomaly detection pipeline.


## Results & Insights
- **Seasonal spikes and trends in taxi demand are well captured.**
- **The LSTM Autoencoder successfully detects outliers**, which often correspond to real-world events (e.g., holidays, weather changes).
- **Using decomposition improves anomaly detection accuracy** by removing seasonal noise.

## Future Improvements
- Fine-tune LSTM hyperparameters for better anomaly detection.
- Use a **bidirectional LSTM** for capturing both past and future dependencies.
- Apply **Bayesian Change Point Detection** to detect abrupt trend changes.



