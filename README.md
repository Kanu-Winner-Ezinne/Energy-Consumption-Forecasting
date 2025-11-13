# ⚡ Electricity Demand and Price Forecasting Using Machine Learning and Deep Learning

## 📘 Overview
This project focuses on analyzing and forecasting *electricity demand and price* using time-series data combined with weather information.  
The goal is to understand how environmental and energy generation factors influence electricity consumption and pricing  and to build predictive models capable of estimating future electricity prices accurately.

---

## 🎯 Objectives
- Clean, preprocess, and merge electricity generation and weather datasets.  
- Explore correlations between different energy sources, weather conditions, and power demand.  
- Perform statistical time series analysis (stationarity tests, decomposition, autocorrelation).  
- Apply *machine learning (XGBoost)* and *deep learning (LSTM)* models for electricity price forecasting.  
- Compare model performances using *Mean Absolute Error (MAE)* and visual analysis.  

---

## 🧩 Datasets

### 1. Energy Dataset (energy_dataset.csv)
Contains hourly data on:
- Electricity generation by type (biomass, fossil, hydro, wind, solar, etc.)
- Total load (forecast and actual)
- Electricity prices (day-ahead and actual)  
*Duration:* 2015–2018  

### 2. Weather Dataset (weather_features.csv)
Hourly weather data for five major Spanish cities:
- *Barcelona, **Bilbao, **Madrid, **Seville, and **Valencia*  
Features include:
- Temperature, Pressure, Humidity, Wind speed, Rainfall, and Cloud coverage.

---

## 🧹 Data Preprocessing
- Removed columns with all zero or NaN values.  
- Interpolated missing values using *linear interpolation*.  
- Merged energy and weather datasets based on timestamps.  
- Added *hour, **weekday, **month, and **year* as temporal features.  
- Normalized data for better model convergence.  
- Applied *PCA (Principal Component Analysis)* for dimensionality reduction, retaining *80% variance*.  

---

## 📊 Exploratory Data Analysis (EDA)
- Visualized electricity demand, price, and generation trends.  
- Observed high correlation between fossil fuel sources — merged into a single feature: generation fossil total.  
- Performed *seasonal decomposition* and *ADF test*:  
  - ADF Statistic = −9.15, p-value < 0.01 → *Stationary series confirmed.*  
- Identified strong *autocorrelation* up to 25 lags, guiding time window selection for models.  

---

## 🤖 Modeling and Forecasting

### 1. XGBoost Regressor
- Input: reshaped time-windowed data from PCA features.  
- Evaluated using *MAE (Mean Absolute Error)*.  
- *Performance:*  
  - *Test MAE = 0.017*  
  - Accurately captured short-term and seasonal variations.  

### 2. LSTM (Long Short-Term Memory) Neural Network
- Architecture:
  - LSTM layer (32 units) → Flatten → Dense(128, ReLU) → Dropout(0.1) → Output layer.  
- Optimizer: Adam (lr=1e-4)  
- Trained for 100 epochs with validation split.  
- *Validation MAE ≈ 0.03*  
- Effectively modeled long-term dependencies in electricity pricing trends.  

---

## 🧠 Evaluation

| Model | Mean Absolute Error (MAE) | Remarks |
|-------|----------------------------|----------|
| *XGBoost* | 0.017 | Best overall accuracy |
| *LSTM* | 0.03 | Captured temporal dependencies effectively |

Both models tracked general trends accurately.  
*XGBoost* showed faster convergence and lower test error, while *LSTM* performed well for long-term temporal patterns.

---

## 🔍 Key Insights
- *Electricity prices* and *total load* exhibit strong daily and seasonal patterns.  
- *Fossil fuel generation* strongly influences total demand.  
- *Weather features* (especially temperature and pressure) impact consumption behavior.  
- *XGBoost* slightly outperformed *LSTM* in terms of accuracy and computation time.

---

## 📈 Future Improvements
- Implement hybrid models (e.g., CNN-LSTM or Bi-LSTM).  
- Integrate macroeconomic indicators (fuel costs, import/export rates).  
- Develop a *real-time forecasting dashboard* using Flask or Streamlit.  
- Expand model explainability using *SHAP* or *LIME*.

---

## 🛠 Tech Stack
*Languages:* Python  
*Libraries & Tools:*  
- NumPy, Pandas, Matplotlib, Seaborn, Plotly  
- Scikit-learn, Statsmodels, XGBoost, TensorFlow, Keras  

---

## 📂 Project Structure

