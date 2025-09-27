import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from statsmodels.tsa.arima.model import ARIMA # REMOVED
from statsmodels.tsa.statespace.sarimax import SARIMAX # SARIMAX remains
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from datetime import timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller # For ADF Test
import matplotlib.pyplot as plt

# --- Configuration ---
# Set the page configuration for better aesthetics
st.set_page_config(layout="wide", page_title="Bread Revenue Forecasting Dashboard")

# --- Global Parameters based on user's implementation ---
# All parameters below reflect the exact orders and split date provided in your code:
TRAIN_END_DATE = '2024-09-01'
SARIMA_ORDER = (0, 0, 0) # p, d, q (Non-seasonal part)
SARIMA_SEASONAL_ORDER = (1, 0, 1, 28) # P, D, Q, s (Seasonal part with period s=28)
# ARIMA_ORDER is no longer needed as we only use SARIMAX
FORECAST_HORIZON_DAYS = 101 # Forecasting the next quarter (101 days)

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_and_prepare_data():
    """Loads, preprocesses, filters for 'Bread', and creates the time series."""
    try:
        df = pd.read_csv('Urban_Grocers.csv')
        
        # 1. Convert Date using user's explicit format: df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        
        # 2. Filter for 'Bread' only
        bread_df = df[df['Food_Category'] == 'Bread'].copy()
        
        # 3. Calculate Revenue: bread_df['Revenue'] = bread_df['Units_Sold'] * bread_df['Price_per_Unit']
        bread_df['Revenue'] = bread_df['Units_Sold'] * bread_df['Price_per_Unit']
        
        # 4. Aggregate by Date: bread_revenue_df = bread_df.groupby('Date')['Revenue'].sum()
        bread_revenue_df = bread_df.groupby('Date')['Revenue'].sum()
        
        # Resample to ensure all dates are present (daily frequency) and fill missing with 0
        ts = bread_revenue_df.resample('D').sum().fillna(0)
        
        return ts, bread_df
    except FileNotFoundError:
        st.error("Error: 'Urban_Grocers.csv' not found. Please ensure the file is in the correct directory.")
        return pd.Series(), pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.Series(), pd.DataFrame()

# --- XGBoost Feature Engineering ---
def create_ts_features(df, target_col='Revenue'):
    """Creates time series features for XGBoost, mimicking user's feature set."""
    
    # Reset index for feature creation
    df_feat = df.reset_index()
    df_feat.columns = ['Date', target_col]

    # Time-based features
    df_feat['dayofweek'] = df_feat['Date'].dt.dayofweek
    df_feat['dayofmonth'] = df_feat['Date'].dt.day
    df_feat['weekofyear'] = df_feat['Date'].dt.isocalendar().week.astype(int)
    df_feat['month'] = df_feat['Date'].dt.month
    df_feat['year'] = df_feat['Date'].dt.year
    
    # Lag and Rolling Features (User's specifications: lag1, lag7, rolling_mean7)
    df_feat['lag1'] = df_feat[target_col].shift(1)
    df_feat['lag7'] = df_feat[target_col].shift(7)
    df_feat['rolling_mean7'] = df_feat[target_col].rolling(window=7).mean().shift(1)

    df_feat.dropna(inplace=True)
    df_feat.set_index('Date', inplace=True)
    
    feature_cols = ['dayofweek', 'dayofmonth', 'weekofyear', 'month', 'year', 'lag1', 'lag7', 'rolling_mean7']
    X = df_feat[feature_cols]
    y = df_feat[target_col]
    
    return X, y, feature_cols

# --- Model Training and Forecasting ---

def train_and_forecast(ts):
    """Trains SARIMAX and XGBoost using user-specified split and parameters."""
    
    if ts.empty:
        return {'SARIMA': {'MSE': np.inf, 'Error': 'Time series is empty'}, 
                'XGBoost': {'MSE': np.inf, 'Error': 'Time series is empty'}}, None, None

    # User's explicit split point: train = bread_revenue_df[bread_revenue_df.index < '2024-09-01']
    train_ts = ts[ts.index < TRAIN_END_DATE]
    test_ts = ts[ts.index >= TRAIN_END_DATE]
    
    if train_ts.empty or test_ts.empty:
        st.error(f"Training or Test set is empty. Check data range relative to split date {TRAIN_END_DATE}.")
        return {'SARIMA': {'MSE': np.inf, 'Error': 'Split failed/Insufficient data'}, 
                'XGBoost': {'MSE': np.inf, 'Error': 'Split failed/Insufficient data'}}, None, None
    
    # Define future index for forecasting
    future_index = pd.date_range(start=ts.index.max() + timedelta(days=1), 
                                 periods=FORECAST_HORIZON_DAYS, freq='D')
    
    results = {}
    test_size = len(test_ts)

    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return mse, rmse, mae

    # --- 1. SARIMA Model (User's order: (0,0,0)(1,0,1, 28)) ---
    try:
        sarima_model = SARIMAX(train_ts, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER).fit(disp=False)
        
        # Predict on test set: dynamic=True ensures a multi-step forecast.
        sarima_pred_test = sarima_model.predict(start=train_ts.index.max() + timedelta(days=1), 
                                               end=test_ts.index.max(), 
                                               dynamic=True) 
        sarima_pred_test.index = test_ts.index
        mse, rmse, mae = calculate_metrics(test_ts, sarima_pred_test)

        results['SARIMA'] = {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'Test_Predictions': sarima_pred_test
        }
        
        # Forecast future
        sarima_forecast = sarima_model.predict(start=future_index[0], end=future_index[-1])
        results['SARIMA']['Future_Forecast'] = pd.Series(sarima_forecast, index=future_index).clip(lower=0)
    except Exception as e:
        results['SARIMA'] = {'MSE': np.inf, 'Error': str(e), 'Test_Predictions': pd.Series(), 'Future_Forecast': pd.Series()}

    # --- 2. XGBoost Model (Feature-based Regression) ---
    try:
        ts_df = ts.to_frame(name='Revenue')
        # 1. Feature engineering on the whole series
        X_all, y_all, feature_cols = create_ts_features(ts_df)
        
        # 2. Split X/y using index based on user's split date
        X_train = X_all[X_all.index < TRAIN_END_DATE]
        X_test = X_all[X_all.index >= TRAIN_END_DATE]
        y_train = y_all[y_all.index < TRAIN_END_DATE]
        y_test = y_all[y_all.index >= TRAIN_END_DATE]

        # Handle case where X_test might be short due to feature lags
        if X_test.empty or len(X_test) < test_size:
            raise ValueError("XGBoost training data insufficient after feature engineering/split.")

        # 3. Train model (User's specific parameters used here)
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, 
                                 learning_rate=0.05, max_depth=5, 
                                 subsample=0.8, colsample_bytree=0.8, 
                                 random_state=42)
        xgb_reg.fit(X_train, y_train)
        
        # 4. Predict on test set
        xgb_pred_test = xgb_reg.predict(X_test)
        mse, rmse, mae = calculate_metrics(y_test, xgb_pred_test)
        
        results['XGBoost'] = {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae, 
            'Test_Predictions': pd.Series(xgb_pred_test, index=X_test.index).clip(lower=0)
        }
        
        # 5. Iterative Forecast Future (Mimicking user's iterative logic)
        xgb_forecast_series = ts.to_frame(name='Revenue')
        
        for future_date in future_index:
            
            # Create features for the next day based on the growing series
            features = {
                'dayofweek': [future_date.dayofweek],
                'dayofmonth': [future_date.day],
                'weekofyear': [future_date.isocalendar().week],
                'month': [future_date.month],
                'year': [future_date.year],
                # Lag features use the last available data from the series
                'lag1': [xgb_forecast_series['Revenue'].iloc[-1]],
                'lag7': [xgb_forecast_series['Revenue'].iloc[-7]],
                'rolling_mean7': [xgb_forecast_series['Revenue'].iloc[-7:].mean()]
            }
            
            # Create DataFrame and explicitly re-index to match training feature order (Crucial step)
            X_future = pd.DataFrame(features).reindex(columns=feature_cols)
            
            # Predict
            next_pred = xgb_reg.predict(X_future)[0]
            
            # Append prediction to the series for the next iteration's features
            xgb_forecast_series = pd.concat([
                xgb_forecast_series, 
                pd.Series([next_pred], index=[future_date], name='Revenue').to_frame()
            ])

        xgb_future_forecast = xgb_forecast_series['Revenue'].loc[future_index].clip(lower=0)
        results['XGBoost']['Future_Forecast'] = xgb_future_forecast

    except Exception as e:
        results['XGBoost'] = {'MSE': np.inf, 'Error': str(e), 'Test_Predictions': pd.Series(), 'Future_Forecast': pd.Series()}
        
    return results, train_ts, test_ts


# --- Function to Plot ACF and PACF ---
def plot_acf_pacf(ts):
    """Plots ACF and PACF for the time series."""
    st.markdown("##### Autocorrelation and Partial Autocorrelation Plots")
    
    if len(ts) < 50:
        st.warning("Time series is too short to generate meaningful ACF/PACF plots.")
        return

    # Plot ACF and PACF side by side
    col1, col2 = st.columns(2)

    with col1:
        lags_limit = min(40, len(ts) // 2 - 1) 
        fig_acf, ax_acf = plt.subplots(figsize=(6, 4))
        plot_acf(ts, ax=ax_acf, lags=lags_limit, title='ACF - Bread Revenue')
        st.pyplot(fig_acf)

    with col2:
        fig_pacf, ax_pacf = plt.subplots(figsize=(6, 4))
        # Use 'ywm' method as requested by the user's provided code
        plot_pacf(ts, ax=ax_pacf, lags=lags_limit, title='PACF - Bread Revenue', method='ywm')
        st.pyplot(fig_pacf)
    
    st.markdown("""
    * **ACF (q):** Use for selecting the Moving Average ($q$) order.
    * **PACF (p):** Use for selecting the AutoRegressive ($p$) order.
    """)

# --- Function to Run ADF Test ---
def run_adf_test(ts):
    """Runs and displays the Augmented Dickey-Fuller (ADF) Test."""
    st.markdown("##### Augmented Dickey-Fuller (ADF) Test Results")
    
    if len(ts) < 20:
        st.info("Insufficient data length to run the ADF test meaningfully.")
        return

    # Run the ADF test on the series
    adf_result = adfuller(ts)
    
    results_df = pd.DataFrame({
        'Metric': ['ADF Statistic', 'p-value', '1%', '5%', '10%'],
        'Value': [
            f"{adf_result[0]:.4f}", 
            f"{adf_result[1]:.4f}", 
            f"{adf_result[4]['1%']:.4f}", 
            f"{adf_result[4]['5%']:.4f}", 
            f"{adf_result[4]['10%']:.4f}"
        ]
    })
    
    st.dataframe(results_df, hide_index=True, use_container_width=True)
    
    if adf_result[1] <= 0.05:
        st.success("Conclusion: **Reject the null hypothesis**. The time series is likely stationary (d=0).")
    else:
        st.warning("Conclusion: **Fail to reject the null hypothesis**. The time series may be non-stationary (requires differencing, i.e., $d > 0$).")


# --- Streamlit App UI ---
def main():
    st.title("🍞 Bread Revenue Forecasting Dashboard")
    st.subheader("SARIMAX & XGBoost Model Comparison for Bread") # Updated title
    st.write("---")

    # Load and prepare data (Filtered for Bread)
    ts, bread_df = load_and_prepare_data()
    
    if ts.empty:
        st.stop()
    
    # 1. ACF/PACF and ADF Test Tooling (Sidebar)
    st.sidebar.header("Stationarity & Seasonality Tools")
    
    with st.sidebar.expander("Show Time Series Analysis (ACF/PACF & ADF)"):
        run_adf_test(ts)
        st.markdown("---")
        plot_acf_pacf(ts)
        
        # Show the currently used orders
        st.markdown("---")
        st.markdown("##### Currently Used SARIMAX Parameters") # Updated title
        st.write(f"Non-Seasonal Order: {SARIMA_ORDER} (p, d, q)")
        st.write(f"Seasonal Order: {SARIMA_SEASONAL_ORDER} (P, D, Q, s)")

    # Data Display
    st.markdown("### 1. Filtered and Revenue-Calculated Data (Bread Only)")
    st.dataframe(bread_df.head(), use_container_width=True)
    
    st.markdown("### 2. Time Series Data (Daily Revenue Aggregation)")
    st.write(f"Train/Test Split Date: **{TRAIN_END_DATE}**")
    st.write(f"Total time span: {ts.index.min().date()} to {ts.index.max().date()}")
    st.dataframe(ts.tail(), use_container_width=True)

    # 3. Forecasting
    st.markdown("### 3. Model Training & Forecasting")
    
    with st.spinner(f"Training SARIMAX and XGBoost models... Train data ends at {TRAIN_END_DATE}."):
        model_results, train_ts, test_ts = train_and_forecast(ts)

    if train_ts is None:
        return

    # 4. Model Evaluation
    st.markdown("#### Model Performance Metrics (Lower RMSE/MAE is Better)")
    
    metrics = {
        'Model': [], 'MSE': [], 'RMSE': [], 'MAE (Mean Absolute Error)': [], 'Status': []
    }
    
    min_mae = np.inf
    best_model_name = ""

    # Only iterate over SARIMA and XGBoost
    for name in ['SARIMA', 'XGBoost']:
        res = model_results.get(name)
        if res is None: continue # Skip if model failed during execution setup

        metrics['Model'].append(name)
        if 'Error' in res:
            metrics['MSE'].append("Error")
            metrics['RMSE'].append("Error")
            metrics['MAE (Mean Absolute Error)'].append("Error")
            metrics['Status'].append(f"Failed: {res['Error']}")
        else:
            mae_val = res['MAE']
            metrics['MSE'].append(f"{res['MSE']:.2f}")
            metrics['RMSE'].append(f"{res['RMSE']:.2f}")
            metrics['MAE (Mean Absolute Error)'].append(f"{mae_val:.2f}")
            metrics['Status'].append("Success")

            # Determine the best model based on MAE (Mean Absolute Error)
            if mae_val < min_mae:
                min_mae = mae_val
                best_model_name = name

    metrics_df = pd.DataFrame(metrics)
    
    # Sort the dataframe by MAE
    def sort_key(series):
        return series.apply(lambda x: float(x.replace('Error', str(np.inf))))

    metrics_df = metrics_df.sort_values(
        by='MAE (Mean Absolute Error)', 
        key=sort_key, 
        ascending=True, 
        ignore_index=True
    )
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    
    if best_model_name:
        st.success(f"🏆 The best-performing model based on MAE/RMSE is **{best_model_name}**.")
    else:
        st.warning("⚠️ Both models failed or encountered errors. Please check the Detailed Error report below.")
        return
    
    # --- Model Debugging: Detailed Errors ---
    st.markdown("#### Model Debugging: Detailed Errors")
    failed_models = False
    
    for name in ['SARIMA', 'XGBoost']:
        res = model_results.get(name)
        if res and 'Error' in res:
            st.error(f"**{name} Failure:**")
            st.code(res['Error'], language='text')
            failed_models = True

    if not failed_models:
        st.info("All selected models trained and ran successfully on the chosen parameters.")
    
    # --- End Model Debugging Section ---


    # 5. Visualization (Plotting all forecasts)
    st.markdown("### 5. Historical Data, Test Predictions, and 90-Day Forecasts")
    
    
    fig = go.Figure()
    
    # 5a. Historical Training Data
    fig.add_trace(go.Scatter(
        x=train_ts.index, 
        y=train_ts.values, 
        mode='lines', 
        name='Historical (Train)', 
        line=dict(color='gray', width=1)
    ))

    # 5b. Historical Test Data (Actuals)
    fig.add_trace(go.Scatter(
        x=test_ts.index, 
        y=test_ts.values, 
        mode='lines', 
        name='Actual (Test)', 
        line=dict(color='#007BFF', width=2)
    ))

    # Define forecast colors
    colors = {'SARIMA': '#00B050', 'XGBoost': '#9933FF'}

    for name in ['SARIMA', 'XGBoost']:
        res = model_results.get(name)
        
        if res is None or 'Error' in res: continue
            
        # Trace: Test Period Predictions
        test_pred = res.get('Test_Predictions')
        if not test_pred.empty:
            fig.add_trace(go.Scatter(
                x=test_pred.index, 
                y=test_pred.values, 
                mode='lines', 
                name=f'{name} Test Prediction', 
                line=dict(color=colors[name], dash='dot', width=1)
            ))

        # Trace: Future Forecast
        future_forecast = res.get('Future_Forecast')
        if not future_forecast.empty:
            fig.add_trace(go.Scatter(
                x=future_forecast.index, 
                y=future_forecast.values, 
                mode='lines', 
                name=f'{name} 90-Day Forecast', 
                line=dict(color=colors[name], width=3)
            ))

    # Add Train/Test Split Line
    train_end_date_str = train_ts.index.max().strftime('%Y-%m-%d')
    fig.add_vline(x=train_end_date_str, line_width=1, line_dash="dash", line_color="black")
    
    # Add the annotation separately
    fig.add_annotation(
        x=train_end_date_str,
        y=1, # Top of the chart (normalized coordinates)
        xref="x",
        yref="paper",
        text="Train End",
        showarrow=False,
        xshift=-5,
        yshift=10,
        font=dict(color="black", size=12),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
    )
    
    fig.update_layout(
        title=f'Bread Revenue Forecasting Comparison',
        xaxis_title="Date",
        yaxis_title="Daily Revenue ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


    # 6. Best Model Forecasted Values
    st.markdown(f"### 6. Forecasted Revenue Values ({best_model_name})")
    st.write(f"The following table shows the next {FORECAST_HORIZON_DAYS} days of revenue forecasted by the best model: **{best_model_name}**.")
    
    best_forecast = model_results.get(best_model_name, {}).get('Future_Forecast', pd.Series())
    
    if not best_forecast.empty:
        st.dataframe(
            best_forecast.rename('Forecasted Revenue')
                .reset_index()
                .rename(columns={'index': 'Date'})
                .style.format({'Forecasted Revenue': '${:,.2f}'}), 
            hide_index=True, 
            use_container_width=True
        )
    else:
        st.info("No future forecast could be generated by the best model.")

if __name__ == '__main__':
    main()
