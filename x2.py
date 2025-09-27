import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from datetime import timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Bread Demand Forecasting Dashboard")

TRAIN_END_DATE = '2024-09-01'
SARIMA_ORDER = (0, 0, 0)  
SARIMA_SEASONAL_ORDER = (2, 0, 2, 28)  
FORECAST_HORIZON_DAYS = 101  

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_prepare_data():
    """Loads, preprocesses, filters for 'Bread', and creates the demand time series."""
    try:
        df = pd.read_csv('Urban_Grocers.csv')

        # Convert Date
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        # Filter for 'Bread'
        bread_df = df[df['Food_Category'] == 'Bread'].copy()

        # ✅ Use Units_Sold instead of Revenue
        bread_demand_df = bread_df.groupby('Date')['Units_Sold'].sum()

        # Resample daily
        ts = bread_demand_df.resample('D').sum().fillna(0)

        return ts, bread_df
    except FileNotFoundError:
        st.error("Error: 'Urban_Grocers.csv' not found. Please ensure the file is in the correct directory.")
        return pd.Series(), pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.Series(), pd.DataFrame()

# --- XGBoost Feature Engineering ---
def create_ts_features(df, target_col='Units_Sold'):
    """Creates time series features for XGBoost."""
    df_feat = df.reset_index()
    df_feat.columns = ['Date', target_col]

    # Time-based features
    df_feat['dayofweek'] = df_feat['Date'].dt.dayofweek
    df_feat['dayofmonth'] = df_feat['Date'].dt.day
    df_feat['weekofyear'] = df_feat['Date'].dt.isocalendar().week.astype(int)
    df_feat['month'] = df_feat['Date'].dt.month
    df_feat['year'] = df_feat['Date'].dt.year

    # Lag and rolling features
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
    """Trains SARIMAX and XGBoost on demand (units sold)."""
    if ts.empty:
        return {'SARIMA': {'MSE': np.inf, 'Error': 'Time series is empty'},
                'XGBoost': {'MSE': np.inf, 'Error': 'Time series is empty'}}, None, None

    train_ts = ts[ts.index < TRAIN_END_DATE]
    test_ts = ts[ts.index >= TRAIN_END_DATE]

    if train_ts.empty or test_ts.empty:
        st.error(f"Training or Test set is empty. Check data range relative to split date {TRAIN_END_DATE}.")
        return {'SARIMA': {'MSE': np.inf, 'Error': 'Split failed/Insufficient data'},
                'XGBoost': {'MSE': np.inf, 'Error': 'Split failed/Insufficient data'}}, None, None

    future_index = pd.date_range(start=ts.index.max() + timedelta(days=1),
                                 periods=FORECAST_HORIZON_DAYS, freq='D')

    results = {}
    test_size = len(test_ts)

    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return mse, rmse, mae

    # --- SARIMA ---
    try:
        sarima_model = SARIMAX(train_ts, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER).fit(disp=False)
        sarima_pred_test = sarima_model.predict(start=train_ts.index.max() + timedelta(days=1),
                                                end=test_ts.index.max(), dynamic=True)
        sarima_pred_test.index = test_ts.index
        mse, rmse, mae = calculate_metrics(test_ts, sarima_pred_test)

        results['SARIMA'] = {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'Test_Predictions': sarima_pred_test
        }

        sarima_forecast = sarima_model.predict(start=future_index[0], end=future_index[-1])
        results['SARIMA']['Future_Forecast'] = pd.Series(sarima_forecast, index=future_index).clip(lower=0)
    except Exception as e:
        results['SARIMA'] = {'MSE': np.inf, 'Error': str(e), 'Test_Predictions': pd.Series(), 'Future_Forecast': pd.Series()}

    # --- XGBoost ---
    try:
        ts_df = ts.to_frame(name='Units_Sold')
        X_all, y_all, feature_cols = create_ts_features(ts_df)

        X_train = X_all[X_all.index < TRAIN_END_DATE]
        X_test = X_all[X_all.index >= TRAIN_END_DATE]
        y_train = y_all[y_all.index < TRAIN_END_DATE]
        y_test = y_all[y_all.index >= TRAIN_END_DATE]

        if X_test.empty or len(X_test) < test_size:
            raise ValueError("XGBoost training data insufficient after feature engineering/split.")

        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000,
                                   learning_rate=0.05, max_depth=5,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42)
        xgb_reg.fit(X_train, y_train)

        xgb_pred_test = xgb_reg.predict(X_test)
        mse, rmse, mae = calculate_metrics(y_test, xgb_pred_test)

        results['XGBoost'] = {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae,
            'Test_Predictions': pd.Series(xgb_pred_test, index=X_test.index).clip(lower=0)
        }

        xgb_forecast_series = ts.to_frame(name='Units_Sold')
        for future_date in future_index:
            features = {
                'dayofweek': [future_date.dayofweek],
                'dayofmonth': [future_date.day],
                'weekofyear': [future_date.isocalendar().week],
                'month': [future_date.month],
                'year': [future_date.year],
                'lag1': [xgb_forecast_series['Units_Sold'].iloc[-1]],
                'lag7': [xgb_forecast_series['Units_Sold'].iloc[-7]],
                'rolling_mean7': [xgb_forecast_series['Units_Sold'].iloc[-7:].mean()]
            }
            X_future = pd.DataFrame(features).reindex(columns=feature_cols)
            next_pred = xgb_reg.predict(X_future)[0]
            xgb_forecast_series = pd.concat([
                xgb_forecast_series,
                pd.Series([next_pred], index=[future_date], name='Units_Sold').to_frame()
            ])
        xgb_future_forecast = xgb_forecast_series['Units_Sold'].loc[future_index].clip(lower=0)
        results['XGBoost']['Future_Forecast'] = xgb_future_forecast
    except Exception as e:
        results['XGBoost'] = {'MSE': np.inf, 'Error': str(e), 'Test_Predictions': pd.Series(), 'Future_Forecast': pd.Series()}

    return results, train_ts, test_ts

# --- ACF/PACF Plots ---
def plot_acf_pacf(ts):
    st.markdown("##### Autocorrelation and Partial Autocorrelation Plots")
    if len(ts) < 50:
        st.warning("Time series is too short to generate meaningful ACF/PACF plots.")
        return

    col1, col2 = st.columns(2)
    with col1:
        lags_limit = min(40, len(ts) // 2 - 1)
        fig_acf, ax_acf = plt.subplots(figsize=(6, 4))
        plot_acf(ts, ax=ax_acf, lags=lags_limit, title='ACF - Bread Demand')
        st.pyplot(fig_acf)

    with col2:
        fig_pacf, ax_pacf = plt.subplots(figsize=(6, 4))
        plot_pacf(ts, ax=ax_pacf, lags=lags_limit, title='PACF - Bread Demand', method='ywm')
        st.pyplot(fig_pacf)

# --- ADF Test ---
def run_adf_test(ts):
    st.markdown("##### Augmented Dickey-Fuller (ADF) Test Results")
    if len(ts) < 20:
        st.info("Insufficient data length to run the ADF test meaningfully.")
        return

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
        st.success("Conclusion: **Reject the null hypothesis**. The series is likely stationary (d=0).")
    else:
        st.warning("Conclusion: **Fail to reject the null hypothesis**. The series may be non-stationary (requires differencing, i.e., d>0).")

# --- Streamlit App UI ---
def main():
    st.title("🍞 Bread Demand Forecasting Dashboard")
    st.subheader("SARIMAX & XGBoost Model Comparison for Demand (Units Sold)")
    st.write("---")

    ts, bread_df = load_and_prepare_data()
    if ts.empty:
        st.stop()

    # Sidebar tools
    st.sidebar.header("Stationarity & Seasonality Tools")
    with st.sidebar.expander("Show Time Series Analysis (ACF/PACF & ADF)"):
        run_adf_test(ts)
        st.markdown("---")
        plot_acf_pacf(ts)
        st.markdown("---")
        st.markdown("##### Currently Used SARIMAX Parameters")
        st.write(f"Non-Seasonal Order: {SARIMA_ORDER} (p, d, q)")
        st.write(f"Seasonal Order: {SARIMA_SEASONAL_ORDER} (P, D, Q, s)")

    # Data preview
    st.markdown("### 1. Filtered Data (Bread Only)")
    st.dataframe(bread_df.head(), use_container_width=True)

    st.markdown("### 2. Time Series Data (Daily Demand - Units Sold)")
    st.write(f"Train/Test Split Date: **{TRAIN_END_DATE}**")
    st.write(f"Total time span: {ts.index.min().date()} to {ts.index.max().date()}")
    st.dataframe(ts.tail(), use_container_width=True)

    # Training
    st.markdown("### 3. Model Training & Forecasting")
    with st.spinner(f"Training SARIMAX and XGBoost models... Train data ends at {TRAIN_END_DATE}."):
        model_results, train_ts, test_ts = train_and_forecast(ts)
    if train_ts is None:
        return

    # Performance metrics
    st.markdown("#### Model Performance Metrics (Lower RMSE/MAE is Better)")
    metrics = {'Model': [], 'MSE': [], 'RMSE': [], 'MAE': [], 'Status': []}
    min_mae = np.inf
    best_model_name = ""
    for name in ['SARIMA', 'XGBoost']:
        res = model_results.get(name)
        if res is None: continue
        metrics['Model'].append(name)
        if 'Error' in res:
            metrics['MSE'].append("Error")
            metrics['RMSE'].append("Error")
            metrics['MAE'].append("Error")
            metrics['Status'].append(f"Failed: {res['Error']}")
        else:
            mae_val = res['MAE']
            metrics['MSE'].append(f"{res['MSE']:.2f}")
            metrics['RMSE'].append(f"{res['RMSE']:.2f}")
            metrics['MAE'].append(f"{mae_val:.2f}")
            metrics['Status'].append("Success")
            if mae_val < min_mae:
                min_mae = mae_val
                best_model_name = name
    metrics_df = pd.DataFrame(metrics)
    def sort_key(series):
        return series.apply(lambda x: float(x.replace('Error', str(np.inf))))
    metrics_df = metrics_df.sort_values(by='MAE', key=sort_key, ascending=True, ignore_index=True)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    if best_model_name:
        st.success(f"🏆 Best model based on MAE/RMSE: **{best_model_name}**")
    else:
        st.warning("⚠️ Both models failed. See errors below.")
        return

    # Debugging
    st.markdown("#### Model Debugging: Detailed Errors")
    failed_models = False
    for name in ['SARIMA', 'XGBoost']:
        res = model_results.get(name)
        if res and 'Error' in res:
            st.error(f"**{name} Failure:**")
            st.code(res['Error'], language='text')
            failed_models = True
    if not failed_models:
        st.info("All models trained successfully.")

    # Visualization
    st.markdown("### 5. Historical Data, Test Predictions, and 90-Day Forecasts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_ts.index, y=train_ts.values, mode='lines', name='Historical (Train)',
                             line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=test_ts.index, y=test_ts.values, mode='lines', name='Actual (Test)',
                             line=dict(color='#007BFF', width=2)))
    colors = {'SARIMA': '#00B050', 'XGBoost': '#9933FF'}
    for name in ['SARIMA', 'XGBoost']:
        res = model_results.get(name)
        if res is None or 'Error' in res: continue
        test_pred = res.get('Test_Predictions')
        if not test_pred.empty:
            fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred.values, mode='lines',
                                     name=f'{name} Test Prediction',
                                     line=dict(color=colors[name], dash='dot', width=1)))
        future_forecast = res.get('Future_Forecast')
        if not future_forecast.empty:
            fig.add_trace(go.Scatter(x=future_forecast.index, y=future_forecast.values, mode='lines',
                                     name=f'{name} 90-Day Forecast',
                                     line=dict(color=colors[name], width=3)))
    train_end_date_str = train_ts.index.max().strftime('%Y-%m-%d')
    fig.add_vline(x=train_end_date_str, line_width=1, line_dash="dash", line_color="black")
    fig.add_annotation(x=train_end_date_str, y=1, xref="x", yref="paper", text="Train End",
                       showarrow=False, xshift=-5, yshift=10,
                       font=dict(color="black", size=12),
                       bgcolor="rgba(255, 255, 255, 0.7)",
                       bordercolor="black", borderwidth=1)
    fig.update_layout(title=f'Bread Demand Forecasting Comparison',
                      xaxis_title="Date", yaxis_title="Daily Demand (Units Sold)",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # Forecasted demand
    st.markdown(f"### 6. Forecasted Demand Values ({best_model_name})")
    st.write(f"Next {FORECAST_HORIZON_DAYS} days of **demand (units sold)** forecasted by {best_model_name}.")
    best_forecast = model_results.get(best_model_name, {}).get('Future_Forecast', pd.Series())
    if not best_forecast.empty:
        st.dataframe(best_forecast.rename('Forecasted Demand')
                     .reset_index()
                     .rename(columns={'index': 'Date'})
                     .style.format({'Forecasted Demand': '{:,.0f}'}), 
                     hide_index=True, use_container_width=True)
    else:
        st.info("No future forecast generated.")

if __name__ == '__main__':
    main()
