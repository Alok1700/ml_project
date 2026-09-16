# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('.')

from src.data_loader import load_uac_data, create_daily_index
from src.feature_engineering import build_feature_matrix
from src.models import (
    NaivePersistenceModel, MovingAverageModel,
    ARIMAForecaster, ExpSmoothingForecaster, MLForecaster
)
from src.evaluation import evaluate_forecast, mape

st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")

st.title("🧒 Predictive Forecasting of Care Load & Placement Demand")
st.markdown("**U.S. Department of Health and Human Services — UAC Program**")

# ---------- Data Loading ----------
@st.cache_data
def load_data():
    df = load_uac_data('data/abc.csv')
    df = create_daily_index(df)
    return df

df = load_data()

# ---------- Sidebar Controls ----------
st.sidebar.header("⚙️ Forecast Configuration")
horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 60, 14)
model_choice = st.sidebar.selectbox(
    "Model",
    ["Naive Persistence", "Moving Average", "ARIMA", "Exponential Smoothing",
     "Random Forest", "Gradient Boosting", "Compare All"]
)
target = st.sidebar.selectbox(
    "Target Variable",
    ["hhs_care_load", "hhs_discharges", "cbp_care_load", "cbp_intake", "cbp_transfers_out"]
)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Historical & Forecast",
    "🔮 Discharge Demand",
    "📊 Model Comparison",
    "⚠️ Capacity Stress"
])

# ---------- Tab 1: Forecast ----------
with tab1:
    st.subheader(f"Forecast: {target.replace('_', ' ').title()}")
    
    series = df[target]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Value", f"{series.iloc[-1]:,.0f}")
    col2.metric("7-day Avg", f"{series.iloc[-7:].mean():,.0f}")
    col3.metric("30-day Trend", f"{series.iloc[-30:].mean() - series.iloc[-60:-30].mean():+,.0f}")
    
    # Fit selected model
    if model_choice == "Naive Persistence":
        model = NaivePersistenceModel()
    elif model_choice == "Moving Average":
        model = MovingAverageModel(window=7)
    elif model_choice == "ARIMA":
        model = ARIMAForecaster()
    elif model_choice == "Exponential Smoothing":
        model = ExpSmoothingForecaster()
    elif model_choice == "Random Forest":
        model = MLForecaster(model_type='rf')
    elif model_choice == "Gradient Boosting":
        model = MLForecaster(model_type='gb')
    else:
        model = None
    
    if model is not None:
        try:
            model.fit(series)
            preds = model.predict(horizon)
            
            future_dates = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=1),
                periods=horizon, freq='D'
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index[-90:], y=series.iloc[-90:],
                name='Historical', line=dict(color='#1f77b4')
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=preds,
                name='Forecast', line=dict(color='#ff7f0e', dash='dash')
            ))
            
            # Confidence interval (simple band)
            std = np.std(series.iloc[-30:]) * 0.5
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(preds + 1.96*std) + list((preds - 1.96*std)[::-1]),
                fill='toself', fillcolor='rgba(255,127,14,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'
            ))
            
            fig.update_layout(
                height=500, hovermode='x unified',
                title=f"{model_choice} Forecast — {horizon} days",
                xaxis_title="Date", yaxis_title="Children"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Model failed: {e}")

# ---------- Tab 2: Discharge Demand ----------
with tab2:
    st.subheader("🔮 Discharge (Placement) Demand Forecast")
    
    discharges = df['hhs_discharges']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Daily Discharges", f"{discharges.iloc[-1]:,.0f}")
        st.metric("7-day Avg Discharges", f"{discharges.iloc[-7:].mean():,.0f}")
    with col2:
        st.metric("Transfers In (last)", f"{df['cbp_transfers_out'].iloc[-1]:,.0f}")
        net = df['cbp_transfers_out'].iloc[-1] - discharges.iloc[-1]
        st.metric("Net Pressure", f"{net:+,.0f}", delta_color="inverse" if net > 0 else "normal")
    
    # Forecast discharges
    try:
        rf = MLForecaster(model_type='rf')
        rf.fit(discharges)
        d_preds = rf.predict(horizon)
        
        future_dates = pd.date_range(
            start=discharges.index[-1] + pd.Timedelta(days=1),
            periods=horizon, freq='D'
        )
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=discharges.index[-60:], y=discharges.iloc[-60:],
                             name='Historical Discharges', marker_color='#2ca02c'))
        fig.add_trace(go.Scatter(x=future_dates, y=d_preds,
                                 name='Forecast', line=dict(color='#d62728', dash='dash')))
        fig.update_layout(height=450, title="Discharge Demand Forecast",
                          xaxis_title="Date", yaxis_title="Discharges")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Forecast failed: {e}")

# ---------- Tab 3: Model Comparison ----------
with tab3:
    st.subheader("📊 Model Comparison (Walk-Forward Validation)")
    
    if st.button("Run Model Comparison"):
        with st.spinner("Training models..."):
            target_series = df[target]
            test_size = min(horizon, 14)
            train = target_series.iloc[:-test_size]
            test = target_series.iloc[-test_size:]
            
            models = {
                'Naive': NaivePersistenceModel(),
                'MovingAvg': MovingAverageModel(window=7),
                'ARIMA': ARIMAForecaster(),
                'ExpSmoothing': ExpSmoothingForecaster(),
                'RandomForest': MLForecaster(model_type='rf'),
                'GradientBoosting': MLForecaster(model_type='gb')
            }
            
            results = []
            for name, m in models.items():
                try:
                    m.fit(train)
                    preds = m.predict(test_size)
                    res = evaluate_forecast(test.values, preds, name)
                    results.append(res)
                except Exception as e:
                    st.warning(f"{name} failed: {e}")
            
            results_df = pd.DataFrame(results).sort_values('RMSE')
            st.dataframe(results_df, use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=results_df['model'], y=results_df['MAE'],
                                 name='MAE', marker_color='#1f77b4'))
            fig.add_trace(go.Bar(x=results_df['model'], y=results_df['RMSE'],
                                 name='RMSE', marker_color='#ff7f0e'))
            fig.update_layout(barmode='group', height=400,
                              title="Model Error Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
            best = results_df.iloc[0]
            st.success(f"🏆 Best model: **{best['model']}** (RMSE={best['RMSE']:.2f}, MAE={best['MAE']:.2f})")

# ---------- Tab 4: Capacity Stress ----------
with tab4:
    st.subheader("⚠️ Early Warning: Capacity Stress Indicators")
    
    df_feat = df.copy()
    df_feat['net_pressure'] = df_feat['cbp_transfers_out'] - df_feat['hhs_discharges']
    df_feat['pressure_7d'] = df_feat['net_pressure'].rolling(7).mean()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("HHS Care Load", "Net Pressure (Transfers - Discharges)"))
    
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['hhs_care_load'],
                             name='HHS Care Load', line=dict(color='#1f77b4')),
                  row=1, col=1)
    
    colors = ['red' if v > 0 else 'green' for v in df_feat['pressure_7d']]
    fig.add_trace(go.Bar(x=df_feat.index, y=df_feat['pressure_7d'],
                         name='Net Pressure (7d avg)', marker_color=colors),
                  row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True,
                      title="Capacity Stress Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Stress indicator
    recent_pressure = df_feat['pressure_7d'].iloc[-1]
    if recent_pressure > 10:
        st.error(f"🚨 HIGH STRESS: Net pressure is +{recent_pressure:.1f}/day. Scale up capacity.")
    elif recent_pressure > 0:
        st.warning(f"⚠️ MODERATE STRESS: Net pressure is +{recent_pressure:.1f}/day. Monitor closely.")
    else:
        st.success(f"✅ STABLE: Net pressure is {recent_pressure:.1f}/day. Discharges offset intake.")

st.sidebar.markdown("---")
st.sidebar.caption("Built for HHS UAC Program | Predictive Analytics")