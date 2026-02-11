import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Apple Stock Price Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🍎 Apple Stock Price Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("### Select Options")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("AAPL.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        return df
    except FileNotFoundError:
        st.error("❌ AAPL.csv not found in repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


# Load the data
df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.markdown("### 📅 Date Range")
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Filter data based on date range
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    filtered_df = df.loc[mask]
    
    # Sidebar visualization options
    st.sidebar.markdown("### 📊 Visualization Options")
    show_volume = st.sidebar.checkbox("Show Trading Volume", value=True)
    chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Candlestick", "Area"])
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Historical Data", "🔍 Analysis", "🤖 Model Performance", "🔮 Predictions"])
    
    # TAB 1: Historical Data
    with tab1:
        st.markdown('<h2 class="sub-header">Historical Stock Prices</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Close Price",
                value=f"${filtered_df['Close'].iloc[-1]:.2f}",
                delta=f"{((filtered_df['Close'].iloc[-1] - filtered_df['Close'].iloc[-2]) / filtered_df['Close'].iloc[-2] * 100):.2f}%"
            )
        
        with col2:
            st.metric(
                label="Period High",
                value=f"${filtered_df['High'].max():.2f}"
            )
        
        with col3:
            st.metric(
                label="Period Low",
                value=f"${filtered_df['Low'].min():.2f}"
            )
        
        with col4:
            st.metric(
                label="Average Volume",
                value=f"{filtered_df['Volume'].mean()/1e6:.2f}M"
            )
        
        # Price chart
        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=filtered_df.index,
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Close']
            )])
            fig.update_layout(
                title="Apple Stock Price - Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                template="plotly_white"
            )
        elif chart_type == "Area":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=filtered_df['Close'],
                fill='tozeroy',
                name='Close Price',
                line=dict(color='#1f77b4')
            ))
            fig.update_layout(
                title="Apple Stock Price - Area Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                template="plotly_white"
            )
        else:  # Line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Open'], mode='lines', name='Open'))
            fig.update_layout(
                title="Apple Stock Price - Line Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                template="plotly_white"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        if show_volume:
            fig_volume = px.bar(
                filtered_df,
                x=filtered_df.index,
                y='Volume',
                title="Trading Volume",
                labels={'x': 'Date', 'Volume': 'Volume'}
            )
            fig_volume.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(filtered_df, use_container_width=True)
    
    # TAB 2: Analysis
    with tab2:
        st.markdown('<h2 class="sub-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Statistical summary
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(filtered_df.describe(), use_container_width=True)
            
            # Price distribution
            st.markdown("### 📉 Close Price Distribution")
            fig_dist = px.histogram(
                filtered_df,
                x='Close',
                nbins=50,
                title="Distribution of Closing Prices"
            )
            fig_dist.update_layout(template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Correlation matrix
            st.markdown("### 🔗 Correlation Matrix")
            corr = filtered_df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu',
                title="Feature Correlation"
            )
            fig_corr.update_layout(template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Moving averages
            st.markdown("### 📈 Moving Averages")
            ma_20 = filtered_df['Close'].rolling(window=20).mean()
            ma_50 = filtered_df['Close'].rolling(window=50).mean()
            
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Close'], mode='lines', name='Close Price'))
            fig_ma.add_trace(go.Scatter(x=filtered_df.index, y=ma_20, mode='lines', name='MA 20'))
            fig_ma.add_trace(go.Scatter(x=filtered_df.index, y=ma_50, mode='lines', name='MA 50'))
            fig_ma.update_layout(
                title="Moving Averages (20 & 50 days)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white"
            )
            st.plotly_chart(fig_ma, use_container_width=True)
    
    # TAB 3: Model Performance
    with tab3:
        st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        # Model comparison data
        comparison_df = pd.DataFrame({
            "RMSE": [3.5062, 35.5952, 32.5873, 7.0225],
            "MAE":  [2.5602, 27.4110, 21.6896, 5.5460],
            "MAPE": [1.2997, np.nan, 9.3558, 2.6529]
        }, index=["Naive_Baseline", "ARIMA", "Random_Forest", "LSTM"])
        
        st.markdown("### 📊 Performance Metrics")
        st.dataframe(comparison_df.style.highlight_min(axis=0, color='lightgreen'), use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rmse = px.bar(
                comparison_df,
                y='RMSE',
                title="RMSE Comparison (Lower is Better)",
                labels={'index': 'Model', 'RMSE': 'Root Mean Squared Error'}
            )
            fig_rmse.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            fig_mae = px.bar(
                comparison_df,
                y='MAE',
                title="MAE Comparison (Lower is Better)",
                labels={'index': 'Model', 'MAE': 'Mean Absolute Error'},
                color='MAE',
                color_continuous_scale='Reds'
            )
            fig_mae.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # Best model
        best_model = comparison_df['RMSE'].idxmin()
        st.success(f"🏆 **Best Performing Model:** {best_model} with RMSE = {comparison_df.loc[best_model, 'RMSE']:.4f}")
        
        st.info("""
        **Model Explanations:**
        - **Naive Baseline**: Uses the previous day's value as prediction
        - **ARIMA**: AutoRegressive Integrated Moving Average model
        - **Random Forest**: Ensemble learning method using decision trees
        - **LSTM**: Long Short-Term Memory neural network
        """)
    
    # TAB 4: Predictions
    with tab4:
        st.markdown('<h2 class="sub-header">Price Predictions</h2>', unsafe_allow_html=True)
        
        st.sidebar.markdown("### 🔮 Prediction Settings")
        pred_days = st.sidebar.slider("Days to Predict", 1, 30, 7)
        
        # Simple prediction using naive baseline (best model)
        last_price = filtered_df['Close'].iloc[-1]
        last_date = filtered_df.index[-1]
        
        # Generate predictions (using last value as baseline)
        pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_days)
        
        # Add some random walk for demonstration
        np.random.seed(42)
        predictions = [last_price]
        for i in range(pred_days):
            # Random walk with slight upward bias
            change = np.random.normal(0.5, 3)
            predictions.append(predictions[-1] + change)
        predictions = predictions[1:]
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted_Price': predictions
        })
        pred_df.set_index('Date', inplace=True)
        
        # Visualization
        fig_pred = go.Figure()
        
        # Historical data
        fig_pred.add_trace(go.Scatter(
            x=filtered_df.index[-60:],
            y=filtered_df['Close'][-60:],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue')
        ))
        
        # Predictions
        fig_pred.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Predicted_Price'],
            mode='lines+markers',
            name='Predicted Prices',
            line=dict(color='red', dash='dash')
        ))
        
        fig_pred.update_layout(
            title=f"Stock Price Prediction - Next {pred_days} Days",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Prediction table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📋 Predicted Values")
            pred_display = pred_df.copy()
            pred_display['Predicted_Price'] = pred_display['Predicted_Price'].round(2)
            st.dataframe(pred_display, use_container_width=True)
        
        with col2:
            st.markdown("### 📌 Key Insights")
            avg_pred = pred_df['Predicted_Price'].mean()
            max_pred = pred_df['Predicted_Price'].max()
            min_pred = pred_df['Predicted_Price'].min()
            
            st.metric("Average Predicted Price", f"${avg_pred:.2f}")
            st.metric("Highest Predicted", f"${max_pred:.2f}")
            st.metric("Lowest Predicted", f"${min_pred:.2f}")
        
        st.warning("⚠️ **Disclaimer**: These predictions are for demonstration purposes only and should not be used for actual trading decisions.")

else:
    st.error("❌ Could not load data. Please ensure 'AAPL.csv' is in the same directory as this script.")
    #st.info("Expected file path: C:\\Users\\bhuva\\Downloads\\apple\\AAPL.csv")

# Footer
st.markdown("---")
st.markdown("📊 **Apple Stock Price Dashboard** | Built with Streamlit | Data: Historical AAPL Stock Prices (2012-2019)")




