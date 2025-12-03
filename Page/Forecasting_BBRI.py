# ==========================================
# FILE: Page/Forecasting_BBRI.py
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import joblib  # Tambahkan joblib
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import lightgbm as lgb

def load_model():
    """Load trained model and scaler"""
    try:
        # Coba beberapa kemungkinan path dan nama file
        model_paths = [
            ('best_model.pkl', ['scaler_pack.pkl', 'scaler.pkl'], 'feature_columns.pkl', 'model_metrics.pkl'),
            ('Model/best_model.pkl', ['Model/scaler_pack.pkl', 'Model/scaler.pkl'], 'Model/feature_columns.pkl', 'Model/model_metrics.pkl'),
            ('models/best_model.pkl', ['models/scaler_pack.pkl', 'models/scaler.pkl'], 'models/feature_columns.pkl', 'models/model_metrics.pkl'),
            ('../best_model.pkl', ['../scaler_pack.pkl', '../scaler.pkl'], '../feature_columns.pkl', '../model_metrics.pkl'),
        ]
        
        model, scaler_pack, feature_columns, metrics = None, None, None, None
        
        for model_path, scaler_paths, features_path, metrics_path in model_paths:
            if os.path.exists(model_path):
                # Load model dengan joblib (prioritas) atau pickle
                try:
                    model = joblib.load(model_path)
                    st.success(f"✅ Model loaded from: {model_path} (joblib)")
                except:
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        st.success(f"✅ Model loaded from: {model_path} (pickle)")
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {e}")
                        continue
                
                # Load scaler - coba beberapa nama file
                scaler_loaded = False
                for scaler_path in scaler_paths:
                    if os.path.exists(scaler_path):
                        try:
                            # Prioritas joblib
                            scaler_pack = joblib.load(scaler_path)
                            st.success(f"✅ Scaler loaded from: {scaler_path} (joblib)")
                            scaler_loaded = True
                            break
                        except:
                            try:
                                # Fallback ke pickle
                                with open(scaler_path, 'rb') as f:
                                    scaler_pack = pickle.load(f)
                                st.success(f"✅ Scaler loaded from: {scaler_path} (pickle)")
                                scaler_loaded = True
                                break
                            except Exception as e:
                                st.warning(f"⚠️ Failed to load {scaler_path}: {e}")
                                continue
                
                if not scaler_loaded:
                    st.warning("⚠️ Could not load scaler, will proceed without scaling")
                    scaler_pack = None
                
                # Load feature columns dengan joblib (prioritas) atau pickle
                if os.path.exists(features_path):
                    try:
                        feature_columns = joblib.load(features_path)
                        st.success(f"✅ Feature columns loaded: {len(feature_columns)} features (joblib)")
                    except:
                        try:
                            with open(features_path, 'rb') as f:
                                feature_columns = pickle.load(f)
                            st.success(f"✅ Feature columns loaded: {len(feature_columns)} features (pickle)")
                        except Exception as e:
                            st.warning(f"⚠️ Failed to load feature_columns: {e}")
                else:
                    st.warning("⚠️ feature_columns.pkl not found")
                
                # Load metrics (optional)
                if os.path.exists(metrics_path):
                    try:
                        metrics = joblib.load(metrics_path)
                        st.success(f"✅ Model metrics loaded (RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}, MAPE: {metrics['MAPE']}%)")
                    except Exception as e:
                        st.info(f"ℹ️ Could not load metrics: {e}")
                        metrics = None
                else:
                    st.info("ℹ️ model_metrics.pkl not found, using placeholder values")
                    metrics = None
                
                return model, scaler_pack, feature_columns, metrics
        
        # Jika tidak ada yang cocok
        st.error(f"❌ Model files not found in any of the expected locations")
        st.info(f"📁 Current working directory: {os.getcwd()}")
        
        # Tampilkan file yang ada
        try:
            files = os.listdir('.')
            pkl_files = [f for f in files if f.endswith('.pkl')]
            if pkl_files:
                st.info(f"📦 PKL files found: {', '.join(pkl_files)}")
            else:
                st.info("❌ No .pkl files found in current directory")
        except:
            pass
            
        return None, None, None, None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

def prepare_features(data, feature_columns):
    """Prepare features for prediction - must match training exactly"""
    df = data.copy()
    
    # Pastikan kolom dasar dalam lowercase (sesuai feature_columns)
    df.columns = df.columns.str.lower()
    
    # 1. Basic OHLCV features (sudah ada dari Yahoo Finance)
    # high, low, open, volume, close
    
    # 2. Returns
    df['returns'] = df['close'].pct_change()
    
    # 3. Moving Averages - RECALCULATE every time
    df['MA7'] = df['close'].rolling(window=7, min_periods=1).mean()
    df['MA14'] = df['close'].rolling(window=14, min_periods=1).mean()
    df['MA30'] = df['close'].rolling(window=30, min_periods=1).mean()
    
    # 4. RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # 5. MACD (Moving Average Convergence Divergence)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 6. Lag features (previous day prices) - RECALCULATE every time
    df['lag1'] = df['close'].shift(1)
    df['lag3'] = df['close'].shift(3)
    df['lag7'] = df['close'].shift(7)
    
    # Fill NaN in lag features with the last available value
    df['lag1'] = df['lag1'].fillna(method='ffill')
    df['lag3'] = df['lag3'].fillna(method='ffill')
    df['lag7'] = df['lag7'].fillna(method='ffill')
    
    # Fill remaining NaN with 0
    df = df.fillna(0)
    
    # PASTIKAN SEMUA feature_columns ada (jika tidak ada, isi dengan 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            st.warning(f"⚠️ Feature '{col}' not found, filled with 0")
    
    # Return HANYA kolom yang ada di feature_columns, dengan URUTAN YANG SAMA
    return df[feature_columns]

def generate_forecast(model, scaler_pack, feature_columns, horizon_days):
    """Generate forecast for specified horizon"""
    try:
        # Get historical data
        ticker = "BBRI.JK"
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="3mo")  # Get 3 months of data
        
        if len(hist_data) == 0:
            return None, None, None
        
        # Get last known price
        last_price = hist_data['Close'].iloc[-1]
        
        # DEBUG: Tampilkan info data
        with st.expander("🔍 Debug Info - Click to expand"):
            st.write("**Historical Data Info:**")
            st.write(f"- Last Close Price: {last_price:,.0f}")
            st.write(f"- Data points: {len(hist_data)}")
            st.write(f"- Date range: {hist_data.index[0]} to {hist_data.index[-1]}")
            st.write(f"- Price range: {hist_data['Close'].min():,.0f} - {hist_data['Close'].max():,.0f}")
            
            st.write("\n**Feature Columns Expected:**")
            st.write(feature_columns)
        
        # Prepare features
        features = prepare_features(hist_data, feature_columns)
        
        # DEBUG: Tampilkan features yang dibuat
        with st.expander("🔍 Debug Info - Features"):
            st.write("**Generated Features (last 5 rows):**")
            st.dataframe(features.tail())
            st.write(f"\n**Feature Stats:**")
            st.dataframe(features.describe())
        
        # Get scaler_X and scaler_y dari scaler_pack (dictionary)
        if scaler_pack is not None and isinstance(scaler_pack, dict):
            scaler_X = scaler_pack.get('scaler_X')
            scaler_y = scaler_pack.get('scaler_y')
            
            # DEBUG: Info scaler
            with st.expander("🔍 Debug Info - Scalers"):
                st.write("**Scaler X Info:**")
                if scaler_X is not None:
                    st.write(f"- Type: {type(scaler_X)}")
                    st.write(f"- Feature range: {scaler_X.feature_range}")
                    st.write(f"- Data min: {scaler_X.data_min_[:5]}... (first 5)")
                    st.write(f"- Data max: {scaler_X.data_max_[:5]}... (first 5)")
                
                st.write("\n**Scaler Y Info:**")
                if scaler_y is not None:
                    st.write(f"- Type: {type(scaler_y)}")
                    st.write(f"- Feature range: {scaler_y.feature_range}")
                    st.write(f"- Data min: {scaler_y.data_min_}")
                    st.write(f"- Data max: {scaler_y.data_max_}")
        else:
            scaler_X = scaler_pack
            scaler_y = None
        
        # Generate predictions with iterative approach
        predictions = []
        predictions_scaled = []
        lower_bounds = []
        upper_bounds = []
        
        # Create a copy of historical data for iterative prediction
        forecast_data = hist_data.copy()
        # Lowercase all column names from the start
        forecast_data.columns = forecast_data.columns.str.lower()
        
        # Get last known price
        last_price = forecast_data['close'].iloc[-1]
        
        # DEBUG: Track iterations
        iteration_info = []
        
        for i in range(horizon_days):
            # Prepare features from current data
            current_features = prepare_features(forecast_data, feature_columns)
            
            if len(current_features) == 0:
                st.error(f"No features generated for iteration {i}")
                break
            
            # Use last available features for prediction
            X = current_features.iloc[-1:].values
            
            # DEBUG: Track feature values
            iteration_info.append({
                'day': i+1,
                'lag1': current_features['lag1'].iloc[-1],
                'close_in_data': forecast_data['close'].iloc[-1],
                'ma7': current_features['MA7'].iloc[-1]
            })
            
            # Scale features jika scaler tersedia
            if scaler_X is not None:
                X_scaled = scaler_X.transform(X)
            else:
                X_scaled = X
            
            # Predict
            pred_raw = model.predict(X_scaled)[0]
            predictions_scaled.append(pred_raw)
            
            # Cek apakah hasil prediksi sudah dalam skala asli atau masih scaled
            if pred_raw > 100:
                pred = pred_raw
            else:
                if scaler_y is not None:
                    pred = scaler_y.inverse_transform([[pred_raw]])[0][0]
                else:
                    pred = pred_raw
            
            predictions.append(pred)
            
            # Calculate confidence intervals with increasing uncertainty
            base_std = forecast_data['close'].std() * 0.01  # 1% base std
            uncertainty = base_std * np.sqrt(i + 1)  # Increase with sqrt(time)
            lower_bounds.append(pred - uncertainty)
            upper_bounds.append(pred + uncertainty)
            
            # UPDATE: Add predicted price to historical data for next iteration
            next_date = forecast_data.index[-1] + timedelta(days=1)
            
            # Create new row with predicted values
            # Assume High/Low/Open based on predicted close with small variation
            price_variation = pred * 0.005  # 0.5% variation
            new_row = pd.DataFrame({
                'open': [pred - price_variation],
                'high': [pred + price_variation],
                'low': [pred - price_variation],
                'close': [pred],
                'volume': [forecast_data['volume'].iloc[-5:].mean()]  # Use avg of last 5 days
            }, index=[next_date])
            
            # Append to forecast_data
            forecast_data = pd.concat([forecast_data, new_row])
        
        # DEBUG: Show iteration tracking
        with st.expander("🔍 Debug Info - Iteration Tracking"):
            st.write("**Feature Evolution Across Iterations:**")
            iter_df = pd.DataFrame(iteration_info)
            st.dataframe(iter_df)
            
            st.write("\n**Analysis:**")
            if len(iter_df) > 1:
                lag1_change = iter_df['lag1'].iloc[-1] - iter_df['lag1'].iloc[0]
                st.write(f"- Lag1 changed by: {lag1_change:.2f}")
                st.write(f"- Close values range: {iter_df['close_in_data'].min():.2f} - {iter_df['close_in_data'].max():.2f}")
                
                # Check if predictions are changing
                pred_std = np.std(predictions)
                st.write(f"- Prediction std deviation: {pred_std:.2f}")
                if pred_std < 1:
                    st.warning("⚠️ Predictions are not varying much!")
            else:
                st.write("- Only one iteration completed")
        
        # DEBUG: Tampilkan predictions
        with st.expander("🔍 Debug Info - Predictions"):
            st.write("**Raw Predictions (scaled):**")
            st.write(predictions_scaled[:5])
            st.write("\n**Inverse Transformed Predictions:**")
            st.write(predictions[:5])
            st.write(f"\n**Prediction Range:**")
            st.write(f"- Min: {min(predictions):,.0f}")
            st.write(f"- Max: {max(predictions):,.0f}")
            st.write(f"- Mean: {np.mean(predictions):,.0f}")
            st.write(f"\n**Expected vs Predicted:**")
            st.write(f"- Last Close: {last_price:,.0f}")
            st.write(f"- First Prediction: {predictions[0]:,.0f}")
            st.write(f"- Difference: {predictions[0] - last_price:,.0f} ({((predictions[0] - last_price) / last_price * 100):.2f}%)")
        
        # Create forecast dates
        last_date = hist_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=horizon_days, freq='D')
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted': predictions,
            'Lower Bound': lower_bounds,
            'Upper Bound': upper_bounds
        })
        
        return hist_data, forecast_df, last_price
        
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

def show():
    st.markdown("# Forecasting BBRI")
    st.markdown("### Future Price Projection Powered by AI")
    st.markdown("---")
    
    # Load model
    model, scaler_pack, feature_columns, metrics = load_model()
    
    if model is None:
        st.warning("⚠️ Model files not found. Please ensure best_model.pkl, scaler_pack.pkl, and feature_columns.pkl are in the project directory.")
        return
    
    # Top controls
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        if st.button("🔄 Predict", use_container_width=True, type="primary"):
            st.session_state.generate_forecast = True
    
    with col2:
        auto_update = st.checkbox("Auto-update", value=False)
    
    with col3:
        st.markdown(f"""
        <div style='text-align: right; color: #94a3b8; padding-top: 8px;'>
            Last updated: {datetime.now().strftime('%Y-%m-%d')}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Horizon selection
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.markdown("<h3 style='color: #cbd5e1; margin-top: 10px;'>Horizon:</h3>", unsafe_allow_html=True)
    
    with col2:
        horizon = st.radio(
            "Horizon",
            ["7D", "14D", "30D"],
            horizontal=True,
            label_visibility="collapsed",
            index=0
        )
    
    # Map horizon to days
    horizon_map = {"7D": 7, "14D": 14, "30D": 30}
    horizon_days = horizon_map[horizon]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'generate_forecast' not in st.session_state:
        st.session_state.generate_forecast = False
    
    # Generate forecast
    if st.session_state.generate_forecast or auto_update:
        with st.spinner('Generating forecast...'):
            hist_data, forecast_df, last_price = generate_forecast(
                model, scaler_pack, feature_columns, horizon_days
            )
        
        if hist_data is not None and forecast_df is not None:
            # Calculate metrics
            forecast_end_price = forecast_df['Forecasted'].iloc[-1]
            price_change = forecast_end_price - last_price
            price_change_pct = (price_change / last_price) * 100
            avg_daily_change = forecast_df['Forecasted'].pct_change().mean() * 100
            
            # Get last close data
            last_close = last_price
            last_volume = hist_data['Volume'].iloc[-1]
            last_change_pct = hist_data['Close'].pct_change().iloc[-1] * 100
            
            # Display summary cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155;'>
                    <h3 style='color: #60a5fa; margin: 0 0 15px 0;'>Today Overview</h3>
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #94a3b8;'>Last Close</span>
                        <span style='color: #e2e8f0; float: right; font-weight: bold;'>{last_close:,.0f}</span>
                    </div>
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #94a3b8;'>Change</span>
                        <span style='color: {"#22c55e" if last_change_pct >= 0 else "#ef4444"}; float: right; font-weight: bold;'>{"+" if last_change_pct >= 0 else ""}{last_change_pct:.2f}%</span>
                    </div>
                    <div>
                        <span style='color: #94a3b8;'>Volume</span>
                        <span style='color: #e2e8f0; float: right; font-weight: bold;'>{last_volume/1e6:.1f}M</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155;'>
                    <h3 style='color: #60a5fa; margin: 0 0 15px 0;'>Forecast Summary</h3>
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #94a3b8;'>Horizon: {horizon_days} Days</span>
                        <span style='color: #e2e8f0; float: right; font-weight: bold;'>{forecast_end_price:,.0f}</span>
                    </div>
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #94a3b8;'>End Price</span>
                        <span style='color: {"#22c55e" if price_change_pct >= 0 else "#ef4444"}; float: right; font-weight: bold;'>{price_change_pct:.2f}%</span>
                    </div>
                    <div>
                        <span style='color: #94a3b8;'>Avg Daily Change</span>
                        <span style='color: {"#22c55e" if avg_daily_change >= 0 else "#ef4444"}; float: right; font-weight: bold;'>{"+" if avg_daily_change >= 0 else ""}{avg_daily_change:.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Model Evaluation - gunakan nilai dari notebook atau dari file
                if metrics is not None:
                    rmse_val = f"{metrics['RMSE']:.2f}"
                    mae_val = f"{metrics['MAE']:.2f}"
                    mape_val = f"{metrics['MAPE']:.2f}%"
                else:
                    # Nilai dari notebook evaluation (hardcoded)
                    rmse_val = "32.30"
                    mae_val = "24.97"
                    mape_val = "0.65%"
                
                st.markdown(f"""
                <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155;'>
                    <h3 style='color: #60a5fa; margin: 0 0 15px 0;'>Model Evaluation</h3>
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #94a3b8;'>RMSE</span>
                        <span style='color: #e2e8f0; float: right; font-weight: bold;'>{rmse_val}</span>
                    </div>
                    <div style='margin-bottom: 10px;'>
                        <span style='color: #94a3b8;'>MAE</span>
                        <span style='color: #e2e8f0; float: right; font-weight: bold;'>{mae_val}</span>
                    </div>
                    <div>
                        <span style='color: #94a3b8;'>MAPE</span>
                        <span style='color: #e2e8f0; float: right; font-weight: bold;'>{mape_val}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Model info
            st.markdown(f"""
            <div style='text-align: right; color: #94a3b8; font-size: 0.9rem; margin-bottom: 10px;'>
                Model Used: LightGBM
            </div>
            """, unsafe_allow_html=True)
            
            # Chart and table in two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<h2 style='color: #60a5fa; margin-bottom: 15px;'>Historical vs Forecasted Price</h2>", unsafe_allow_html=True)
                
                # Create interactive chart
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=hist_data.index[-30:],  # Last 30 days
                    y=hist_data['Close'][-30:],
                    name='Historical',
                    line=dict(color='#22c55e', width=2),
                    mode='lines'
                ))
                
                # Forecasted data
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Forecasted'],
                    name='Forecasted',
                    line=dict(color='#3b82f6', width=2),
                    mode='lines'
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                    y=forecast_df['Upper Bound'].tolist() + forecast_df['Lower Bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30, 41, 59, 1)',
                    font=dict(color='#cbd5e1'),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)',
                        showline=False
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)',
                        showline=False
                    ),
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0),
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("<h3 style='color: #60a5fa; margin-bottom: 15px;'>Forecast Table</h3>", unsafe_allow_html=True)
                
                # Prepare table data
                table_df = forecast_df.copy()
                table_df['Date'] = table_df['Date'].dt.strftime('%d/%m/%y')
                table_df['Forecasted'] = table_df['Forecasted'].round(0).astype(int)
                table_df['Lower Bound'] = table_df['Lower Bound'].round(0).astype(int)
                table_df['Upper Bound'] = table_df['Upper Bound'].round(0).astype(int)
                
                # Display table
                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Prepare CSV
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"bbri_forecast_{horizon}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    label="📊 Download Plot",
                    data="",
                    file_name="forecast_plot.png",
                    mime="image/png",
                    use_container_width=True,
                    disabled=True
                )
    
    else:
        st.info("👆 Click the 'Predict' button to generate forecast")
    
    # Footer
    st.markdown("---")
    st.markdown('''
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p>BBRI-AI © 2024 | Indonesian Stock Insights Platform</p>
    </div>
    ''', unsafe_allow_html=True)


# ==========================================
# Struktur Folder yang Dibutuhkan:
# ==========================================

"""
BBRI-AI/
├── streamlit_app.py
├── best_model.pkl          # Model LightGBM Anda
├── scaler_pack.pkl         # Scaler untuk preprocessing
├── feature_columns.pkl     # List kolom fitur
├── Page/
│   ├── __init__.py
│   ├── Dashboard.py
│   ├── Market_overview.py
│   └── Forecasting_BBRI.py  # File ini
└── Utils/
    └── styles.py

CATATAN PENTING:
- Pastikan file pkl (best_model.pkl, scaler_pack.pkl, feature_columns.pkl) 
  berada di root folder project (sejajar dengan streamlit_app.py)
- Sesuaikan fungsi prepare_features() dengan feature engineering yang 
  Anda gunakan saat training model
"""