import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show():
    st.markdown("# Market Overview")
    st.markdown("### Live Market Data & Intraday Trends")
    st.markdown("---")

    # Dropdown untuk memilih ticker
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Select Ticker :")
    
    with col2:
        ticker_options = {
            "BBRI.JK": "BBRI - Bank Rakyat Indonesia",
            "BMRI.JK": "BMRI - Bank Mandiri",
            "BBNI.JK": "BBNI - Bank Negara Indonesia",
            "ASII.JK": "ASII - Astra International",
            "TLKM.JK": "TLKM - Telkom Indonesia",
            "ICBP.JK": "ICBP - Indofood CBP",
            "UNVR.JK": "UNVR - Unilever Indonesia",
            "ANTM.JK": "ANTM - Aneka Tambang",
            "PGAS.JK": "PGAS - Perusahaan Gas Negara",
            "MEDC.JK": "MEDC - Medco Energi"
        }
        
        selected_ticker = st.selectbox(
            "Select Ticker",
            options=list(ticker_options.keys()),
            format_func=lambda x: ticker_options[x],
            label_visibility="collapsed",
            index=0  # Default ke BBRI.JK (index 0)
        )
    
    st.markdown("---")
    
    # Fetch data dari Yahoo Finance
    try:
        # Get real-time data
        stock = yf.Ticker(selected_ticker)
        
        # Get today's data
        today_data = stock.history(period="1d", interval="1m")
        
        # Get info
        info = stock.info
        
        # Current price (last close)
        current_price = today_data['Close'].iloc[-1] if len(today_data) > 0 else info.get('regularMarketPrice', 0)
        open_price = today_data['Open'].iloc[0] if len(today_data) > 0 else info.get('regularMarketOpen', 0)
        high_price = today_data['High'].max() if len(today_data) > 0 else info.get('dayHigh', 0)
        low_price = today_data['Low'].min() if len(today_data) > 0 else info.get('dayLow', 0)
        volume = today_data['Volume'].sum() if len(today_data) > 0 else info.get('volume', 0)
        
        # Calculate change percentage
        change = current_price - open_price
        change_percent = (change / open_price * 100) if open_price > 0 else 0
        
        # Display metrics in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; text-align: center;'>
                <h3 style='color: #60a5fa; margin: 0;'>Current Price</h3>
                <h1 style='color: #e2e8f0; margin: 10px 0; font-size: 3rem;'>{current_price:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; text-align: center;'>
                <h3 style='color: #60a5fa; margin: 0;'>Open</h3>
                <h1 style='color: #e2e8f0; margin: 10px 0; font-size: 3rem;'>{open_price:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            volume_display = f"{volume/1e6:.2f} M" if volume >= 1e6 else f"{volume/1e3:.2f} K"
            st.markdown(f"""
            <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; text-align: center;'>
                <h3 style='color: #60a5fa; margin: 0;'>Volume</h3>
                <h1 style='color: #e2e8f0; margin: 10px 0; font-size: 3rem;'>{volume_display}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Second row of metrics
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Low and High in one card
            st.markdown(f"""
            <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155;'>
                <div style='display: flex; justify-content: space-around;'>
                    <div style='text-align: center;'>
                        <h3 style='color: #60a5fa; margin: 0;'>Low</h3>
                        <h1 style='color: #e2e8f0; margin: 10px 0; font-size: 2.5rem;'>{low_price:,.0f}</h1>
                    </div>
                    <div style='text-align: center;'>
                        <h3 style='color: #60a5fa; margin: 0;'>High</h3>
                        <h1 style='color: #e2e8f0; margin: 10px 0; font-size: 2.5rem;'>{high_price:,.0f}</h1>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Change percentage card
            change_color = "#22c55e" if change_percent >= 0 else "#ef4444"
            change_sign = "+" if change_percent >= 0 else ""
            st.markdown(f"""
            <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; text-align: center;'>
                <h3 style='color: #60a5fa; margin: 0;'>Change %</h3>
                <h1 style='color: {change_color}; margin: 10px 0; font-size: 3rem;'>{change_sign}{change_percent:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Price Trend Chart
            st.markdown("""
            <h2 style='color: #60a5fa; margin: 0 0 20px 0; font-size: 1.8rem;'>Price Trend</h2>
            """, unsafe_allow_html=True)
            
            # Get historical data for different periods
            period_option = st.radio(
                "Period",
                ["1D", "1W", "1M", "1Y"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            # Map period to yfinance format
            period_map = {
                "1D": ("1d", "5m"),
                "1W": ("5d", "30m"),
                "1M": ("1mo", "1d"),
                "1Y": ("1y", "1d")
            }
            
            period, interval = period_map[period_option]
            hist_data = stock.history(period=period, interval=interval)
            
            if len(hist_data) > 0:
                # Create area chart with Plotly
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Close'],
                    fill='tozeroy',
                    fillcolor='rgba(34, 197, 94, 0.3)',
                    line=dict(color='rgb(34, 197, 94)', width=2),
                    mode='lines',
                    name='Price'
                ))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1'),
                    xaxis=dict(
                        showgrid=False,
                        showline=False,
                        zeroline=False
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)',
                        showline=False,
                        zeroline=False
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    showlegend=False,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected period")
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.info("Please make sure you have internet connection and the ticker symbol is correct.")
    
    # Footer
    st.markdown("---")
    st.markdown('''
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p>BBRI-AI © 2025 | Indonesian Stock Insights Platform</p>
        <p style='font-size: 0.9rem;'>Data provided by Yahoo Finance • Updates every minute</p>
    </div>
    ''', unsafe_allow_html=True)