import streamlit as st

def show():
    # Header
    st.markdown("# Indonesian Stock Insights")
    st.markdown("### AI-Driven Stock Forecasting Platform")
    st.markdown("---")

    # About section dalam box
    st.markdown('''
    <div style='background-color: #1e293b; padding: 25px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px;'>
        <h2 style='color: #60a5fa; margin-top: 0;'>About BBRI-AI</h2>
        <p style='color: #cbd5e1; line-height: 1.6;'>
        BBRI-AI is an intelligent analytics platform designed to provide deep insights into 
        Indonesian stocks, with a special focus on BBRI. By combining real-time market 
        data with advanced AI forecasting models, BBRI-AI helps investors understand 
        price movements, short-term trends, and potential future scenarios. The 
        platform offers interactive visualizations, technical indicators, and model 
        evaluation metrics so users can monitor performance, compare forecast 
        horizons, and make data-driven investment decisions with greater confidence.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Two columns layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('''
        <div style='background-color: #1e293b; padding: 25px; border-radius: 10px; border: 1px solid #334155; height: 100%;'>
            <h2 style='color: #60a5fa; margin-top: 0;'>Key Features</h2>
            <ul style='color: #cbd5e1; line-height: 1.8;'>
                <li>Price predictions for 7, 14, and 30 days</li>
                <li>Real-time market data from Yahoo Finance</li>
                <li>Interactive historical & forecasting charts</li>
                <li>Confidence interval visualization</li>
                <li>Daily auto-update data</li>
                <li>Downloadable plots & datasets</li>
                <li>Best Model: Light GBM</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div style='background-color: #1e293b; padding: 25px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px;'>
            <h2 style='color: #60a5fa; margin-top: 0;'>Technology Used</h2>
            <p style='color: #cbd5e1; line-height: 1.6;'>
            Python, Streamlit, Pandas & NumPy, Plotly visualizations, Yahoo Finance API, 
            Machine Learning model
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div style='background-color: #1e293b; padding: 25px; border-radius: 10px; border: 1px solid #334155;'>
            <h2 style='color: #60a5fa; margin-top: 0;'>Team Members</h2>
            <ul style='color: #cbd5e1; line-height: 1.8;'>
                <li>Alza Karmatul Lailiyah</li>
                <li>Anggi Rahmadillah</li>
                <li>Geoffrey Jedidiah. S</li>
                <li>Riche Chalimul Habibah</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown('''
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p>BBRI-AI © 2025 | Indonesian Stock Insights Platform</p>
    </div>
    ''', unsafe_allow_html=True)