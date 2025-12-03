import streamlit as st
from Utils.styles import apply_custom_css

# Import halaman-halaman
from Page import Dashboard, Market_overview, Forecasting_BBRI

# Page configuration
st.set_page_config(
    page_title="BBRI-AI - Indonesian Stock Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'

# Sidebar
with st.sidebar:
    # Logo dan Title dengan styling
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("logo bbri.png", width=60)
    with col2:
        st.markdown('''
        <div style='padding-top: 10px;'>
            <span style='font-size: 28px; font-weight: bold; color: #60a5fa;'>BBRI-AI</span>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation buttons
    if st.button("🏠 Dashboard", use_container_width=True, 
                 type="primary" if st.session_state.current_page == 'Dashboard' else "secondary"):
        st.session_state.current_page = 'Dashboard'
        st.rerun()
    
    if st.button("📈 Market Overview", use_container_width=True,
                 type="primary" if st.session_state.current_page == 'Market Overview' else "secondary"):
        st.session_state.current_page = 'Market Overview'
        st.rerun()
    
    if st.button("🔮 Forecasting BBRI", use_container_width=True,
                 type="primary" if st.session_state.current_page == 'Forecasting BBRI' else "secondary"):
        st.session_state.current_page = 'Forecasting BBRI'
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("AI-Driven Stock Forecasting Platform")

# Route to appropriate page
if st.session_state.current_page == 'Dashboard':
    Dashboard.show()
elif st.session_state.current_page == 'Market Overview':
    Market_overview.show()
elif st.session_state.current_page == 'Forecasting BBRI':
    Forecasting_BBRI.show()