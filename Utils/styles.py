import streamlit as st

def apply_custom_css():
    st.markdown('''
    <style>
        /* Main background */
        .stApp {
            background-color: #0f172a;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
        }
        
        /* Headers */
        h1 {
            color: #60a5fa;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        h2 {
            color: #60a5fa;
            font-size: 1.5rem;
            margin-top: 1rem;
        }
        
        h3 {
            color: #60a5fa;
        }
        
        /* Text color */
        p, li {
            color: #cbd5e1;
        }
        
        /* Card-like containers */
        .stMarkdown {
            color: #cbd5e1;
        }
        
        /* Sidebar text */
        .sidebar .sidebar-content {
            color: #e2e8f0;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #60a5fa;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #3b82f6;
            color: white;
        }
        
        .stButton>button:hover {
            background-color: #2563eb;
        }
    </style>
    ''', unsafe_allow_html=True)