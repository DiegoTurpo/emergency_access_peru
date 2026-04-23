"""
Emergency Healthcare Access Inequality in Peru
Streamlit application — 4 tabs as required by the assignment.
"""

import streamlit as st

st.set_page_config(
    page_title="Emergency Healthcare Access — Peru",
    page_icon="🏥",
    layout="wide",
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & Methodology",
    "📊 Static Analysis",
    "🗺️ GeoSpatial Results",
    "🔍 Interactive Exploration",
])

with tab1:
    st.header("Data & Methodology")
    st.info("Task 6 — coming soon.")

with tab2:
    st.header("Static Analysis")
    st.info("Task 4 — coming soon.")

with tab3:
    st.header("GeoSpatial Results")
    st.info("Task 5 — coming soon.")

with tab4:
    st.header("Interactive Exploration")
    st.info("Task 5 — coming soon.")
