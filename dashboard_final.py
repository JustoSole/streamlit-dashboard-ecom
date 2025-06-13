import streamlit as st
import pandas as pd
from google.cloud import bigquery
from typing import Dict, Optional

# Local module imports
import config
import utils
import data
from components import sidebar
# from components import charts, metrics
from tabs import overview
from tabs import frequency_analysis
from tabs import product_insights
from tabs import ltv
from tabs import channels
from tabs import diagnostics

st.set_page_config(
    page_title="Customer Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar CSS personalizado
utils.apply_custom_css()

# --- Caching Wrappers ---
# These wrappers add the Streamlit caching layer to our pure data functions.
# This is the correct way to separate concerns: data logic in data.py,
# caching and app logic here.

@st.cache_resource
def get_cached_bigquery_client() -> Optional[bigquery.Client]:
    """Cached wrapper for getting the BigQuery client."""
    return utils.get_bigquery_client()

@st.cache_data(ttl=config.CACHE_TTL)
def load_cached_data(_client: bigquery.Client) -> Dict[str, pd.DataFrame]:
    """Cached wrapper for loading all table data."""
    # The _client argument is used as a "cache key" to invalidate the cache
    # if the client object changes, but the real work is done by data.load_all_tables.
    return data.load_all_tables(_client)

def main():
    """
    Main function that orchestrates the dashboard.
    """
    st.title("📊 Customer & Performance Analytics")
    st.markdown("Welcome to the analytics dashboard. Use the menu on the left to filter the data.")

    # --- 1. Data Loading ---
    # Use the new cached wrapper functions
    client = get_cached_bigquery_client()
    if client:
        all_data = load_cached_data(client)

        if not all_data:
            st.error("Could not load any data. Check the connection and table configuration.")
            return

        # --- 2. Sidebar and Filters ---
        start_date, end_date = sidebar.create(all_data)
        
        # --- 3. Data Filtering ---
        filtered_data = data.filter_data_by_date(all_data, start_date, end_date)

        st.success(f"Data filtered from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # --- 4. Tab Rendering ---
        (
            overview_tab, 
            frequency_tab, 
            ltv_tab, 
            product_insights_tab, 
            channels_tab, 
            diagnostics_tab
        ) = st.tabs([
            "Overview", 
            "Frequency & Recurrence", 
            "LTV", 
            "Product Insights", 
            "Channels", 
            "Diagnostics"
        ])

        with overview_tab:
            overview.render(filtered_data, client, (start_date, end_date))

        with frequency_tab:
            frequency_analysis.render(all_data, filtered_data)

        with ltv_tab:
            ltv.render(filtered_data, all_data)
            
        with product_insights_tab:
            product_insights.render(filtered_data, all_data)
            
        with channels_tab:
            channels.render(filtered_data)
            
        with diagnostics_tab:
            diagnostics.render(all_data, filtered_data)


if __name__ == "__main__":
    main()
