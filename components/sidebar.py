# components/sidebar.py
import streamlit as st
import pandas as pd
from typing import Dict, Tuple

def create(data: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Creates the sidebar with all global filters.
    
    Args:
        data: A dictionary of all dataframes, used to determine min/max dates.
        
    Returns:
        A tuple containing the selected start and end dates.
    """
    st.sidebar.header("Global Filters")

    # --- Date Range Selector ---
    # Find the overall min and max dates across all relevant tables
    min_dates = []
    max_dates = []
    
    # Corrected to use lowercase column names as standardized in data.py
    date_cols = ['order_create_date', 'campaign_date', 'customer_create_date']
    for df_name, df in data.items():
        # It's safer to check against lowercase columns
        df.columns = [str(c).lower() for c in df.columns]
        for col in date_cols:
            if col in df.columns:
                # Ensure the column is in datetime format before getting min/max
                date_series = pd.to_datetime(df[col], errors='coerce')
                if not date_series.empty:
                    min_dates.append(date_series.min())
                    max_dates.append(date_series.max())
    
    # Filter out NaT values before finding min/max
    min_dates = [d for d in min_dates if pd.notna(d)]
    max_dates = [d for d in max_dates if pd.notna(d)]

    if not min_dates or not max_dates:
        st.sidebar.warning("No date information available to create a date filter.")
        # Return default values if no dates are found
        return pd.to_datetime("2023-01-01"), pd.to_datetime("today")

    overall_min_date = min(min_dates)
    overall_max_date = max(max_dates)

    selected_start_date = st.sidebar.date_input(
        "Start Date",
        value=overall_min_date,
        min_value=overall_min_date,
        max_value=overall_max_date
    )

    selected_end_date = st.sidebar.date_input(
        "End Date",
        value=overall_max_date,
        min_value=overall_min_date,
        max_value=overall_max_date
    )
    
    if selected_start_date > selected_end_date:
        st.sidebar.error("Error: Start date cannot be after end date.")
        # Fallback to valid dates
        return overall_min_date, overall_max_date

    return pd.to_datetime(selected_start_date), pd.to_datetime(selected_end_date) 