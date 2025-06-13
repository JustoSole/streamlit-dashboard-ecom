# components/metrics.py
import streamlit as st
from typing import List, Dict, Any

def create(metrics_data: List[Dict[str, Any]], num_columns: int = 4):
    """
    Renders a responsive layout of metrics.
    
    Args:
        metrics_data: A list of dictionaries, where each dict represents a metric.
                      e.g., [{'label': 'Total Sales', 'value': '$1M', 'delta': '5%'}]
        num_columns: The number of columns for the layout.
    """
    if not metrics_data:
        st.info("No metric data to display.")
        return
        
    cols = st.columns(num_columns)
    for i, metric in enumerate(metrics_data):
        with cols[i % num_columns]:
            st.metric(
                label=metric.get("label", "N/A"),
                value=metric.get("value", "N/A"),
                delta=metric.get("delta", None),
                delta_color=metric.get("delta_color", "normal")
            ) 