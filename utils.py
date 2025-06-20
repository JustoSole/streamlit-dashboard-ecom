# utils.py
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import config
from typing import Optional, List, Dict, Any, Union
import os
import json

def get_bigquery_client() -> Optional[bigquery.Client]:
    """
    Establishes a connection to Google BigQuery using Streamlit secrets for deployment
    or a local service account file for local development.

    Returns:
        Optional[bigquery.Client]: A BigQuery client object if connection is successful,
                                   otherwise None.
    """
    try:
        # Check if we have Streamlit secrets available (deployed environment)
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            try:
                # Use Streamlit secrets for deployed environment
                credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"],
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                client = bigquery.Client(credentials=credentials, project=credentials.project_id)
                return client
            except Exception as e:
                st.error(f"Failed to connect using Streamlit secrets: {str(e)}")
                return None
        
        # Fallback to local service account file for development
        service_account_path = 'service_account.json'
        if os.path.exists(service_account_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                client = bigquery.Client(credentials=credentials, project=credentials.project_id)
                st.success("✅ Connected to BigQuery using local service account")
                return client
            except Exception as e:
                st.error(f"Failed to connect using local service account: {str(e)}")
                return None
        else:
            # No credentials available
            st.error("Service account file not found at: service_account.json")
            st.info("Please ensure the file exists or set up Streamlit secrets for deployment.")
            st.info("For deployment, add your service account JSON to Streamlit secrets under 'gcp_service_account'")
            return None

    except Exception as e:
        st.error(f"Failed to connect to BigQuery: {e}", icon="🚨")
        return None

def apply_custom_css():
    """Applies custom CSS to the Streamlit app with fallback for deployment."""
    # Enhanced CSS that works in both local and deployed environments
    custom_css = """
    <style>
    /* Main dashboard font and base colors */
    .stApp {
        font-family: 'Source Sans Pro', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #F8F9FA !important;
    }

    /* Headers styling for a clean, modern look */
    h1, h2, h3 {
        color: #008080 !important; /* Teal from config.py */
        font-weight: 600 !important;
    }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.75rem !important; }
    h3 { font-size: 1.5rem !important; }
    h4 { font-size: 1.25rem !important; }
    h5 { font-size: 1.1rem !important; }

    /* Custom st.metric styling */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 18px 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        border: 1px solid #E0E0E0 !important;
        transition: box-shadow 0.3s ease-in-out, transform 0.2s ease !important;
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.08) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="metric-container"] label {
        font-weight: 500 !important;
        color: #4B5563 !important; /* Subdued label color */
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2.1rem !important;
        font-weight: 600 !important;
        color: #008080 !important; /* Teal from config.py */
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #555E67 !important;
    }

    /* Style st.tabs for a modern feel */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px !important;
        border-bottom: 2px solid #D1D5DB !important;
        padding-bottom: 0px !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        white-space: pre-wrap !important;
        background-color: transparent !important;
        border-radius: 6px 6px 0px 0px !important;
        padding: 0px 18px !important;
        margin-bottom: -2px !important;
        border: 2px solid transparent !important;
        font-weight: 500 !important;
        color: #4B5563 !important;
        transition: color 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #F3F4F6 !important;
        color: #1F2937 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #008080 !important; /* Teal from config.py */
        border-color: #D1D5DB #D1D5DB #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Consistent styling for Alerts and Infos */
    .stAlert, .stInfo {
        border-radius: 6px !important;
    }
    .stAlert[data-baseweb="alert"] p, .stInfo p {
        font-size: 0.9rem !important;
    }
    
    /* Fix sidebar contrast */
    .css-1d391kg {
        background-color: #FFFFFF !important;
    }
    
    /* Ensure proper text contrast */
    .stMarkdown, .stText {
        color: #1F2937 !important;
    }
    
    /* Fix button styling */
    .stButton > button {
        background-color: #008080 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background-color: #006666 !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """
    
    # Try to load external CSS file first, then fallback to inline CSS
    try:
        if os.path.exists("style.css"):
            with open("style.css", "r") as f:
                file_css = f"<style>{f.read()}</style>"
                st.markdown(file_css, unsafe_allow_html=True)
        else:
            # Use inline CSS for deployment
            st.markdown(custom_css, unsafe_allow_html=True)
    except Exception as e:
        # Fallback to inline CSS if file loading fails
        st.markdown(custom_css, unsafe_allow_html=True)

def safe_division(numerator: Union[float, pd.Series], denominator: Union[float, pd.Series], default: float = 0.0) -> Union[float, pd.Series]:
    """
    Safely divides two numbers or pandas Series, returning a default value for division by zero.
    Handles both scalar and vector (Series) operations robustly.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
    
    if isinstance(result, (np.ndarray, pd.Series)):
        result[~np.isfinite(result)] = default
    elif not np.isfinite(result):
        result = default
        
    return result

def format_currency(amount: float) -> str:
    """Formats a numeric value into a currency string, e.g., $1,234.56."""
    if pd.isna(amount):
        return "N/A"
    return f"${amount:,.2f}"

def format_currency_rounded(amount: float) -> str:
    """Formats a numeric value into a currency string with no decimals, e.g., $1,234."""
    if pd.isna(amount):
        return "N/A"
    return f"${amount:,.0f}"

def calculate_aov(orders_df: pd.DataFrame) -> float:
    """
    Calculates the Average Order Value (AOV) from an orders DataFrame.
    This is the standard, reusable function for this metric.
    CRITICAL: It now de-duplicates the DataFrame by order_id to prevent
    inflated results when using data that contains one row per line item.
    """
    if orders_df is None or orders_df.empty or 'order_final_total_amount' not in orders_df.columns:
        return 0.0
    
    # Ensure AOV is calculated on unique orders, not line items
    unique_orders_df = orders_df.drop_duplicates(subset=['order_id'])
    
    total_revenue = unique_orders_df['order_final_total_amount'].sum()
    total_orders = unique_orders_df['order_id'].nunique()
    
    return safe_division(total_revenue, total_orders)

def format_percentage(value: float, decimals: int = 1) -> str:
    """Formats a float as a percentage string, e.g., 75.1%."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"

def categorize_product(product_name: str) -> str:
    """Categorizes a product based on keywords found in its name."""
    if pd.isna(product_name):
        return 'Unclassified'
    
    product_lower = str(product_name).lower().strip()
    
    for category, keywords in config.PRODUCT_CATEGORIES.items():
        if any(keyword in product_lower for keyword in keywords):
            return category
    
    return 'Other'

def identify_sport_universe(product_name: str) -> str:
    """Identifies if a product belongs to the Pickleball or Padel universe."""
    if pd.isna(product_name):
        return 'Unclassified'
        
    product_lower = str(product_name).lower().strip()

    for sport, keywords in config.SPORT_KEYWORDS.items():
        if any(keyword in product_lower for keyword in keywords):
            return sport
    
    return 'Generic/Both'

def get_player_level_from_tags(tags_string: str) -> Optional[str]:
    """
    Parses a 'product_tags' string to find and extract a player level.
    e.g., "..., Level: Advanced, ..." -> "Advanced"
    """
    if pd.isna(tags_string):
        return None
    
    # Standardize to lower case for consistent matching
    tags_lower = str(tags_string).lower()
    
    if 'level:' not in tags_lower:
        return None

    # Split all tags and find the one starting with 'level:'
    for tag in tags_lower.split(','):
        tag = tag.strip()
        if tag.startswith('level:'):
            # Extract the value after 'level:'
            level_value = tag.split(':', 1)[1].strip()
            
            # It's crucial to standardize the extracted level against our config
            for standard_level, keywords in config.LEVEL_KEYWORDS.items():
                if any(kw in level_value for kw in keywords):
                    return standard_level
            
            # If no keyword matches, return the capitalized raw value as a fallback
            return level_value.capitalize()
    
    return None

def get_player_level_from_name(product_name: str) -> Optional[str]:
    """
    Determines player level by searching for keywords in the product name.
    """
    if pd.isna(product_name):
        return None
        
    product_lower = str(product_name).lower().strip()
    for level, keywords in config.LEVEL_KEYWORDS.items():
        if any(keyword in product_lower for keyword in keywords):
            return level
    return None

def enrich_dataframe_with_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a DataFrame with custom categorical columns based on product info.
    - product_category: General category (e.g., 'Rackets', 'Apparel')
    - sport_universe: Primary sport (e.g., 'Pádel', 'Pickleball')
    - player_level: Skill level (e.g., 'Beginner', 'Advanced'), now with tag parsing.
    """
    if df is None or df.empty:
        return pd.DataFrame() # Return empty frame if input is invalid

    df_enriched = df.copy()
    
    # Ensure product_name exists before applying functions that depend on it
    if 'product_name' not in df_enriched.columns:
        # If no product name, fill enriched columns with default values
        df_enriched['product_category'] = 'Unclassified'
        df_enriched['sport_universe'] = 'Unclassified'
        df_enriched['player_level'] = 'Unspecified'
        return df_enriched
    
    df_enriched['product_category'] = df_enriched['product_name'].apply(categorize_product)
    df_enriched['sport_universe'] = df_enriched['product_name'].apply(identify_sport_universe)
    
    # --- New Player Level Logic ---
    # Step 1: Attempt to get level from tags.
    # The 'product_tags' column is added in data.py and may not exist in all dataframes.
    if 'product_tags' in df_enriched.columns:
        df_enriched['level_from_tags'] = df_enriched['product_tags'].apply(get_player_level_from_tags)
    else:
        df_enriched['level_from_tags'] = None

    # Step 2: Get level from product name as a fallback.
    df_enriched['level_from_name'] = df_enriched['product_name'].apply(get_player_level_from_name)
    
    # Step 3: Coalesce the results. Prioritize tags, then name, then 'Unspecified'.
    df_enriched['player_level'] = df_enriched['level_from_tags'].fillna(df_enriched['level_from_name']).fillna('Unspecified')
    
    # Step 4: Clean up intermediate columns.
    df_enriched.drop(columns=['level_from_tags', 'level_from_name'], inplace=True)
    
    return df_enriched

def identify_recurrent_customers(all_orders_df: pd.DataFrame) -> pd.Index:
    """
    Identifies customers with more than one unique order from the complete order history.
    
    Args:
        all_orders_df (pd.DataFrame): DataFrame with the entire order history.

    Returns:
        pd.Index: An index of customer_ids for recurrent customers.
    """
    if all_orders_df.empty or 'customer_id' not in all_orders_df.columns:
        return pd.Index([])

    customer_order_counts = all_orders_df.groupby('customer_id')['order_id'].nunique()
    recurrent_customers_index = customer_order_counts[customer_order_counts > 1].index
    return recurrent_customers_index
