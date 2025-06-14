# data.py
import pandas as pd
from google.cloud import bigquery
from typing import Dict, Optional

import config
import utils

def load_single_table(table_name: str, client: bigquery.Client, columns: Optional[list] = None) -> Optional[pd.DataFrame]:
    """
    Loads a single table from BigQuery.
    """
    try:
        query = f"SELECT {', '.join(columns) if columns else '*'} FROM `{table_name}`"
        if config.MAX_QUERY_ROWS:
            query += f" LIMIT {config.MAX_QUERY_ROWS}"
            
        print(f"Querying table: {table_name}...")
        df = client.query(query).to_dataframe()
        print(f"Successfully loaded {len(df)} rows from {table_name}.")
        return df
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: Failed to load table {table_name}: {e}")
        import streamlit as st
        st.error(f"❌ Failed to load table {table_name}: {error_msg}")
        
        # Provide specific error information for common issues
        if "db-dtypes" in error_msg.lower():
            st.error("💡 Missing required package 'db-dtypes' for BigQuery data types. Please check requirements.txt")
        elif "not found" in error_msg.lower():
            st.error(f"💡 Table {table_name} was not found. Check if the table name and dataset are correct.")
        elif "permission" in error_msg.lower() or "access" in error_msg.lower():
            st.error(f"💡 Permission denied accessing {table_name}. Check BigQuery permissions.")
        
        return None

def load_all_tables(client: bigquery.Client) -> Dict[str, pd.DataFrame]:
    """
    Loads all necessary tables from BigQuery.
    This is now a pure data function, with no Streamlit decorators.
    """
    if not client:
        return {}

    table_data = {}
    # Use the new, clean table names from the config
    for name, table_name in config.BIGQUERY_TABLES.items():
        # Dynamically construct the full table ID here. This is the robust way.
        full_table_id = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{table_name}"
        
        loaded_table = load_single_table(full_table_id, client)
        if loaded_table is not None:
            table_data[name] = loaded_table
        else:
            # Create empty DataFrame with expected schema for failed tables
            table_data[name] = pd.DataFrame()
    
    # Always pass all tables to process_all_tables, even if empty
    # The process_all_tables function will handle empty tables properly
    return process_all_tables(table_data)

def process_all_tables(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates the processing of all loaded tables.
    This function is designed to be resilient, always returning a dictionary
    with the expected keys ('customers', 'orders', etc.), even if the
    source data is empty, preventing downstream errors.
    """
    processed_data = {}
    
    # Define all schemas that the app expects to exist.
    all_schemas = {
        "customers": config.EXPECTED_CUSTOMERS_SCHEMA,
        "orders": config.EXPECTED_ORDERS_SCHEMA,
        "products": config.EXPECTED_PRODUCTS_SCHEMA,
        "line_items": config.EXPECTED_LINE_ITEMS_SCHEMA,
        "ga_metrics": config.EXPECTED_GA_METRICS_SCHEMA,
        "ga_transactions": config.EXPECTED_GA_TRANSACTIONS_SCHEMA,
        "raw_shopify_products": ["product_id", "product_tags"] # Ad-hoc schema for the raw tags table
    }

    # --- Pre-process and validate all raw dataframes ---
    # This loop standardizes column names and ensures all expected tables exist, even if empty.
    for name, schema in all_schemas.items():
        raw_df = raw_data.get(name)
        if raw_df is not None and not raw_df.empty:
            processed_df = raw_df.copy()
            processed_df.columns = [col.lower() for col in processed_df.columns]
            processed_data[name] = processed_df
        else:
            # If data is missing or empty, create a structured empty DataFrame.
            print(f"WARNING: {name} data is empty or missing. Creating an empty, structured table.")
            processed_data[name] = pd.DataFrame(columns=[col.lower() for col in schema])
            
    # --- Main Processing & Joining Logic ---
    
    # Process customers first, as it's used in the orders join
    processed_data["customers"] = process_customers(processed_data["customers"])

    # Process core orders data by joining orders, line items, and products
    processed_data["orders"] = process_orders(
        processed_data["orders"],
        processed_data["line_items"],
        processed_data["products"]
    )
    
    # Join processed orders with customer email for final enrichment
    if not processed_data["orders"].empty and not processed_data["customers"].empty:
        customers_subset = processed_data["customers"][['customer_id', 'customer_email']]
        processed_data["orders"] = pd.merge(
            processed_data["orders"], 
            customers_subset, 
            on='customer_id', 
            how='inner' # Use inner join to ensure only orders with valid customers remain
        )

    # Process GA Metrics
    processed_data["ga_metrics"] = process_ga_metrics(processed_data["ga_metrics"])
            
    # --- Passthrough any non-defined tables that were loaded ---
    for key, df in raw_data.items():
        if key not in processed_data:
            processed_data[key] = df # Already has standardized columns if it came from the pre-processing loop
            
    return processed_data

def process_orders(df_orders: pd.DataFrame, df_line_items: pd.DataFrame, df_products: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and joins the core sales data, creating a single, clean 'orders' table.
    This is the single source of truth for order/line-item data.
    - Standardizes all column names to lowercase.
    - Intelligently joins orders with their line items, prioritizing order-level data to resolve conflicts.
    - Joins with product details.
    """
    if df_orders.empty or df_line_items.empty:
        return pd.DataFrame()

    # --- Standardize all column names to lowercase at the source ---
    orders = df_orders.copy()
    line_items = df_line_items.copy()
    products = df_products.copy()
    orders.columns = [col.lower() for col in orders.columns]
    line_items.columns = [col.lower() for col in line_items.columns]
    products.columns = [col.lower() for col in products.columns]

    # --- Step 0: Pre-processing and Cleaning ---
    # Correct known typos in column names before any processing
    if 'vartiant_stock_qty' in products.columns:
        products.rename(columns={'vartiant_stock_qty': 'variant_stock_qty'}, inplace=True)
        
    # De-duplicate source tables to prevent join explosions
    if 'order_id' in orders.columns:
        orders.drop_duplicates(subset=['order_id'], keep='last', inplace=True)
    if 'lineitem_id' in line_items.columns:
        line_items.drop_duplicates(subset=['lineitem_id'], keep='last', inplace=True)

    # --- Step 1: Merge orders and line_items ---
    if 'order_id' not in orders.columns or 'order_id' not in line_items.columns:
        print("Cannot join orders and line items: 'order_id' is missing.")
        return pd.DataFrame()
    
    merged_data = pd.merge(orders, line_items, on='order_id', how='left', suffixes=('_order', '_line'))

    # --- Step 1a: Filter out line items with invalid quantity ---
    if 'lineitem_final_qty' in merged_data.columns:
        merged_data = merged_data[merged_data['lineitem_final_qty'] > 0]

    # --- Step 2: Resolve column conflicts, prioritizing the main 'orders' table ---
    for col in orders.columns:
        if col == 'order_id':
            continue # Skip the join key
        
        order_col = f"{col}_order"
        line_col = f"{col}_line"
        
        if order_col in merged_data.columns and line_col in merged_data.columns:
            # The _order column is the source of truth. Rename it to the clean base name.
            merged_data.rename(columns={order_col: col}, inplace=True)
            # Drop the redundant column from the line_items table.
            merged_data.drop(columns=[line_col], inplace=True)

    # --- Step 3: Join with products for more detail ---
    if not products.empty and 'product_id' in merged_data.columns and 'product_id' in products.columns:
        product_cols_to_add = products.columns.difference(merged_data.columns).tolist()
        if 'product_id' not in product_cols_to_add:
            product_cols_to_add.append('product_id')
            
        # Use an INNER join to remove line items that don't have a matching product
        final_df = pd.merge(merged_data, products[product_cols_to_add], on='product_id', how='inner')
    else:
        final_df = merged_data
        print("WARNING: Could not join with product details table. Product-specific analyses might be limited.")
    
    # --- Step 4: Enrich the merged data with custom categories ---
    # This adds player_level, sport_universe, etc.
    final_df = utils.enrich_dataframe_with_categories(final_df)

    return final_df

def process_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the raw customers data.
    - Converts all column names to lowercase.
    - Converts date columns to datetime objects.
    - Fills missing values.
    """
    if df.empty:
        # If the input is empty, return it immediately to avoid errors
        return df
        
    df_processed = df.copy()
    
    # Convert date columns
    for col in ["customer_create_date", "order_create_date"]:
        if col in df_processed.columns:
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            
    # Fill missing numeric values
    for col in ["customer_total_spent_amount", "customer_total_orders"]:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(0)
            
    return df_processed

def get_all_time_first_purchase_dates(client: bigquery.Client) -> Optional[pd.Series]:
    """
    Fetches the very first purchase date for every customer from the entire dataset.
    This is crucial for accurately distinguishing new vs. returning customers.
    """
    table_name = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['orders']}"
    
    query = f"""
    -- This query finds the first order date for each customer
    SELECT
        customer_id,
        MIN(order_create_date) as first_purchase_date
    FROM
        `{table_name}`
    WHERE
        customer_id IS NOT NULL
    GROUP BY
        customer_id
    """
    
    try:
        df = client.query(query).to_dataframe()
        if df.empty:
            return pd.Series(dtype='datetime64[ns]') # Return empty series if no data
            
        # Set customer ID as the index for quick lookups
        series = df.set_index('customer_id')['first_purchase_date']
        
        # Convert to timezone-naive to prevent comparison errors with Streamlit's date picker
        if series.dt.tz is not None:
            series = series.dt.tz_localize(None)
        return series
    except Exception as e:
        print(f"ERROR: Failed to get all-time first purchase dates: {e}")
        return None

def process_ga_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and renames the GA metrics table for consistency.
    """
    df_processed = df.copy()
    df_processed.columns = [col.lower() for col in df_processed.columns]
    
    rename_map = {
        'campaign_users': 'total_users',
        'campaign_cost': 'ads_cost',
        'campaign_impressions': 'ads_impressions',
        'campaign_clicks': 'ads_clicks'
    }
    df_processed.rename(columns=rename_map, inplace=True)
    
    # Placeholder for primary_channel_group as it's not in the source data
    df_processed['primary_channel_group'] = 'N/A'
    
    return df_processed

def filter_data_by_date(data: Dict[str, pd.DataFrame], start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """
    Filters all dataframes in the dictionary based on a date range.
    It checks for common date columns like 'order_create_date' or 'campaign_date'.
    """
    filtered_data = {}
    for name, df in data.items():
        if df.empty:
            filtered_data[name] = df
            continue

        # Determine which date column to use for filtering
        date_col = None
        if 'order_create_date' in df.columns:
            date_col = 'order_create_date'
        elif 'customer_create_date' in df.columns:
            date_col = 'customer_create_date'
        elif 'campaign_date' in df.columns:
            date_col = 'campaign_date'
        
        if date_col:
            # Make a copy only if we need to modify the date column
            df_to_filter = df
            
            # Ensure the date column is in datetime format.
            # This is safer than assuming it's already converted.
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df_to_filter = df.copy()
                df_to_filter[date_col] = pd.to_datetime(df_to_filter[date_col], errors='coerce')

            # If the column from BigQuery is timezone-aware (UTC), convert it to naive
            # to allow comparison with the naive timestamps from the date picker.
            if df_to_filter[date_col].dt.tz is not None:
                # This operation also requires a copy if we haven't made one yet
                if df_to_filter is df:
                    df_to_filter = df.copy()
                df_to_filter[date_col] = df_to_filter[date_col].dt.tz_localize(None)

            # Filter the dataframe and handle potential NaT from coercion
            valid_dates = df_to_filter[date_col].notna()
            mask = (
                (df_to_filter.loc[valid_dates, date_col].dt.date >= start_date.date()) &
                (df_to_filter.loc[valid_dates, date_col].dt.date <= end_date.date())
            )
            filtered_df = df_to_filter[valid_dates][mask]
            filtered_data[name] = filtered_df
        else:
            # If no date column is found, return the original dataframe
            filtered_data[name] = df
            
    return filtered_data

# You can add more processing functions for other tables below
# def process_orders(df: pd.DataFrame) -> pd.DataFrame:
#     ... 