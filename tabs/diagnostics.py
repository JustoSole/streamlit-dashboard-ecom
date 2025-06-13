# tabs/diagnostics.py
import streamlit as st
import pandas as pd
from typing import Dict, List, Set, Tuple
import config

# --- Helper Functions for Checks ---

def check_schema_conformance(data_dict: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[str]]:
    """
    Compares loaded tables against expected schemas from config.
    Returns lists of error and warning messages.
    """
    errors = []
    warnings = []
    
    # Map table names to their expected schema configurations
    schema_map = {
        "orders": config.EXPECTED_ORDERS_SCHEMA,
        "customers": config.EXPECTED_CUSTOMERS_SCHEMA,
        "products": config.EXPECTED_PRODUCTS_SCHEMA,
        "line_items": config.EXPECTED_LINE_ITEMS_SCHEMA,
        "ga_metrics": config.EXPECTED_GA_METRICS_SCHEMA,
        "ga_transactions": config.EXPECTED_GA_TRANSACTIONS_SCHEMA,
    }

    for name, expected_cols_list in schema_map.items():
        if name not in data_dict or data_dict[name].empty:
            warnings.append(f"**{name}**: Table not loaded or is empty. Cannot perform schema check.")
            continue
            
        df = data_dict[name]
        actual_cols = set(df.columns)
        expected_cols = set(expected_cols_list)

        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            errors.append(f"**{name}**: Missing essential columns: `{', '.join(missing_cols)}`.")
        if extra_cols:
            warnings.append(f"**{name}**: Found extra columns: `{', '.join(extra_cols)}`.")
            
    return errors, warnings

def check_data_integrity(data_dict: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[str]]:
    """
    Performs various data integrity checks (duplicates, negative values).
    Returns lists of error and warning messages.
    """
    errors = []
    warnings = []
    
    # Check for duplicate line items in the final processed orders table
    orders = data_dict.get("orders")
    if orders is not None and not orders.empty:
        if 'lineitem_id' in orders.columns:
            # Drop temporary/placeholder null lineitem_ids before checking
            valid_line_items = orders['lineitem_id'].dropna()
            if valid_line_items.duplicated().any():
                num_dupes = valid_line_items.duplicated().sum()
                errors.append(f"**orders**: Found {num_dupes} duplicate `lineitem_id` values. This indicates a serious data integrity issue in the source line items table.")
        else:
            warnings.append("**orders**: The `lineitem_id` column is missing, cannot check for duplicate line items.")

    # Check for negative financial values
    if orders is not None and not orders.empty and 'order_final_total_amount' in orders.columns:
        if (orders['order_final_total_amount'] < 0).any():
            num_neg = (orders['order_final_total_amount'] < 0).sum()
            warnings.append(f"**orders**: Found {num_neg} orders with negative `order_final_total_amount`.")
            
    # Check for negative quantities
    line_items = data_dict.get("line_items")
    if line_items is not None and not line_items.empty and 'lineitem_final_qty' in line_items.columns:
        if (line_items['lineitem_final_qty'] <= 0).any():
            num_neg_qty = (line_items['lineitem_final_qty'] <= 0).sum()
            warnings.append(f"**line_items**: Found {num_neg_qty} line items with a quantity of 0 or less.")
            
    return errors, warnings

def check_foreign_keys(data_dict: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Checks for orphan records between tables.
    Returns a list of warning messages.
    """
    warnings = []
    
    orders = data_dict.get("orders")
    customers = data_dict.get("customers")
    line_items = data_dict.get("line_items")
    products = data_dict.get("products")

    # Check orders -> customers relationship
    if orders is not None and not orders.empty and customers is not None and not customers.empty:
        if 'customer_id' in orders.columns and 'customer_id' in customers.columns:
            orphan_orders = orders[~orders['customer_id'].isin(customers['customer_id'])]
            if not orphan_orders.empty:
                warnings.append(f"**orders/customers**: Found {len(orphan_orders)} orders with a `customer_id` that doesn't exist in the customers table.")
    
    # Check line_items -> products relationship
    if line_items is not None and not line_items.empty and products is not None and not products.empty:
        if 'product_id' in line_items.columns and 'product_id' in products.columns:
            orphan_items = line_items[~line_items['product_id'].isin(products['product_id'])]
            if not orphan_items.empty:
                warnings.append(f"**line_items/products**: Found {len(orphan_items)} line items with a `product_id` that doesn't exist in the products table.")
                
    return warnings

def check_date_columns(data_dict: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Checks date columns for correct dtype and NaT values.
    Returns a list of warning messages.
    """
    warnings = []
    date_cols_map = {
        "orders": "order_create_date",
        "customers": "customer_create_date",
        "ga_metrics": "campaign_date",
    }
    
    for name, col in date_cols_map.items():
        df = data_dict.get(name)
        if df is not None and not df.empty and col in df.columns:
            # Check for NaT values after coercion in data processing step
            if df[col].isnull().any():
                num_null = df[col].isnull().sum()
                warnings.append(f"**{name}**: Found {num_null} NULL/NaT values in the `{col}` column after date conversion.")
    return warnings


# --- Main Render Function ---

def render(all_data: Dict[str, pd.DataFrame], filtered_data: Dict[str, pd.DataFrame]):
    """Renders the complete, enhanced Diagnostics tab."""
    st.header("Data Health & Diagnostics")
    st.markdown("""
        This tab provides a comprehensive health check of your data pipeline. Use it to diagnose
        issues with data loading, schema mismatches, and data quality that could impact dashboard accuracy.
    """)

    # --- 1. Data Loading Summary ---
    with st.expander("Data Loading Summary", expanded=True):
        st.subheader("Raw Data (Before Date Filtering)")
        summary_all = []
        for name, df in all_data.items():
            status = "✅ Loaded" if df is not None and not df.empty else "❌ Empty or Not Loaded"
            rows = len(df) if df is not None else "N/A"
            summary_all.append({"Table": f"`{name}`", "Status": status, "Total Rows": rows})
        st.dataframe(pd.DataFrame(summary_all), use_container_width=True)

        st.subheader("Filtered Data (After Date Filtering)")
        summary_filtered = []
        for name, df in filtered_data.items():
            rows = len(df) if df is not None else 0
            summary_filtered.append({"Table": f"`{name}`", "Rows in Selected Period": rows})
        st.dataframe(pd.DataFrame(summary_filtered), use_container_width=True)
        if filtered_data.get("orders") is not None and filtered_data["orders"].empty:
             st.warning("The 'orders' table has no data for the selected date range. Most charts will be empty.")


    # --- 2. Health Checks ---
    st.subheader("Data Quality Health Checks")
    
    schema_errors, schema_warnings = check_schema_conformance(all_data)
    integrity_errors, integrity_warnings = check_data_integrity(filtered_data)
    fk_warnings = check_foreign_keys(filtered_data)
    date_warnings = check_date_columns(filtered_data)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Errors")
        st.markdown("These are critical issues that should be fixed.")
        all_errors = schema_errors + integrity_errors
        if not all_errors:
            st.success("No critical errors found.")
        else:
            for error in all_errors:
                st.error(error)

    with col2:
        st.subheader("🟡 Warnings")
        st.markdown("These are potential issues that might lead to incorrect analysis.")
        all_warnings = schema_warnings + integrity_warnings + fk_warnings + date_warnings
        if not all_warnings:
            st.success("No warnings found.")
        else:
            for warning in all_warnings:
                st.warning(warning)
    
    # --- 3. Detailed Schema View ---
    with st.expander("View Full Table Schemas"):
        st.markdown("Shows the column names for each data table that was successfully loaded. Use this to verify that the columns you expect are present.")
        for name, df in all_data.items():
            st.subheader(f"Table: `{name}`")
            if df is not None and not df.empty:
                st.text(f"{len(df.columns)} columns found.")
                st.dataframe(df.head(2))
            else:
                st.info("This table is empty or was not loaded.") 