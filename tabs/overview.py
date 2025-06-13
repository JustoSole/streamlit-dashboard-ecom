# tabs/overview.py
import streamlit as st
import pandas as pd
from typing import Dict, Tuple, Optional
from google.cloud import bigquery
from components import metrics, charts
import utils
import data

def get_overview_kpis(orders_df: Optional[pd.DataFrame], first_purchases: Optional[pd.Series], date_range: Tuple[pd.Timestamp, pd.Timestamp]) -> Dict:
    """
    Calculates key performance indicators for the overview tab.
    
    Args:
        orders_df (pd.DataFrame): DataFrame with order data for the selected period.
        first_purchases (pd.Series): Series with the first purchase date for every customer.
        date_range (Tuple): The start and end date of the selected period.

    Returns:
        Dict: A dictionary containing all calculated KPIs.
    """
    if orders_df is None or orders_df.empty:
        return {}

    start_date, end_date = date_range
    
    # De-duplicate to ensure all calculations are based on unique orders, not line items
    unique_orders_df = orders_df.drop_duplicates(subset=['order_id'])
    
    total_revenue = unique_orders_df['order_final_total_amount'].sum()
    total_orders = unique_orders_df['order_id'].nunique()
    total_customers = unique_orders_df['customer_id'].nunique()
    
    # Calculate New vs. Returning Customers
    new_customers = 0
    if first_purchases is not None and not first_purchases.empty:
        customers_in_period = unique_orders_df['customer_id'].unique()
        # A customer is new if their first-ever purchase is within the selected date range
        new_customer_ids = first_purchases[
            (first_purchases.dt.date >= start_date.date()) &
            (first_purchases.dt.date <= end_date.date())
        ].index
        
        # The intersection of customers in this period and all-time new customers gives the true new count
        new_customers = len(set(customers_in_period) & set(new_customer_ids))

    return {
        'Total Revenue': utils.format_currency_rounded(total_revenue),
        'Total Orders': f"{total_orders:,}",
        'New Customers': f"{new_customers:,}" if first_purchases is not None else "N/A",
        'Avg. Order Value': utils.format_currency(utils.calculate_aov(orders_df)),
    }

def display_monthly_trends(orders_df: pd.DataFrame, all_time_first_purchases: Optional[pd.Series]):
    """
    Displays monthly trend charts for key metrics like revenue, orders, and customers.
    Now uses the all-time first purchase data for consistent new customer calculation.
    """
    st.subheader("Monthly Performance Trends")
    if orders_df is None or orders_df.empty:
        st.info("No data available to display monthly trends.")
        return

    df = orders_df.copy()
    df['order_create_date'] = pd.to_datetime(df['order_create_date'])
    
    # --- Aggregate core metrics ---
    # CRITICAL FIX: To avoid inflating revenue, first get unique orders before aggregating.
    unique_orders_df = df.drop_duplicates(subset=['order_id'])
    
    monthly_data = unique_orders_df.set_index('order_create_date').resample('ME').agg(
        total_revenue=('order_final_total_amount', 'sum'),
        total_orders=('order_id', 'nunique'),
        unique_customers=('customer_id', 'nunique')
    ).reset_index()
    monthly_data.rename(columns={'order_create_date': 'month'}, inplace=True)
    
    if monthly_data.empty:
        st.info("Not enough data to generate monthly trend charts.")
        return

    # --- Charting Section ---
    col1, col2 = st.columns(2)
    with col1:
        fig_revenue = charts.create_bar_chart(monthly_data, x='month', y='total_revenue', title="Monthly Net Revenue")
        charts.safe_plotly_chart(fig_revenue)
        
        fig_customers = charts.create_line_chart(monthly_data, x='month', y='unique_customers', title="Monthly Unique Customers")
        charts.safe_plotly_chart(fig_customers)
        
    with col2:
        fig_orders = charts.create_bar_chart(monthly_data, x='month', y='total_orders', title="Monthly Net Orders")
        charts.safe_plotly_chart(fig_orders)

        if all_time_first_purchases is not None and not all_time_first_purchases.empty:
            # Join the all-time first purchase date to each order
            df_with_first_purchase = df.join(all_time_first_purchases.rename('first_purchase_date'), on='customer_id')
            
            # An order is from a "new" customer if its date matches their all-time first purchase date
            new_customer_orders = df_with_first_purchase[
                df_with_first_purchase['order_create_date'].dt.date == df_with_first_purchase['first_purchase_date'].dt.date
            ]
            
            # Aggregate new customers by month
            new_customer_monthly = new_customer_orders.set_index('order_create_date').resample('ME')['customer_id'].nunique().reset_index()
            new_customer_monthly.rename(columns={'order_create_date': 'month', 'customer_id': 'new_customers'}, inplace=True)

            fig_new_customers = charts.create_bar_chart(
                new_customer_monthly, 
                x='month', 
                y='new_customers',
                title='Monthly New Customers'
            )
            charts.safe_plotly_chart(fig_new_customers)
        else:
            st.info("First purchase data not available to plot new customers over time.")

def render(filtered_data: Dict[str, pd.DataFrame], client: bigquery.Client, date_range: tuple):
    """
    Renders the complete Overview tab with KPIs and monthly trends.
    """
    st.header("Business Overview")
    
    # --- Data Preparation ---
    orders_df = filtered_data.get("orders")
    # Fetch data using the correct module
    all_time_first_purchases = data.get_all_time_first_purchase_dates(client)
    
    # --- KPI Display ---
    kpi_data = get_overview_kpis(orders_df, all_time_first_purchases, date_range)
    if kpi_data:
        # The metrics component expects a list of dicts
        metrics.create([{"label": k, "value": v} for k, v in kpi_data.items()], num_columns=len(kpi_data))
    else:
        st.warning("No data available to calculate KPIs for the selected period.")

    st.markdown("---")
    
    # --- Charting Section ---
    if orders_df is not None and not orders_df.empty:
        display_monthly_trends(orders_df, all_time_first_purchases)
    else:
        st.info("No order data to display charts for the selected period.") 