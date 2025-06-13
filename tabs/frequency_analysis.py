import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Tuple, Optional

from components import metrics, charts
import utils

# Segmentation constants
FREQUENCY_BINS = [0, 1, 3, 5, float('inf')]
FREQUENCY_LABELS = [
    '1 Order (Single Buyer)', 
    '2-3 Orders (Occasional)', 
    '4-5 Orders (Frequent)', 
    '6+ Orders (Loyal)'
]

def analyze_frequency_metrics(df_orders: pd.DataFrame) -> Dict:
    """
    Calculates key frequency and recurrence metrics from order data.

    Args:
        df_orders (pd.DataFrame): DataFrame containing order information.

    Returns:
        Dict: A dictionary of summary metrics (e.g., avg orders, recurrence %).
    """
    if df_orders.empty or 'customer_id' not in df_orders.columns:
        return {}

    customer_orders_freq = df_orders.groupby('customer_id')['order_id'].nunique().reset_index(name='num_orders')
    
    total_customers = customer_orders_freq['customer_id'].nunique()
    
    # Use the standardized function to identify recurrent customers
    recurrent_customer_ids = utils.identify_recurrent_customers(df_orders)
    recurrent_customers_count = len(recurrent_customer_ids)
    
    avg_time_between_purchases = 0
    if recurrent_customers_count > 0:
        df_recurrent_orders = df_orders[df_orders['customer_id'].isin(recurrent_customer_ids)].copy()
        df_recurrent_orders.sort_values(['customer_id', 'order_create_date'], inplace=True)
        time_diffs = df_recurrent_orders.groupby('customer_id')['order_create_date'].diff().dt.days
        avg_time_between_purchases = time_diffs.mean()

    aov = utils.calculate_aov(df_orders)

    return {
        'Avg. Orders per Customer': f"{customer_orders_freq['num_orders'].mean():.2f}",
        '% Recurrent Customers': utils.format_percentage(utils.safe_division(recurrent_customers_count, total_customers) * 100),
        'Avg. Time Between Purchases': f"{avg_time_between_purchases:.1f} days" if avg_time_between_purchases else "N/A",
        'Avg. Order Value (AOV)': utils.format_currency(aov)
    }

def display_frequency_distributions(df_orders: pd.DataFrame):
    """
    Displays charts for customer frequency distribution and time between purchases
    based on the data within the selected date range.
    """
    if df_orders.empty:
        st.info("No frequency data to display for the selected period.")
        return
        
    customer_orders_freq = df_orders.groupby('customer_id')['order_id'].nunique().reset_index(name='num_orders')
    
    col1, col2 = st.columns(2)
    with col1:
        customer_orders_freq['order_group'] = pd.cut(
            customer_orders_freq['num_orders'], bins=FREQUENCY_BINS, labels=FREQUENCY_LABELS, right=False
        )
        segment_dist = customer_orders_freq.groupby('order_group')['customer_id'].nunique().reset_index()
        fig1 = charts.create_bar_chart(segment_dist, x='order_group', y='customer_id', title="Customer Distribution by Order Frequency")
        charts.safe_plotly_chart(fig1)

    with col2:
        recurrent_ids = customer_orders_freq[customer_orders_freq['num_orders'] > 1]['customer_id']
        if not recurrent_ids.empty:
            df_recurrent = df_orders[df_orders['customer_id'].isin(recurrent_ids)].copy()
            df_recurrent.sort_values(['customer_id', 'order_create_date'], inplace=True)
            df_recurrent['time_diff_days'] = df_recurrent.groupby('customer_id')['order_create_date'].diff().dt.days
            
            fig2 = px.histogram(df_recurrent.dropna(subset=['time_diff_days']), x='time_diff_days', nbins=20, title="Distribution of Time Between Purchases")
            charts.safe_plotly_chart(fig2)
        else:
            st.info("No customers with multiple orders to analyze time between purchases.")

def display_cohort_retention(all_orders: pd.DataFrame):
    """
    Calculates and displays a cohort retention heatmap.
    """
    st.info("This heatmap shows the percentage of customers from a cohort (grouped by their first purchase month) who made a repeat purchase in subsequent months.", icon="ℹ️")

    if all_orders.empty or 'customer_id' not in all_orders.columns:
        st.warning("Full order history with customer IDs is required for cohort analysis.")
        return

    df = all_orders.copy()
    df['order_month'] = df['order_create_date'].dt.to_period('M')
    df['cohort_month'] = df.groupby('customer_id')['order_create_date'].transform('min').dt.to_period('M')
    
    def get_cohort_index(df_cohort: pd.DataFrame) -> pd.Series:
        year_diff = df_cohort['order_month'].dt.year - df_cohort['cohort_month'].dt.year
        month_diff = df_cohort['order_month'].dt.month - df_cohort['cohort_month'].dt.month
        return year_diff * 12 + month_diff

    df['cohort_index'] = get_cohort_index(df)
    
    cohort_data = df.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()
    cohort_counts = cohort_data.pivot_table(index='cohort_month', columns='cohort_index', values='customer_id')
    
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention_matrix = cohort_counts.divide(cohort_sizes, axis=0)

    # Convert PeriodIndex to strings for JSON serialization
    retention_matrix.index = retention_matrix.index.strftime('%Y-%m')
    
    fig = px.imshow(
        retention_matrix,
        title="Monthly Customer Retention by Cohort",
        labels=dict(x="Months Since First Purchase", y="Customer Cohort", color="Retention %"),
        text_auto=".0%",
        color_continuous_scale=px.colors.sequential.Blues
    )
    charts.safe_plotly_chart(fig)

def display_recurrence_by_first_purchase(df_orders: pd.DataFrame):
    """
    Displays heatmaps analyzing recurrence patterns based on the sport and level of a customer's first purchase.
    """
    if df_orders.empty:
        st.warning("Not enough data for recurrence analysis.")
        return

    first_purchases = df_orders.loc[df_orders.groupby('customer_id')['order_create_date'].idxmin()].copy()
    customer_order_counts = df_orders.groupby('customer_id')['order_id'].nunique().reset_index(name='total_orders')
    analysis_df = pd.merge(first_purchases, customer_order_counts, on='customer_id')
    analysis_df['is_recurrent'] = analysis_df['total_orders'] > 1
    analysis_df = analysis_df[analysis_df['player_level'] != 'Unspecified']

    if analysis_df.empty:
        st.info("No data available for this analysis after filtering.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("% of Customers Who Become Recurrent")
        recurrent_percentage = analysis_df.groupby(['sport_universe', 'player_level'])['is_recurrent'].mean().unstack() * 100
        fig1 = charts.create_correlation_heatmap(recurrent_percentage, title='')
        charts.safe_plotly_chart(fig1)

    with col2:
        st.subheader("Avg. Orders from Recurrent Customers")
        avg_orders_recurrent = analysis_df[analysis_df['is_recurrent']].groupby(['sport_universe', 'player_level'])['total_orders'].mean().unstack()
        fig2 = px.imshow(
            avg_orders_recurrent.fillna(0),
            text_auto=".1f",
            labels=dict(x="Player Level", y="First Purchase Sport", color="Avg. Orders"),
            color_continuous_scale=px.colors.sequential.Greens
        )
        charts.safe_plotly_chart(fig2)

def render(all_data: Dict[str, pd.DataFrame], filtered_data: Dict[str, pd.DataFrame]):
    """
    Renders the complete Frequency & Recurrence analysis tab.
    """
    st.header("Customer Frequency & Recurrence")
    st.markdown("This tab explores how often customers return and what defines their loyalty.")

    # --- Data Preparation ---
    orders_filtered = filtered_data.get("orders")
    if orders_filtered is None or orders_filtered.empty:
        st.warning("No sales data available for frequency analysis in the selected period.")
        return
        
    all_orders = all_data.get("orders", pd.DataFrame())
    
    # Enrich data with categories (only once)
    orders_filtered_enriched = utils.enrich_dataframe_with_categories(orders_filtered)
    all_orders_enriched = utils.enrich_dataframe_with_categories(all_orders) if not all_orders.empty else pd.DataFrame()

    # --- KPI Display ---
    # Pass the complete, unfiltered dataset to the metrics function for an accurate recurrent %
    kpi_summary = analyze_frequency_metrics(all_orders_enriched)
    if kpi_summary:
        metrics.create([{"label": k, "value": v} for k, v in kpi_summary.items()], num_columns=len(kpi_summary))
    
    st.markdown("---")
    
    # --- Charting Section ---
    st.subheader("Customer Frequency Analysis (for Selected Period)")
    display_frequency_distributions(orders_filtered_enriched)
    st.markdown("---")

    st.subheader("Monthly Customer Retention by Cohort")
    display_cohort_retention(all_orders_enriched)
    st.markdown("---")

    st.subheader("Recurrence by First Purchase: Sport & Player Level")
    display_recurrence_by_first_purchase(orders_filtered_enriched) 