# tabs/ltv.py
import streamlit as st
import pandas as pd
from typing import Dict

import utils
from components import charts, metrics

# Frequency definitions, translated to English for consistency
FREQUENCY_BINS = [0, 1, 3, 5, float('inf')]
FREQUENCY_LABELS = ['1 Order', '2-3 Orders', '4-5 Orders', '6+ Orders']

def calculate_ltv_data(df_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Lifetime Value (LTV) for each customer based on their complete order history.
    """
    if df_orders.empty or 'customer_id' not in df_orders.columns:
        return pd.DataFrame()
    
    # LTV must be calculated on all orders, not just filtered ones.
    # The dataframe passed here should be all_data['orders'].
    unique_orders_df = df_orders.drop_duplicates(subset=['order_id'])
    ltv = unique_orders_df.groupby('customer_id')['order_final_total_amount'].sum().reset_index()
    return ltv.rename(columns={'order_final_total_amount': 'ltv'})

def display_ltv_by_frequency(ltv_data: pd.DataFrame, all_orders: pd.DataFrame):
    """Displays the average LTV for different customer frequency segments."""
    if ltv_data.empty or all_orders.empty:
        return
        
    order_counts = all_orders.groupby('customer_id')['order_id'].nunique().reset_index(name='order_count')
    ltv_with_freq = pd.merge(ltv_data, order_counts, on='customer_id')
    ltv_with_freq['freq_segment'] = pd.cut(
        ltv_with_freq['order_count'], 
        bins=FREQUENCY_BINS, 
        labels=FREQUENCY_LABELS, 
        right=False
    )
    
    avg_ltv_by_freq = ltv_with_freq.groupby('freq_segment')['ltv'].mean().reset_index()
    fig = charts.create_bar_chart(avg_ltv_by_freq, x='freq_segment', y='ltv', title='Average LTV per Frequency Segment')
    charts.safe_plotly_chart(fig)

def display_ltv_by_player_level(ltv_data: pd.DataFrame, all_orders_enriched: pd.DataFrame):
    """Displays the average LTV based on the player level of the customer's first purchase."""
    if ltv_data.empty or all_orders_enriched.empty:
        return
        
    first_purchase_level = all_orders_enriched.loc[all_orders_enriched.groupby('customer_id')['order_create_date'].idxmin()]
    ltv_with_level = pd.merge(ltv_data, first_purchase_level[['customer_id', 'player_level']], on='customer_id')
    
    # Filter out 'Unspecified' levels for cleaner analysis
    ltv_with_level = ltv_with_level[ltv_with_level['player_level'] != 'Unspecified']
    
    if ltv_with_level.empty:
        st.info("No data available for LTV by player level analysis.")
        return
    
    avg_ltv_by_level = ltv_with_level.groupby('player_level')['ltv'].mean().reset_index()
    fig = charts.create_bar_chart(avg_ltv_by_level, x='player_level', y='ltv', title='Average LTV per Player Level (First Purchase)')
    charts.safe_plotly_chart(fig)

def display_ltv_by_first_category(ltv_data: pd.DataFrame, all_orders_enriched: pd.DataFrame):
    """Displays the average LTV based on the category of the customer's first purchase."""
    if ltv_data.empty or all_orders_enriched.empty:
        return
        
    first_purchase_category = all_orders_enriched.loc[all_orders_enriched.groupby('customer_id')['order_create_date'].idxmin()]
    ltv_with_category = pd.merge(ltv_data, first_purchase_category[['customer_id', 'product_category']], on='customer_id')
    
    avg_ltv_by_category = ltv_with_category.groupby('product_category')['ltv'].mean().reset_index()
    avg_ltv_by_category = avg_ltv_by_category.sort_values('ltv', ascending=False)
    
    fig = charts.create_bar_chart(avg_ltv_by_category, x='product_category', y='ltv', title='Average LTV per First Purchase Category')
    charts.safe_plotly_chart(fig)

def display_detailed_ltv_metrics_by_sport(ltv_data: pd.DataFrame, all_orders_enriched: pd.DataFrame):
    """
    Displays detailed LTV metrics segmented by sport universe with additional metrics.
    """
    st.subheader("Detailed LTV Metrics by Sport Universe")
    
    if ltv_data.empty or all_orders_enriched.empty:
        st.info("No data available for detailed LTV analysis by sport.")
        return
        
    first_purchase_sport = all_orders_enriched.loc[all_orders_enriched.groupby('customer_id')['order_create_date'].idxmin()]
    ltv_with_sport = pd.merge(ltv_data, first_purchase_sport[['customer_id', 'sport_universe']], on='customer_id')
    
    # Calculate additional metrics
    customer_order_counts = all_orders_enriched.groupby('customer_id').agg({
        'order_id': 'nunique',  # Total orders per customer
        'order_final_total_amount': 'sum'  # Total spent (LTV) - validation
    }).reset_index()
    customer_order_counts.columns = ['customer_id', 'total_orders', 'total_spent_validation']
    
    # Merge with sport data
    detailed_analysis = pd.merge(
        pd.merge(ltv_with_sport, customer_order_counts, on='customer_id'),
        first_purchase_sport[['customer_id', 'order_final_total_amount']].rename(columns={'order_final_total_amount': 'first_purchase_amount'}),
        on='customer_id'
    )
    
    # Calculate metrics by sport
    sport_metrics = detailed_analysis.groupby('sport_universe').agg({
        'ltv': ['mean', 'median', 'count'],
        'total_orders': 'mean',
        'first_purchase_amount': 'mean'
    }).round(2)
    
    # Flatten column names
    sport_metrics.columns = ['Avg LTV', 'Median LTV', 'Customer Count', 'Avg Orders per Customer', 'Avg First Purchase']
    
    # Display table
    st.dataframe(
        sport_metrics,
        column_config={
            "Avg LTV": st.column_config.NumberColumn("Avg LTV", format="$%.2f"),
            "Median LTV": st.column_config.NumberColumn("Median LTV", format="$%.2f"),
            "Customer Count": st.column_config.NumberColumn("Customers", format="%d"),
            "Avg Orders per Customer": st.column_config.NumberColumn("Avg Orders", format="%.1f"),
            "Avg First Purchase": st.column_config.NumberColumn("Avg First Purchase", format="$%.2f")
        },
        use_container_width=True
    )

def display_ltv_by_sport(ltv_data: pd.DataFrame, all_orders_enriched: pd.DataFrame):
    """Displays the average LTV based on the sport of the customer's first purchase."""
    if ltv_data.empty or all_orders_enriched.empty:
        return
        
    first_purchase_sport = all_orders_enriched.loc[all_orders_enriched.groupby('customer_id')['order_create_date'].idxmin()]
    ltv_with_sport = pd.merge(ltv_data, first_purchase_sport[['customer_id', 'sport_universe']], on='customer_id')
    
    avg_ltv_by_sport = ltv_with_sport.groupby('sport_universe')['ltv'].mean().reset_index()
    fig = charts.create_bar_chart(avg_ltv_by_sport, x='sport_universe', y='ltv', title='Average LTV per Sport of First Purchase')
    charts.safe_plotly_chart(fig)

def display_ltv_by_channel(ltv_data: pd.DataFrame, all_orders_enriched: pd.DataFrame):
    """Displays the average LTV based on the customer's first acquisition channel."""
    if ltv_data.empty or all_orders_enriched.empty:
        st.info("LTV or full order data not available for channel analysis.")
        return

    first_orders = all_orders_enriched.loc[all_orders_enriched.groupby('customer_id')['order_create_date'].idxmin()]
    acquisition_channels = first_orders[['customer_id', 'order_first_visit_source_type']].rename(columns={'order_first_visit_source_type': 'channel'})
    
    ltv_by_channel = pd.merge(ltv_data, acquisition_channels, on='customer_id')
    avg_ltv = ltv_by_channel.groupby('channel')['ltv'].mean().reset_index().sort_values('ltv', ascending=False)
    
    fig = charts.create_bar_chart(avg_ltv, x='channel', y='ltv', title='Average LTV by Acquisition Channel')
    charts.safe_plotly_chart(fig)

def display_ltv_cac_ratio(ltv_data: pd.DataFrame, all_orders: pd.DataFrame, ga_metrics: pd.DataFrame):
    """Calculates and displays the LTV to CAC ratio for each acquisition channel."""
    if ltv_data.empty or all_orders.empty or ga_metrics.empty:
        st.info("LTV, Order, or Ad data is not available for LTV/CAC ratio analysis.")
        return

    first_orders = all_orders.loc[all_orders.groupby('customer_id')['order_create_date'].idxmin()]
    acquisition_channels = first_orders[['customer_id', 'order_first_visit_source_type']].rename(columns={'order_first_visit_source_type': 'channel'})
    ltv_by_channel = pd.merge(ltv_data, acquisition_channels, on='customer_id').groupby('channel')['ltv'].mean().reset_index()

    cac_by_channel = ga_metrics.groupby('primary_channel_group').agg(total_cost=('ads_cost', 'sum'), total_users=('total_users', 'sum')).reset_index()
    cac_by_channel['cac'] = utils.safe_division(cac_by_channel['total_cost'], cac_by_channel['total_users'])
    cac_by_channel = cac_by_channel.rename(columns={'primary_channel_group': 'channel'})
    
    ratio_df = pd.merge(ltv_by_channel, cac_by_channel[['channel', 'cac']], on='channel', how='left')
    ratio_df = ratio_df[ratio_df['cac'] > 0]
    ratio_df['ltv_cac_ratio'] = utils.safe_division(ratio_df['ltv'], ratio_df['cac'])
    ratio_df = ratio_df.sort_values('ltv_cac_ratio', ascending=False)
    
    fig = charts.create_bar_chart(ratio_df, x='channel', y='ltv_cac_ratio', title='LTV to CAC Ratio by Channel')
    
    # Only add lines if the figure was created successfully
    if fig is not None:
        fig.add_hline(y=3, line_dash="dash", line_color="blue", annotation_text="Ideal Ratio (3x)")
        fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Break-even (1x)")
    charts.safe_plotly_chart(fig)

def render(filtered_data: Dict[str, pd.DataFrame], all_data: Dict[str, pd.DataFrame]):
    """Renders the complete LTV analysis tab."""
    st.header("Customer Lifetime Value (LTV)")
    st.markdown("This tab explores the total value a customer brings over their entire relationship with the business.")

    # --- Data Preparation ---
    orders_filtered = filtered_data.get("orders")
    if orders_filtered is None or orders_filtered.empty:
        st.warning("Order data for the selected period is required for LTV analysis.")
        return
    
    all_orders = all_data.get("orders", pd.DataFrame())
    if all_orders.empty:
        st.warning("Complete order history is not available. LTV calculations will be inaccurate.")
        return

    ga_metrics_data = all_data.get("ga_metrics", pd.DataFrame())
    all_orders_enriched = utils.enrich_dataframe_with_categories(all_orders)

    # --- LTV Calculation & KPI ---
    # LTV must be calculated on the entire dataset for accuracy.
    ltv_data = calculate_ltv_data(all_orders)
    if not ltv_data.empty:
        # We only want to see the LTV for customers who were active in the selected period.
        customers_in_period = orders_filtered['customer_id'].unique()
        ltv_for_active_customers = ltv_data[ltv_data['customer_id'].isin(customers_in_period)]
        
        avg_ltv = ltv_for_active_customers['ltv'].mean()
        metrics.create([{"label": "Avg. LTV of Customers Active in Period", "value": utils.format_currency(avg_ltv)}], num_columns=1)
    else:
        st.info("No LTV data could be calculated.")

    st.markdown("---")
    
    # --- Main LTV Analysis Section ---
    st.subheader("🎯 Core LTV Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("LTV by Frequency Segment")
        display_ltv_by_frequency(ltv_data, all_orders)
        
        st.subheader("LTV by Acquisition Channel")
        display_ltv_by_channel(ltv_data, all_orders_enriched)
        
    with col2:
        st.subheader("LTV by First Purchase Sport")
        display_ltv_by_sport(ltv_data, all_orders_enriched)
        
        st.subheader("LTV to CAC Ratio by Channel")
        display_ltv_cac_ratio(ltv_data, all_orders_enriched, ga_metrics_data)
    
    st.markdown("---")
    
    # --- Advanced Segmentation Section ---
    st.subheader("🔍 Advanced LTV Segmentation")
    
    # Detailed metrics by sport first
    display_detailed_ltv_metrics_by_sport(ltv_data, all_orders_enriched)
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("LTV by Player Level")
        display_ltv_by_player_level(ltv_data, all_orders_enriched)
        
    with col4:
        st.subheader("LTV by First Purchase Category")
        display_ltv_by_first_category(ltv_data, all_orders_enriched) 