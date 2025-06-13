# tabs/product_insights.py
import streamlit as st
import pandas as pd
from typing import Dict
from itertools import combinations
import config 
import utils
from components import charts

def display_top_products_recurrent(df_orders: pd.DataFrame, all_orders: pd.DataFrame):
    """
    Analyzes and displays the top 10 most purchased products by recurrent customers.
    This now uses the full order history to correctly identify recurrent customers.
    """
    if df_orders.empty or 'customer_id' not in df_orders.columns:
        st.info("No data available to analyze recurrent customer products.")
        return
    
    # Correctly identify recurrent customers based on their entire purchase history
    recurrent_customers = utils.identify_recurrent_customers(all_orders)
    
    if recurrent_customers.empty:
        st.info("No recurrent customers found in the dataset.")
        return
    
    # Filter the data for the selected period to only include true recurrent customers
    df_recurrent_in_period = df_orders[df_orders['customer_id'].isin(recurrent_customers)]
    
    if df_recurrent_in_period.empty:
        st.info("No sales from recurrent customers in the selected period.")
        return
        
    top_products = df_recurrent_in_period.groupby('product_name')['lineitem_final_qty'].sum().nlargest(10).reset_index()
    
    fig = charts.create_bar_chart(top_products, x='product_name', y='lineitem_final_qty')
    charts.safe_plotly_chart(fig)

def display_cross_sell_analysis(df_orders: pd.DataFrame):
    """
    Analyzes and displays the most common pairs of products purchased together.
    """
    required_cols = ['order_id', 'product_name']
    if df_orders.empty or not all(col in df_orders.columns for col in required_cols):
        st.warning("Insufficient data for cross-sell analysis.")
        return
    
    order_item_counts = df_orders.groupby('order_id')['product_name'].nunique()
    multi_item_orders = order_item_counts[order_item_counts > 1].index
    
    if multi_item_orders.empty:
        st.info("No orders with multiple distinct products found to analyze.")
        return
    
    df_multi_item = df_orders[df_orders['order_id'].isin(multi_item_orders)]
    
    pairs = df_multi_item.groupby('order_id')['product_name'].apply(lambda x: list(combinations(sorted(set(x)), 2)))
    frequent_pairs = pairs.explode().value_counts().reset_index(name='frequency').head(10)
            
    if frequent_pairs.empty:
        st.info("No frequent product pairs were found.")
        return
        
    frequent_pairs['Product Pair'] = frequent_pairs['product_name'].apply(lambda x: f"{x[0]} & {x[1]}")
    fig = charts.create_bar_chart(frequent_pairs, x='Product Pair', y='frequency')
    charts.safe_plotly_chart(fig)

def display_purchase_seasonality(df_orders: pd.DataFrame):
    """
    Displays bar charts showing total revenue by month and by day of the week.
    """
    if df_orders.empty:
        st.info("No data available to analyze seasonality.")
        return
    
    df = df_orders.copy()
    
    # CRITICAL FIX: Ensure revenue is not inflated by summing per line item.
    # De-duplicate by order to get the correct order-level revenue.
    unique_orders_df = df.drop_duplicates(subset=['order_id'])
    
    unique_orders_df['order_create_date'] = pd.to_datetime(unique_orders_df['order_create_date'])
    unique_orders_df['month'] = unique_orders_df['order_create_date'].dt.month_name()
    unique_orders_df['day_of_week'] = unique_orders_df['order_create_date'].dt.day_name()
    
    col1, col2 = st.columns(2)
    with col1:
        sales_by_month = unique_orders_df.groupby('month')['order_final_total_amount'].sum().reindex(config.MONTHS_ORDER).reset_index()
        fig_month = charts.create_bar_chart(sales_by_month, x='month', y='order_final_total_amount', title="Revenue by Month")
        charts.safe_plotly_chart(fig_month)
    with col2:
        sales_by_day = unique_orders_df.groupby('day_of_week')['order_final_total_amount'].sum().reindex(config.DAYS_ORDER).reset_index()
        fig_day = charts.create_bar_chart(sales_by_day, x='day_of_week', y='order_final_total_amount', title="Revenue by Day of Week")
        charts.safe_plotly_chart(fig_day)

def display_category_level_heatmap(df_enriched: pd.DataFrame):
    """
    Displays a heatmap showing the correlation between product categories and player levels.
    """
    required_cols = ['product_category', 'player_level']
    if df_enriched.empty or not all(col in df_enriched.columns for col in required_cols):
        st.warning("Category or level data is missing for the heatmap.")
        return
        
    plot_data = df_enriched[df_enriched['player_level'] != 'Unspecified']
    fig = charts.create_correlation_heatmap(plot_data, x_col='player_level', y_col='product_category', title='Proportion of Categories Purchased by Player Level')
    charts.safe_plotly_chart(fig)

def display_category_migration(df_enriched: pd.DataFrame):
    """
    Displays a Sankey diagram showing how customers migrate between product categories.
    """
    if df_enriched.empty or 'customer_id' not in df_enriched.columns:
        st.info("No data available for category migration analysis.")
        return

    customer_counts = df_enriched['customer_id'].value_counts()
    repeat_customers = customer_counts[customer_counts >= 2].index
    if repeat_customers.empty:
        st.info("No customers with 2 or more purchases to analyze migration.")
        return

    df_repeats = df_enriched[df_enriched['customer_id'].isin(repeat_customers)].sort_values(['customer_id', 'order_create_date'])
    first_purchase = df_repeats.groupby('customer_id').first()
    second_purchase = df_repeats.groupby('customer_id').nth(1)

    migration_df = pd.merge(
        first_purchase['product_category'],
        second_purchase['product_category'],
        left_index=True, right_index=True,
        suffixes=('_first', '_second')
    )
    migration_counts = migration_df.groupby(['product_category_first', 'product_category_second']).size().reset_index(name='count')
    
    if migration_counts.empty:
        st.info("Could not determine migration patterns from first to second purchase.")
        return
        
    fig = charts.create_sankey_diagram(migration_counts, source='product_category_first', target='product_category_second', value='count')
    charts.safe_plotly_chart(fig)

def render(filtered_data: Dict[str, pd.DataFrame], all_data: Dict[str, pd.DataFrame]):
    """Renders the complete Product Insights tab."""
    st.header("Product & Purchase Behavior Insights")
    st.markdown("This tab explores what customers are buying and how different products relate to each other.")

    # --- Data Preparation ---
    orders_filtered = filtered_data.get("orders")
    all_orders = all_data.get("orders", pd.DataFrame()) # Get all orders for accurate recurrent calc

    if orders_filtered is None or orders_filtered.empty:
        st.warning("No sales data available for product analysis in the selected period.")
        return
        
    orders_enriched = utils.enrich_dataframe_with_categories(orders_filtered)

    # --- Charting Section ---
    st.subheader("Top Products for Recurrent Customers")
    display_top_products_recurrent(orders_enriched, all_orders)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cross-Sell & Product Bundles")
        display_cross_sell_analysis(orders_enriched)
    with col2:
        st.subheader("Category & Player Level Correlation")
        display_category_level_heatmap(orders_enriched)
    
    st.markdown("---")
    st.subheader("Purchase Seasonality")
    display_purchase_seasonality(orders_enriched)
    st.markdown("---")
    st.subheader("Customer Migration Between Product Categories")
    display_category_migration(orders_enriched) 