# tabs/channels.py
import streamlit as st
import pandas as pd
from typing import Dict, Tuple
from components import metrics, charts
import utils
import plotly.graph_objects as go

def assign_channel_group(campaign_name: str) -> str:
    """Categorizes a campaign into a channel group based on keywords in its name."""
    if pd.isna(campaign_name):
        return 'Unassigned'
    
    name = campaign_name.lower()
    
    # Simple rule-based assignment. This can be expanded.
    if 'pmax' in name or 'performance max' in name:
        return 'Performance Max'
    if 'search' in name and 'brand' in name:
        return 'Paid Search - Brand'
    if 'search' in name and 'non-brand' in name:
        return 'Paid Search - Non-Brand'
    if 'search' in name:
        return 'Paid Search - Generic'
    if 'shopping' in name:
        return 'Shopping'
    if 'social' in name or 'facebook' in name or 'instagram'in name:
        return 'Paid Social'
    if 'display' in name:
        return 'Display'
    if 'youtube' in name or 'video' in name:
        return 'Video'
    
    return 'Other'

def calculate_channel_kpis(ga_df: pd.DataFrame) -> Dict:
    """Calculates high-level KPIs from the Google Analytics campaign data."""
    if ga_df.empty:
        return {}

    total_cost = ga_df['ads_cost'].sum()
    total_revenue = ga_df['campaign_total_revenue'].sum()
    total_clicks = ga_df['ads_clicks'].sum()
    total_impressions = ga_df['ads_impressions'].sum()
    total_conversions = ga_df['campaign_purchase_events'].sum()

    kpis = {
        'Total Ad Spend': utils.format_currency(total_cost),
        'Total Ad Revenue': utils.format_currency(total_revenue),
        'Overall ROAS': f"{utils.safe_division(total_revenue, total_cost):.2f}x",
        'Total Conversions': f"{total_conversions:,.0f}",
        'Cost Per Click (CPC)': utils.format_currency(utils.safe_division(total_cost, total_clicks)),
        'Cost Per Acq. (CPA)': utils.format_currency(utils.safe_division(total_cost, total_conversions)),
        'Conversion Rate (from Clicks)': utils.format_percentage(100 * utils.safe_division(total_conversions, total_clicks)),
        'Click-Through Rate (CTR)': utils.format_percentage(100 * utils.safe_division(total_clicks, total_impressions))
    }
    return kpis

def display_performance_trends(ga_df: pd.DataFrame):
    """Shows a time-series chart of key performance metrics."""
    st.subheader("Daily Performance Trends")
    if ga_df.empty:
        st.info("No data to display daily trends.")
        return

    daily_perf = ga_df.groupby('campaign_date').agg(
        ads_cost=('ads_cost', 'sum'),
        campaign_total_revenue=('campaign_total_revenue', 'sum')
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_perf['campaign_date'], y=daily_perf['campaign_total_revenue'],
        mode='lines', name='Total Ad Revenue', yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=daily_perf['campaign_date'], y=daily_perf['ads_cost'],
        mode='lines', name='Total Ad Spend', yaxis='y2'
    ))

    fig.update_layout(
        title_text="Ad Revenue vs. Spend Over Time",
        yaxis=dict(title="Revenue ($)", side='left'),
        yaxis2=dict(title="Spend ($)", overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation='h')
    )
    charts.safe_plotly_chart(fig)

def display_performance_table(df: pd.DataFrame, group_by_col: str, display_title: str):
    """
    Displays a detailed performance table grouped by a specific column, with visual enhancements.
    The title for the grouped column is now passed as an argument for reusability.
    """
    st.subheader(f"Performance by {display_title}")
    
    if df.empty:
        st.info("No data for performance breakdown.")
        return

    # Aggregate data to get sums for primary metrics
    perf_table = df.groupby(group_by_col).agg(
        Spend=('ads_cost', 'sum'),
        Revenue=('campaign_total_revenue', 'sum'),
        Clicks=('ads_clicks', 'sum'),
        Impressions=('ads_impressions', 'sum'),
        Conversions=('campaign_purchase_events', 'sum')
    ).reset_index()

    # Calculate derived metrics
    perf_table['ROAS'] = utils.safe_division(perf_table['Revenue'], perf_table['Spend'])
    perf_table['CTR'] = 100 * utils.safe_division(perf_table['Clicks'], perf_table['Impressions'])
    perf_table['CPC'] = utils.safe_division(perf_table['Spend'], perf_table['Clicks'])
    perf_table['CPA'] = utils.safe_division(perf_table['Spend'], perf_table['Conversions'])
    perf_table['AOV'] = utils.safe_division(perf_table['Revenue'], perf_table['Conversions'])

    # Define sensible maximums for progress bars to provide better context than just using the max value in the dataset.
    # This prevents one outlier from making all other bars look tiny.
    roas_max = max(5, perf_table['ROAS'].quantile(0.95) * 1.2) # Set a minimum of 5x ROAS for the scale
    ctr_max = max(5, perf_table['CTR'].quantile(0.95) * 1.2) # Set a minimum of 5% CTR for the scale

    # Reorder columns for logical flow
    perf_table = perf_table[[
        group_by_col, 'Spend', 'Revenue', 'ROAS', 'Conversions', 'CPA', 'AOV',
        'Clicks', 'Impressions', 'CTR', 'CPC'
    ]]

    # Use st.column_config to create rich, visual tables with data bars
    st.dataframe(
        perf_table,
        column_config={
            group_by_col: st.column_config.TextColumn(
                display_title, # Use the dynamic title here
            ),
            "Spend": st.column_config.NumberColumn(
                "Spend ($)", format="$%.2f",
            ),
            "Revenue": st.column_config.NumberColumn(
                "Revenue ($)", format="$%.2f",
            ),
            "ROAS": st.column_config.ProgressColumn(
                "ROAS",
                help="Return on Ad Spend (Revenue / Spend)",
                format="%.2fx",
                min_value=0, max_value=roas_max,
            ),
            "Conversions": st.column_config.NumberColumn(
                "Conversions", format="%d"
            ),
            "CPA": st.column_config.NumberColumn(
                "CPA ($)",
                help="Cost Per Acquisition",
                format="$%.2f",
            ),
            "AOV": st.column_config.NumberColumn(
                "AOV ($)",
                help="Average Order Value",
                format="$%.2f",
            ),
            "CTR": st.column_config.ProgressColumn(
                "CTR (%)",
                help="Click-Through Rate",
                format="%.2f%%",
                min_value=0, max_value=ctr_max,
            ),
            "CPC": st.column_config.NumberColumn(
                "CPC ($)",
                help="Cost Per Click",
                format="$%.2f",
            ),
        },
        use_container_width=True,
        hide_index=True
    )

def display_channel_group_explorer(df: pd.DataFrame):
    """
    Creates an interactive explorer to view campaigns within a selected channel group.
    This is especially useful for understanding which campaigns are in the 'Other' category.
    """
    with st.expander("🕵️‍♀️ Explore Channel Group Details"):
        st.markdown("""
        Select a channel group from the dropdown below to see a detailed performance breakdown 
        of all the campaigns that have been assigned to it.
        """)
        
        # Ensure the 'primary_channel_group' column exists
        if 'primary_channel_group' not in df.columns:
            st.warning("Channel group data not available.")
            return

        # Get the list of unique channel groups to populate the selectbox
        channel_groups = sorted(df['primary_channel_group'].unique())
        # Ensure 'Other' is an option if it exists, and make it the default
        default_index = 0
        if 'Other' in channel_groups:
            default_index = list(channel_groups).index('Other')

        selected_group = st.selectbox(
            "Select a channel group to see its campaigns:",
            channel_groups,
            index=default_index
        )

        if selected_group:
            # Filter the dataframe to only include campaigns from the selected group
            filtered_df = df[df['primary_channel_group'] == selected_group].copy()
            
            st.write(f"Displaying campaigns for the **{selected_group}** group:")
            # Reuse the performance table function to display the filtered campaigns
            display_performance_table(filtered_df, 'campaign_name', "Campaign Name")

def render(filtered_data: Dict[str, pd.DataFrame]):
    """Renders the complete Channels analysis tab."""
    st.header("Paid Channel Performance (Google Ads)")
    st.markdown("""
    This tab analyzes the performance of your Google Ads campaigns. 
    All metrics are based on data directly from the Google Ads platform.
    """)

    ga_df = filtered_data.get("ga_metrics")

    if ga_df is None or ga_df.empty:
        st.warning("No Google Ads data is available for the selected date range. Cannot generate analysis.")
        return

    # --- Data Preparation ---
    ga_df['primary_channel_group'] = ga_df['campaign_name'].apply(assign_channel_group)

    # --- KPI Display ---
    kpi_data = calculate_channel_kpis(ga_df)
    if kpi_data:
        metrics.create([{"label": k, "value": v} for k, v in kpi_data.items()], num_columns=4)
    
    st.markdown("---")
    
    # --- Charting & Tables ---
    display_performance_trends(ga_df)
    st.markdown("---")

    # Use tabs for a cleaner layout instead of columns
    group_tab, campaign_tab = st.tabs(["Performance by Channel Group", "Performance by Campaign"])
    with group_tab:
        display_performance_table(ga_df, 'primary_channel_group', "Channel Group")
    with campaign_tab:
        display_performance_table(ga_df, 'campaign_name', "Campaign Name")
        
    st.markdown("---")
    display_channel_group_explorer(ga_df) 