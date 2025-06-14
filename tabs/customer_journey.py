import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional
import numpy as np

from components import charts, metrics
import utils

def analyze_first_vs_repeat_purchases(all_orders: pd.DataFrame) -> Dict:
    """
    Analiza las diferencias entre primera compra y recompras.
    """
    if all_orders.empty or 'customer_id' not in all_orders.columns:
        return {}
    
    # Identificar primera y segunda compra de cada cliente - asegurar tipo consistente
    customer_orders = all_orders.copy()
    customer_orders['customer_id'] = customer_orders['customer_id'].astype(str)
    customer_orders = customer_orders.sort_values(['customer_id', 'order_create_date'])
    
    first_purchases = customer_orders.groupby('customer_id').first()
    
    # Clientes con al menos 2 compras
    customer_order_counts = customer_orders.groupby('customer_id').size()
    repeat_customers = customer_order_counts[customer_order_counts >= 2].index
    
    if len(repeat_customers) == 0:
        return {'Customers with Repeat Purchases': '0'}
    
    # Segunda compra de clientes recurrentes
    second_purchases = customer_orders[customer_orders['customer_id'].isin(repeat_customers)].groupby('customer_id').nth(1)
    
    # Análisis de AOV - usar loc con customer_id como string
    first_aov = first_purchases.loc[repeat_customers, 'order_final_total_amount'].mean()
    repeat_aov = second_purchases['order_final_total_amount'].mean()
    
    # Tiempo entre compras
    time_between = (second_purchases['order_create_date'] - first_purchases.loc[repeat_customers, 'order_create_date']).dt.days.mean()
    
    return {
        'Customers with Repeat Purchases': f"{len(repeat_customers):,}",
        'First Purchase AOV': utils.format_currency(first_aov),
        'Second Purchase AOV': utils.format_currency(repeat_aov),
        'Avg Days Between Purchases': f"{time_between:.1f} days"
    }

def display_category_transition_matrix(all_orders: pd.DataFrame):
    """
    Muestra una matriz de transición entre categorías de productos.
    """
    st.subheader("Category Transition: First Purchase → Second Purchase")
    
    if all_orders.empty:
        st.info("No data available for category transition analysis.")
        return
    
    # Preparar datos - asegurar que customer_id sea string para consistencia
    customer_orders = all_orders.copy()
    customer_orders['customer_id'] = customer_orders['customer_id'].astype(str)
    customer_orders = customer_orders.sort_values(['customer_id', 'order_create_date'])
    
    customer_order_counts = customer_orders.groupby('customer_id').size()
    repeat_customers = customer_order_counts[customer_order_counts >= 2].index
    
    if len(repeat_customers) == 0:
        st.info("No customers with multiple purchases found.")
        return
    
    # Primera y segunda compra - usar reset_index para evitar problemas de alineación
    first_purchases = customer_orders.groupby('customer_id').first().reset_index()
    second_purchases = customer_orders[customer_orders['customer_id'].isin(repeat_customers)].groupby('customer_id').nth(1).reset_index()
    
    # Merge en lugar de usar loc para evitar problemas de índice
    first_purchase_categories = first_purchases[first_purchases['customer_id'].isin(repeat_customers)][['customer_id', 'product_category']]
    first_purchase_categories.columns = ['customer_id', 'first_category']
    
    second_purchase_categories = second_purchases[['customer_id', 'product_category']]
    second_purchase_categories.columns = ['customer_id', 'second_category']
    
    # Crear datos de transición usando merge
    transition_data = pd.merge(first_purchase_categories, second_purchase_categories, on='customer_id')
    
    if transition_data.empty:
        st.info("No transition data could be created.")
        return
    
    # Crear matriz de contingencia
    transition_matrix = pd.crosstab(transition_data['first_category'], transition_data['second_category'], normalize='index') * 100
    
    # Crear heatmap
    fig = px.imshow(
        transition_matrix,
        title="Category Transition Matrix (%)",
        labels=dict(x="Second Purchase Category", y="First Purchase Category", color="Transition %"),
        text_auto=".1f",
        color_continuous_scale="Blues"
    )
    charts.safe_plotly_chart(fig)

def display_sport_universe_behavior(all_orders: pd.DataFrame):
    """
    Analiza el comportamiento de compra por universo deportivo (Pádel vs Pickleball).
    """
    st.subheader("Behavior by Sport Universe")
    
    if all_orders.empty:
        st.info("No data available for sport universe analysis.")
        return
    
    # Asegurar tipo consistente para customer_id
    all_orders_clean = all_orders.copy()
    all_orders_clean['customer_id'] = all_orders_clean['customer_id'].astype(str)
    
    # Identificar primera compra por deporte
    first_purchases = all_orders_clean.loc[all_orders_clean.groupby('customer_id')['order_create_date'].idxmin()]
    
    # Análisis por deporte
    sport_analysis = first_purchases.groupby('sport_universe').agg({
        'customer_id': 'count',
        'order_final_total_amount': 'mean',
        'player_level': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
    }).round(2)
    
    sport_analysis.columns = ['Customers', 'Avg First Purchase AOV', 'Most Common Level']
    
    # Mostrar tabla
    st.dataframe(
        sport_analysis,
        column_config={
            "Customers": st.column_config.NumberColumn("Customers", format="%d"),
            "Avg First Purchase AOV": st.column_config.NumberColumn("Avg First Purchase AOV", format="$%.2f"),
            "Most Common Level": st.column_config.TextColumn("Most Common Player Level")
        },
        use_container_width=True
    )
    
    # Análisis de retención por deporte
    customer_order_counts = all_orders_clean.groupby('customer_id').size()
    repeat_customers = customer_order_counts[customer_order_counts >= 2].index
    
    retention_by_sport = first_purchases.groupby('sport_universe').apply(
        lambda x: (x['customer_id'].isin(repeat_customers).sum() / len(x)) * 100
    ).reset_index(name='retention_rate')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = charts.create_bar_chart(
            sport_analysis.reset_index(), 
            x='sport_universe', 
            y='Avg First Purchase AOV',
            title="Average First Purchase AOV by Sport"
        )
        charts.safe_plotly_chart(fig1)
    
    with col2:
        fig2 = charts.create_bar_chart(
            retention_by_sport, 
            x='sport_universe', 
            y='retention_rate',
            title="Customer Retention Rate by Sport (%)"
        )
        charts.safe_plotly_chart(fig2)

def display_purchase_sequence_analysis(all_orders: pd.DataFrame):
    """
    Analiza qué compran los clientes en sus primeras 3 compras.
    """
    st.subheader("Purchase Sequence Analysis: What Customers Buy Over Time")
    
    if all_orders.empty:
        st.info("No data available for purchase sequence analysis.")
        return
    
    # Preparar datos de secuencia - asegurar tipo consistente
    customer_orders = all_orders.copy()
    customer_orders['customer_id'] = customer_orders['customer_id'].astype(str)
    customer_orders = customer_orders.sort_values(['customer_id', 'order_create_date'])
    customer_orders['purchase_number'] = customer_orders.groupby('customer_id').cumcount() + 1
    
    # Filtrar primeras 3 compras
    first_three_purchases = customer_orders[customer_orders['purchase_number'] <= 3]
    
    # Análisis por número de compra
    sequence_analysis = first_three_purchases.groupby(['purchase_number', 'product_category']).size().reset_index(name='count')
    sequence_pivot = sequence_analysis.pivot(index='product_category', columns='purchase_number', values='count').fillna(0)
    
    # Normalizar para mostrar porcentajes
    sequence_percent = sequence_pivot.div(sequence_pivot.sum(axis=0), axis=1) * 100
    
    # Crear gráfico de barras apiladas
    fig = go.Figure()
    
    for category in sequence_percent.index:
        fig.add_trace(go.Bar(
            name=category,
            x=['1st Purchase', '2nd Purchase', '3rd Purchase'],
            y=[sequence_percent.loc[category, i] if i in sequence_percent.columns else 0 for i in [1, 2, 3]]
        ))
    
    fig.update_layout(
        title="Product Category Distribution by Purchase Number",
        xaxis_title="Purchase Number",
        yaxis_title="Percentage of Purchases",
        barmode='stack'
    )
    
    charts.safe_plotly_chart(fig)

def display_time_to_repurchase_by_category(all_orders: pd.DataFrame):
    """
    Analiza el tiempo entre compras por categoría de primera compra.
    """
    st.subheader("Time to Repurchase by First Purchase Category")
    
    if all_orders.empty:
        st.info("No data available for repurchase timing analysis.")
        return
    
    # Preparar datos - asegurar que customer_id sea string para consistencia
    customer_orders = all_orders.copy()
    customer_orders['customer_id'] = customer_orders['customer_id'].astype(str)
    customer_orders = customer_orders.sort_values(['customer_id', 'order_create_date'])
    
    # Primera compra por cliente
    first_purchases = customer_orders.groupby('customer_id').first().reset_index()
    
    # Clientes con segunda compra
    customer_order_counts = customer_orders.groupby('customer_id').size()
    repeat_customers = customer_order_counts[customer_order_counts >= 2].index
    
    if len(repeat_customers) == 0:
        st.info("No customers with repeat purchases found.")
        return
    
    # Segunda compra
    second_purchases = customer_orders[customer_orders['customer_id'].isin(repeat_customers)].groupby('customer_id').nth(1).reset_index()
    
    # Merge para calcular tiempo entre compras
    first_purchase_data = first_purchases[first_purchases['customer_id'].isin(repeat_customers)][['customer_id', 'product_category', 'order_create_date']]
    first_purchase_data.columns = ['customer_id', 'first_category', 'first_order_date']
    
    second_purchase_data = second_purchases[['customer_id', 'order_create_date']]
    second_purchase_data.columns = ['customer_id', 'second_order_date']
    
    # Calcular tiempo entre compras usando merge
    time_analysis = pd.merge(first_purchase_data, second_purchase_data, on='customer_id')
    time_analysis['days_to_repurchase'] = (time_analysis['second_order_date'] - time_analysis['first_order_date']).dt.days
    
    if time_analysis.empty:
        st.info("No time analysis data could be created.")
        return
    
    # Análisis por categoría
    repurchase_timing = time_analysis.groupby('first_category')['days_to_repurchase'].agg(['mean', 'median', 'count']).round(1)
    repurchase_timing.columns = ['Avg Days', 'Median Days', 'Customers']
    
    # Mostrar tabla
    st.dataframe(
        repurchase_timing,
        column_config={
            "Avg Days": st.column_config.NumberColumn("Avg Days to Repurchase", format="%.1f"),
            "Median Days": st.column_config.NumberColumn("Median Days to Repurchase", format="%.1f"),
            "Customers": st.column_config.NumberColumn("# Customers", format="%d")
        },
        use_container_width=True
    )
    
    # Gráfico de distribución
    fig = px.box(
        time_analysis, 
        x='first_category', 
        y='days_to_repurchase',
        title="Distribution of Days to Repurchase by First Purchase Category"
    )
    fig.update_xaxes(title="First Purchase Category")
    fig.update_yaxes(title="Days to Repurchase")
    charts.safe_plotly_chart(fig)

def render(all_data: Dict[str, pd.DataFrame], filtered_data: Dict[str, pd.DataFrame]):
    """
    Renderiza el tab completo de Customer Journey.
    """
    st.header("🛒 Customer Journey & Behavior Analysis")
    st.markdown("Deep dive into customer purchase patterns, category transitions, and sport-specific behaviors.")
    
    # --- Data Preparation ---
    all_orders = all_data.get("orders", pd.DataFrame())
    
    if all_orders.empty:
        st.warning("Complete order history is required for customer journey analysis.")
        return
    
    # Enriquecer datos con categorías
    all_orders_enriched = utils.enrich_dataframe_with_categories(all_orders)
    
    # --- KPI Display ---
    kpi_summary = analyze_first_vs_repeat_purchases(all_orders_enriched)
    if kpi_summary:
        metrics.create([{"label": k, "value": v} for k, v in kpi_summary.items()], num_columns=len(kpi_summary))
    
    st.markdown("---")
    
    # --- Analysis Sections ---
    display_category_transition_matrix(all_orders_enriched)
    st.markdown("---")
    
    display_sport_universe_behavior(all_orders_enriched)
    st.markdown("---")
    
    display_purchase_sequence_analysis(all_orders_enriched)
    st.markdown("---")
    
    display_time_to_repurchase_by_category(all_orders_enriched) 