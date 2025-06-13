# components/charts.py
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
from typing import Optional, Dict

import config

def safe_plotly_chart(fig: Optional[go.Figure], use_container_width: bool = True):
    """
    Renders a Plotly figure in Streamlit, displaying an info message if the figure is None.
    """
    if fig:
        st.plotly_chart(fig, use_container_width=use_container_width)
    else:
        st.info("No data available to display this chart with the selected filters.")

def create_bar_chart(data: pd.DataFrame, x: str, y: str, title: Optional[str] = None, 
                    labels: Optional[Dict[str, str]] = None,
                    color: Optional[str] = None,
                    barmode: str = 'relative',
                    height: int = 400) -> Optional[go.Figure]:
    """
    Creates a standardized bar chart.
    """
    if data is None or data.empty:
        return None
    
    try:
        fig = px.bar(
            data, x=x, y=y,
            title=title,
            labels=labels or {}, 
            color=color,
            barmode=barmode
        )
        fig.update_layout(height=height, margin=dict(t=60, b=20, l=20, r=20))
        return fig
    except Exception as e:
        st.error(f"Error creating bar chart '{title}': {e}")
        return None

def create_line_chart(data: pd.DataFrame, x: str, y: str, title: Optional[str] = None,
                     labels: Optional[Dict[str, str]] = None,
                     markers: bool = True,
                     height: int = 400) -> Optional[go.Figure]:
    """
    Creates a standardized line chart.
    """
    if data is None or data.empty:
        return None
    
    try:
        fig = px.line(
            data, x=x, y=y,
            title=title,
            labels=labels or {},
            markers=markers
        )
        fig.update_layout(height=height, margin=dict(t=60, b=20, l=20, r=20))
        return fig
    except Exception as e:
        st.error(f"Error creating line chart '{title}': {e}")
        return None

def create_scatter_chart(
    data: pd.DataFrame, x: str, y: str, title: Optional[str] = None,
    color: Optional[str] = None, size: Optional[str] = None,
    labels: Optional[dict] = None, height: int = 400
) -> Optional[go.Figure]:
    """
    Creates a standardized scatter plot.
    """
    if data.empty:
        return None
    try:
        fig = px.scatter(
            data, x=x, y=y,
            title=title,
            color=color, size=size,
            labels=labels,
            hover_data=[data.index]
        )
        fig.update_layout(height=height, margin=dict(t=60, b=20, l=20, r=20))
        return fig
    except Exception as e:
        st.error(f"Error creating scatter chart '{title}': {e}")
        return None

def create_choropleth_map(data: pd.DataFrame, locations_col: str, values_col: str, title: str) -> Optional[go.Figure]:
    """
    Creates a US choropleth map to visualize geographic data.
    """
    if data is None or data.empty:
        return None

    try:
        fig = px.choropleth(
            data,
            locations=locations_col,
            locationmode="USA-states",
            color=values_col,
            scope="usa",
            title=title,
            color_continuous_scale=px.colors.sequential.Blues,
            labels={values_col: 'Total Sales'}
        )
        fig.update_layout(
            geo=dict(lakecolor='white'),
            margin=dict(t=60, b=20, l=20, r=20)
        )
        return fig
    except Exception as e:
        st.error(f"Error creating map chart '{title}': {e}")
        return None

def create_correlation_heatmap(
    data: pd.DataFrame, title: str,
    x_col: Optional[str] = None, y_col: Optional[str] = None,
    normalize: Optional[str] = 'index'
) -> Optional[go.Figure]:
    """
    Creates a heatmap to show the relationship between two categorical variables.
    """
    if data is None or data.empty:
        return None
    
    try:
        # If the crosstab is pre-calculated
        if x_col is None and y_col is None:
            crosstab = data
        else:
            crosstab = pd.crosstab(data[y_col], data[x_col], normalize=normalize)

        fig = px.imshow(
            crosstab,
            title=title,
            text_auto=".0%",
            aspect="auto",
            color_continuous_scale=px.colors.sequential.Blues,
            labels=dict(x=x_col.replace('_', ' ').title() if x_col else "", 
                        y=y_col.replace('_', ' ').title() if y_col else "", 
                        color="Proportion")
        )
        fig.update_layout(margin=dict(t=60, b=20, l=20, r=20))
        return fig
    except Exception as e:
        st.error(f"Error creating heatmap '{title}': {e}")
        return None

def create_sankey_diagram(data: pd.DataFrame, source: str, target: str, value: str, title: str) -> Optional[go.Figure]:
    """
    Creates a Sankey diagram to visualize flow between two states (e.g., category migration).
    """
    if data is None or data.empty:
        return None
        
    try:
        labels = pd.unique(data[[source, target]].values.ravel('K'))
        label_map = {label: i for i, label in enumerate(labels)}
        source_indices = data[source].map(label_map)
        target_indices = data[target].map(label_map)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=data[value]
            )
        )])
        fig.update_layout(title_text=title, font_size=12)
        return fig
    except Exception as e:
        st.error(f"Error creating Sankey diagram '{title}': {e}")
        return None 