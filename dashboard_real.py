import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página y tema personalizado
st.set_page_config(
    layout="wide",
    page_title="Customer Analytics Dashboard",
    page_icon="👥",
    initial_sidebar_state="expanded"
)

# Global CSS for styling
st.markdown("""
<style>
    /* Main dashboard font */
    body, .stApp { /* Apply font to the whole app */
        font-family: 'Source Sans Pro', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Headers styling */
    h1, h2, h3 { /* Targeting Streamlit's markdown headers */
        color: #005A5A; /* Darker Teal for headers */
        font-weight: 600;
    }
    
    /* Improve st.metric look and feel */
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 18px 20px; /* Adjusted padding */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05); /* Softer shadow */
        border: 1px solid #E0E0E0;
        transition: box-shadow 0.3s ease-in-out;
    }
    .stMetric:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }

    /* Style the help text within st.metric */
    .stMetric [data-testid="stMetricHelp"] p {
        font-style: italic;
        font-size: 0.82rem; /* Slightly smaller */
        color: #555E67; /* Softer color */
        line-height: 1.3;
        margin-top: 5px; /* Add some space above help text */
    }
    .stMetric label { /* Metric Label */
        font-weight: 500; /* Make label a bit bolder */
        color: #4B5563; /* Custom color for label */
    }
    .stMetric div[data-testid="stMetricValue"] { /* Metric Value */
        font-size: 2.1rem; /* Slightly larger value */
        font-weight: 600;
        color: #008080; /* Teal color for value */
    }
     .stMetric div[data-testid="stMetricDelta"] { /* Metric Delta */
        font-size: 0.85rem;
        font-weight: 500;
    }


    /* General descriptive text under tabs or section titles */
    .dashboard-subtext { /* Renamed class for clarity */
        font-size: 0.9rem;
        color: #4A5568; /* Slightly muted color */
        margin-bottom: 1.2rem; /* Space below description */
        font-style: normal;
        line-height: 1.5;
    }
    
    /* Style st.tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px; /* Space between tabs */
        border-bottom: 1px solid #D1D5DB; /* Lighter border for tab list */
        padding-bottom: 0px; /* Align with bottom border of active tab */
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent; /* Make inactive tabs blend more */
        border-radius: 6px 6px 0px 0px;
        padding: 0px 16px; /* Adjust padding */
        margin-bottom: -1px; /* Overlap with bottom border */
        border: 1px solid transparent; /* For border consistency on hover/active */
        font-weight: 500;
        color: #4B5563; /* Inactive tab text color */
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E5E7EB; /* Light hover effect */
        color: #1F2937;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF; /* Active tab background (secondaryBackgroundColor) */
        color: #008080; /* Active tab text color (primaryColor) */
        border-color: #D1D5DB #D1D5DB #FFFFFF; /* Border for active tab */
        font-weight: 600;
    }

    /* Warnings and infos */
    .stAlert {
        border-radius: 6px;
    }
    .stAlert[data-baseweb="alert"] > div:first-child { /* Icon part */
        padding-top: 10px !important; 
    }
    .stAlert[data-baseweb="alert"] p { /* Text part */
        font-size: 0.88rem;
    }
    
    /* Sidebar styling for missing data points */
    [data-testid="stSidebar"] .stAlert[data-baseweb="alert"] p { /* Text part in sidebar warnings */
        font-size: 0.8rem;
    }
    [data-testid="stSidebar"] h3 { /* Sidebar headers */
        color: #006A6A; /* Slightly different teal for sidebar headers */
        font-size: 1.1rem;
    }
    [data-testid="stSidebar"] .stRadio > label { /* Sidebar radio labels */
        font-size: 0.95rem;
    }
    
    /* Style Plotly chart titles (if they render as SVG text with specific classes - usually not directly targetable for sub-parts like <sub>) */
    /* Subtitles within Plotly charts are best handled using the <br><sub>HTML</sub> approach in the title string itself */

    /* General st.info styling */
    .stInfo {
        background-color: #E0F2FE !important; /* Light blue background */
        border: 1px solid #7DD3FC !important; /* Blue border */
        color: #0C5475 !important; /* Darker blue text */
        border-radius: 6px !important;
    }
    .stInfo p {
         color: #0C5475 !important;
         font-size: 0.88rem;
    }


</style>
""", unsafe_allow_html=True)

# —————————————————————————————————————————
# Carga de datos reales
# —————————————————————————————————————————
DATA_DIR = "bigquery_data_20250513_171339"
missing_data_points = []

@st.cache_data(ttl=3600)
def load_real_data():
    """Carga los datos reales desde los archivos CSV."""
    global missing_data_points
    try:
        # Cargar datos de órdenes
        df_items = pd.read_csv(f"{DATA_DIR}/SHOPIFY_ORDERS_LINEITEMS.csv")
        df_items['ORDER_CREATE_DATE'] = pd.to_datetime(df_items['ORDER_CREATE_DATE']).dt.tz_localize(None)
        # Asegurar que USER_EMAIL es string y no hay NaN para evitar errores en groupby
        df_items['USER_EMAIL'] = df_items['USER_EMAIL'].astype(str).fillna('unknown_email')
        df_items['PRODUCT_NAME'] = df_items['PRODUCT_NAME'].astype(str).fillna('Unknown Product')

        
        # Cargar datos de sesiones
        df_sessions = pd.read_csv(f"{DATA_DIR}/GA_SESSIONS.csv")
        df_sessions['EVENT_DATE'] = pd.to_datetime(df_sessions['EVENT_DATE']).dt.tz_localize(None)
        if 'BOUNCE_RATE' not in df_sessions.columns:
            missing_data_points.append("Tasa de Rebote (Bounce Rate) de Sesiones (columna 'BOUNCE_RATE' no encontrada en GA_SESSIONS)")
        if 'AVG_SESSION_DURATION' not in df_sessions.columns:
            missing_data_points.append("Duración Promedio de Sesión (columna 'AVG_SESSION_DURATION' no encontrada en GA_SESSIONS)")
        if 'ADD_TO_CART_EVENTS' not in df_sessions.columns or 'CHECKOUT_INITIATED_EVENTS' not in df_sessions.columns: # Example columns
            missing_data_points.append("Tasa de Abandono de Carrito (CAR) (datos de eventos de carrito no encontrados)")

        
        # Cargar datos de órdenes GA
        df_ga_orders = pd.read_csv(f"{DATA_DIR}/GA_ORDERS.csv")
        df_ga_orders['Date'] = pd.to_datetime(df_ga_orders['Date']).dt.tz_localize(None)
        df_ga_orders['Transaction_ID'] = df_ga_orders['Transaction_ID'].astype(str)
        df_items['ORDER_GID_STR'] = df_items['ORDER_GID'].astype(str) # Para el merge
        
        # Cargar datos de campañas
        df_ads = pd.read_csv(f"{DATA_DIR}/GA_ADS_CAMPAIGNS.csv")
        df_ads['Date'] = pd.to_datetime(df_ads['Date']).dt.tz_localize(None)
        if 'LEADS_GENERATED' not in df_ads.columns: # Example column
            missing_data_points.append("Costo por Lead (CPL) (datos de leads no encontrados)")
        
        # NPS Data
        missing_data_points.append("Índice de Satisfacción NPS (datos de NPS generalmente provienen de encuestas y no están en los CSVs)")
        
        return df_items, df_sessions, df_ga_orders, df_ads
    except Exception as e:
        st.error(f"Error cargando los datos: {str(e)}")
        return None, None, None, None

# —————————————————————————————————————————
# Funciones de análisis EDA
# —————————————————————————————————————————
def analyze_numerical_columns(df, columns):
    """Análisis estadístico de columnas numéricas."""
    stats_df = pd.DataFrame()
    for col in columns:
        stats = df[col].describe()
        stats_df[col] = stats
    return stats_df

def analyze_categorical_columns(df, columns):
    """Análisis de columnas categóricas."""
    cat_stats = {}
    for col in columns:
        value_counts = df[col].value_counts()
        cat_stats[col] = {
            'unique_values': len(value_counts),
            'top_values': value_counts.head(5).to_dict(),
            'missing_values': df[col].isnull().sum()
        }
    return cat_stats

def plot_correlation_matrix(df, columns):
    """Genera matriz de correlación para columnas numéricas."""
    corr_matrix = df[columns].corr()
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlación"),
                    x=columns,
                    y=columns,
                    color_continuous_scale='RdBu_r')
    fig.update_layout(title="Matriz de Correlación")
    return fig

def plot_distribution(df, column, bins=30):
    """Genera gráfico de distribución para una columna numérica."""
    fig = px.histogram(df, x=column, nbins=bins,
                      title=f"Distribución de {column}")
    return fig

def plot_time_series(df, date_column, value_column, title):
    """Genera gráfico de serie temporal."""
    fig = px.line(df, x=date_column, y=value_column,
                  title=title)
    return fig

def analyze_outliers(df, column):
    """Detecta y analiza outliers en una columna numérica."""
    if column not in df.columns or df[column].isnull().all():
        return {
            'outliers_count': 0,
            'outliers_percentage': 0,
            'lower_bound': np.nan,
            'upper_bound': np.nan,
            'error': f"Columna {column} no encontrada o vacía."
        }
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return {
        'outliers_count': len(outliers),
        'outliers_percentage': len(outliers) / len(df) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

# —————————————————————————————————————————
# Carga y procesamiento de datos
# —————————————————————————————————————————
df_items, df_sessions, df_ga_orders, df_ads = load_real_data()

if df_items is None or df_sessions is None or df_ga_orders is None or df_ads is None:
    st.error("No se pudieron cargar algunos o todos los datos. Por favor, verifica que los archivos CSV existan y su contenido.")
    st.stop()

# Sidebar con diseño mejorado
with st.sidebar:
    # st.image("https://via.placeholder.com/200x60?text=CUSTOMER+ANALYTICS", width=200) # Eliminada o comentada
    st.markdown("---")
    
    st.markdown("### 📅 Filtro de Fechas Global")
    st.markdown("Selecciona el rango de fechas para el análisis general del dashboard:")
    
    # Obtener fechas mínimas y máximas de los datos para los defaults del date_input
    min_date_items_sidebar = df_items['ORDER_CREATE_DATE'].min() if not df_items.empty else pd.Timestamp.min.tz_localize(None)
    max_date_items_sidebar = df_items['ORDER_CREATE_DATE'].max() if not df_items.empty else pd.Timestamp.max.tz_localize(None)
    
    min_date_sessions_sidebar = df_sessions['EVENT_DATE'].min() if not df_sessions.empty else pd.Timestamp.min.tz_localize(None)
    max_date_sessions_sidebar = df_sessions['EVENT_DATE'].max() if not df_sessions.empty else pd.Timestamp.max.tz_localize(None)

    min_date_ga_orders_sidebar = df_ga_orders['Date'].min() if not df_ga_orders.empty else pd.Timestamp.min.tz_localize(None)
    max_date_ga_orders_sidebar = df_ga_orders['Date'].max() if not df_ga_orders.empty else pd.Timestamp.max.tz_localize(None)

    min_date_ads_sidebar = df_ads['Date'].min() if not df_ads.empty else pd.Timestamp.min.tz_localize(None)
    max_date_ads_sidebar = df_ads['Date'].max() if not df_ads.empty else pd.Timestamp.max.tz_localize(None)

    # Global min and max dates for the date picker constraints
    overall_min_date = min(min_date_items_sidebar, min_date_sessions_sidebar, min_date_ga_orders_sidebar, min_date_ads_sidebar)
    overall_max_date = max(max_date_items_sidebar, max_date_sessions_sidebar, max_date_ga_orders_sidebar, max_date_ads_sidebar)

    # Default start date: 12 months before the overall_max_date, or 12 months before today if no data
    default_start_date = (overall_max_date - pd.DateOffset(months=12)).date() \
        if pd.notnull(overall_max_date) and overall_max_date != pd.Timestamp.max.tz_localize(None) \
        else (datetime.now() - timedelta(days=365)).date()
    
    # Default end date: overall_max_date, or today if no data
    default_end_date = overall_max_date.date() \
        if pd.notnull(overall_max_date) and overall_max_date != pd.Timestamp.max.tz_localize(None) \
        else datetime.now().date()

    start_date_input, end_date_input = st.date_input(
        "Selecciona el Rango de Fechas:",
        value=[default_start_date, default_end_date],
        min_value=overall_min_date.date() if pd.notnull(overall_min_date) and overall_min_date != pd.Timestamp.min.tz_localize(None) else None,
        max_value=overall_max_date.date() if pd.notnull(overall_max_date) and overall_max_date != pd.Timestamp.max.tz_localize(None) else None,
        help="Este filtro de fechas se aplica globalmente a la mayoría de los análisis del dashboard."
        )

    # Convertir fechas a datetime
    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)

    st.markdown("---")
    st.markdown("### 📖 Guía del Dashboard")
    st.markdown("""
        Este dashboard proporciona un análisis detallado de los clientes basado en los datos de eCommerce.
        Utiliza los filtros de período para enfocar el análisis en rangos de tiempo específicos.
        Cada pestaña se centra en un aspecto diferente del comportamiento y valor del cliente.
    """)

# Filtrar DataFrames principales por el rango de fechas seleccionado
df_items_filtered = df_items[df_items['ORDER_CREATE_DATE'].between(start_date, end_date)]
df_sessions_filtered = df_sessions[df_sessions['EVENT_DATE'].between(start_date, end_date)]
df_ga_orders_filtered = df_ga_orders[df_ga_orders['Date'].between(start_date, end_date)]
df_ads_filtered = df_ads[df_ads['Date'].between(start_date, end_date)]


# —————————————————————————————————————————
# Dashboard con Streamlit
# —————————————————————————————————————————
st.title("👥 Customer Analytics Dashboard")

st.warning("""
    **⚠️ ESTE DASHBOARD ES UNA PRUEBA DE CONCEPTO (POC) ⚠️**

    Los datos mostrados aquí son una **muestra utilizada para el desarrollo** y **NO deben usarse para tomar decisiones de negocio.**
    
    El objetivo actual es recopilar feedback sobre:
    - ¿Qué información adicional sería valiosa?
    - ¿Cómo podemos mejorar la visualización de los datos actuales?
    - ¿Hay secciones o gráficos que no se entienden claramente?
    
    """, icon="📢")

# Tabs según customer_analysis.md
tab_vision, tab_frecuencia, tab_valor, tab_comportamiento, tab_trafico, tab_costo, tab_nps = st.tabs([
    "🌐 Visión General del Cliente", 
    "🔄 Frecuencia y Recurrencia", 
    "💰 Valor del Cliente",
    "🛒 Comportamiento de Compra", 
    "📈 Traffic Analytics", 
    "💸 Costo de Adquisición",
    "😊 NPS"
])

# Helper function para evitar errores con data vacía en gráficos
def safe_plotly_chart(fig, use_container_width=True):
    if fig is not None:
        st.plotly_chart(fig, use_container_width=use_container_width)
    else:
        st.info("No hay datos suficientes para mostrar este gráfico con los filtros seleccionados.")

# Tab 1: Visión General del Cliente
with tab_vision:
    st.markdown("### 🌐 Visión General del Cliente")
    st.markdown("<div class='dashboard-subtext'>Una mirada a quiénes son tus clientes, de dónde vienen y cómo se distribuyen, según el período seleccionado.</div>", unsafe_allow_html=True)

    # --- Métrica Global Independiente ---
    if not df_items.empty:
        total_historical_customers = df_items['USER_EMAIL'].nunique()
        st.metric(
            label="👥 Total Clientes Históricos (Global)",
            value=f"{total_historical_customers:,}",
            help="Número total de clientes únicos (`USER_EMAIL`) identificados en todo el historial de datos de órdenes (`SHOPIFY_ORDERS_LINEITEMS.csv`). No se aplica ningún filtro de fecha a esta métrica."
        )
    else:
        st.warning("No hay datos de órdenes para calcular el total de clientes históricos.")
    st.markdown("---")

    # --- Selector de Período para la Pestaña Visión General ---
    first_purchase_dates_all_time = df_items.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].min().rename('FIRST_PURCHASE_DATE_GLOBAL')
    
    df_items_for_tab_vision = pd.DataFrame()
    current_start_date_tab_vision = None
    current_end_date_tab_vision = None
    selected_period_label_tab_vision = "Usar Filtro Global de Fechas" # Default
    subtitle_period_string_tab_vision = ""

    if not df_items.empty and pd.notnull(df_items['ORDER_CREATE_DATE'].max()):
        data_max_date_tab = df_items['ORDER_CREATE_DATE'].max()
        data_min_date_tab = df_items['ORDER_CREATE_DATE'].min()

        periods_options_tab = [
            (3, "Últimos 3 Meses"), 
            (6, "Últimos 6 Meses"), 
            (12, "Últimos 12 Meses"), 
            (None, "Histórico Completo"), 
            ("GLOBAL", "Usar Filtro Global de Fechas")
        ]
        period_labels_tab = [label for _, label in periods_options_tab]
        
        selected_period_label_tab_vision = st.radio(
            "Selecciona el período para el análisis en esta pestaña:",
            options=period_labels_tab,
            horizontal=True,
            index=len(periods_options_tab) - 1, 
            key="tab_vision_period_selector"
        )

        if selected_period_label_tab_vision == "Usar Filtro Global de Fechas":
            df_items_for_tab_vision = df_items_filtered.copy()
            current_start_date_tab_vision = start_date # Global start_date from sidebar
            current_end_date_tab_vision = end_date   # Global end_date from sidebar
            if pd.notnull(current_start_date_tab_vision) and pd.notnull(current_end_date_tab_vision):
                subtitle_period_string_tab_vision = f"Período Global: {current_start_date_tab_vision.strftime('%d %b %Y')} - {current_end_date_tab_vision.strftime('%d %b %Y')}"
            else:
                 subtitle_period_string_tab_vision = "Filtro Global no definido"
        else:
            num_months_tab = None
            is_full_history_tab = False
            for num_m, lbl in periods_options_tab:
                if lbl == selected_period_label_tab_vision:
                    num_months_tab = num_m
                    if num_m is None and lbl == "Histórico Completo":
                        is_full_history_tab = True
                    break
            
            if is_full_history_tab:
                current_start_date_tab_vision = data_min_date_tab
                current_end_date_tab_vision = data_max_date_tab
                df_items_for_tab_vision = df_items.copy()
                subtitle_period_string_tab_vision = f"Histórico Completo: {data_min_date_tab.strftime('%d %b %Y')} - {data_max_date_tab.strftime('%d %b %Y')}"
            elif num_months_tab is not None:
                current_start_date_tab_vision = data_max_date_tab - pd.DateOffset(months=num_months_tab)
                if current_start_date_tab_vision < data_min_date_tab:
                    current_start_date_tab_vision = data_min_date_tab
                current_end_date_tab_vision = data_max_date_tab
                df_items_for_tab_vision = df_items[df_items['ORDER_CREATE_DATE'].between(current_start_date_tab_vision, current_end_date_tab_vision)]
                subtitle_period_string_tab_vision = f"{selected_period_label_tab_vision}: {current_start_date_tab_vision.strftime('%d %b %Y')} - {current_end_date_tab_vision.strftime('%d %b %Y')}"
    else:
        st.warning("No hay datos de órdenes disponibles para la pestaña de Visión General.")
        # Set dummy df to avoid errors, or st.stop() if preferred
        df_items_for_tab_vision = pd.DataFrame(columns=df_items.columns) 
        current_start_date_tab_vision = pd.Timestamp.min.tz_localize(None)
        current_end_date_tab_vision = pd.Timestamp.max.tz_localize(None)
        subtitle_period_string_tab_vision = "Datos no disponibles"


    st.markdown(f"#### Resumen para: {selected_period_label_tab_vision}")
    st.markdown(f"<div class='dashboard-subtext' style='margin-bottom:1rem;'>{subtitle_period_string_tab_vision}</div>", unsafe_allow_html=True)

    if not df_items_for_tab_vision.empty and pd.notnull(current_start_date_tab_vision) and pd.notnull(current_end_date_tab_vision):
        # --- Sección 1: Métricas Clave del Período Seleccionado ---
        # Ajustar columnas para incluir los nuevos KPIs de porcentaje
        col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)

        total_unique_in_tab_period = df_items_for_tab_vision['USER_EMAIL'].nunique()
        with col_metric1:
            st.metric(label="Total Clientes Únicos", value=f"{total_unique_in_tab_period:,}",
                      help=f"Número de clientes únicos (`USER_EMAIL`) que realizaron al menos una compra dentro del período seleccionado para esta pestaña ({subtitle_period_string_tab_vision}). Se basa en `SHOPIFY_ORDERS_LINEITEMS.csv` filtrado.")

        df_enriched_tab_vision = df_items_for_tab_vision.merge(first_purchase_dates_all_time.reset_index(), on='USER_EMAIL', how='left')
        
        new_in_tab_period = 0
        if 'FIRST_PURCHASE_DATE_GLOBAL' in df_enriched_tab_vision.columns: 
            new_in_tab_period = df_enriched_tab_vision[
                df_enriched_tab_vision['FIRST_PURCHASE_DATE_GLOBAL'].between(current_start_date_tab_vision, current_end_date_tab_vision)
            ]['USER_EMAIL'].nunique()
        
        recurrent_in_tab_period = total_unique_in_tab_period - new_in_tab_period
        
        with col_metric2:
            st.metric(label="Clientes Nuevos", value=f"{new_in_tab_period:,}",
                      help=f"Clientes únicos (`USER_EMAIL`) cuya primera compra *global* (histórica, de `SHOPIFY_ORDERS_LINEITEMS.csv`) ocurrió *dentro* del período seleccionado para esta pestaña ({subtitle_period_string_tab_vision}), y que también realizaron una compra en este mismo período.")
        with col_metric3:
            st.metric(label="Clientes Recurrentes", value=f"{recurrent_in_tab_period:,}",
                      help=f"Clientes únicos (`USER_EMAIL`) que realizaron compras en el período seleccionado para esta pestaña ({subtitle_period_string_tab_vision}), y cuya primera compra *global* (histórica) ocurrió *antes* del inicio de este período seleccionado.")

        # Calcular porcentajes para los nuevos KPIs
        perc_new_in_tab_period = 0
        perc_recurrent_in_tab_period = 0
        if total_unique_in_tab_period > 0:
            perc_new_in_tab_period = (new_in_tab_period / total_unique_in_tab_period) * 100
            perc_recurrent_in_tab_period = (recurrent_in_tab_period / total_unique_in_tab_period) * 100

        with col_metric4:
            st.metric(label="% Clientes Nuevos", 
                      value=f"{perc_new_in_tab_period:.2f}%" if total_unique_in_tab_period > 0 else "0%",
                      help=f"Porcentaje que representan los 'Clientes Nuevos' sobre el 'Total Clientes Únicos' en el período seleccionado para esta pestaña ({subtitle_period_string_tab_vision}).")
        
        with col_metric5:
            st.metric(label="% Clientes Recurrentes", 
                      value=f"{perc_recurrent_in_tab_period:.2f}%" if total_unique_in_tab_period > 0 else "0%",
                      help=f"Porcentaje que representan los 'Clientes Recurrentes' sobre el 'Total Clientes Únicos' en el período seleccionado para esta pestaña ({subtitle_period_string_tab_vision}).")

        # El código del pie chart ha sido eliminado de esta sección.
        
        st.markdown("---")
        # --- Sección 2: Evolución Mensual dentro del Período Seleccionado ---
        st.markdown("##### Evolución Mensual")
        # col_evo1, col_evo2 = st.columns(2) # Eliminado para disposición vertical

        # Gráfico de Evolución de Clientes Únicos Mensuales
        monthly_unique_tab = df_items_for_tab_vision.groupby(
            pd.Grouper(key='ORDER_CREATE_DATE', freq='M')
        )['USER_EMAIL'].nunique().reset_index()
        monthly_unique_tab.rename(columns={'USER_EMAIL': 'Clientes Únicos Mensuales', 'ORDER_CREATE_DATE': 'Mes'}, inplace=True)
        
        if not monthly_unique_tab.empty:
            fig_title_evo_unique = (
                f"Evolución de Clientes Únicos Mensuales<br>"
                f"<sub style='font-size:0.75em;'>Período: {subtitle_period_string_tab_vision}. Conteo de clientes (`USER_EMAIL`) únicos con compras por mes desde `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
            )
            fig_evo_unique = px.line(
                monthly_unique_tab, x='Mes', y='Clientes Únicos Mensuales', title=fig_title_evo_unique, markers=True
            )
            fig_evo_unique.update_layout(title_font_size=15, margin=dict(t=60, b=20, l=20, r=20), height=350)
            safe_plotly_chart(fig_evo_unique)
        else:
            st.caption("No hay datos de evolución de clientes únicos.")

        # Gráfico de Evolución de Nuevos vs. Recurrentes Mensual (Barras Apiladas)
        df_evolution_source_tab = df_items_for_tab_vision.merge(
            first_purchase_dates_all_time.reset_index(), on='USER_EMAIL', how='left'
        )
        if not df_evolution_source_tab.empty and 'FIRST_PURCHASE_DATE_GLOBAL' in df_evolution_source_tab.columns:
            df_evolution_source_tab['ORDER_MONTH_PERIOD'] = df_evolution_source_tab['ORDER_CREATE_DATE'].dt.to_period('M')
            df_evolution_source_tab['FIRST_PURCHASE_MONTH_PERIOD_GLOBAL'] = pd.to_datetime(df_evolution_source_tab['FIRST_PURCHASE_DATE_GLOBAL']).dt.to_period('M')
            df_evolution_source_tab['CUSTOMER_TYPE_FOR_MONTH'] = np.select(
                [df_evolution_source_tab['ORDER_MONTH_PERIOD'] == df_evolution_source_tab['FIRST_PURCHASE_MONTH_PERIOD_GLOBAL'],
                 df_evolution_source_tab['ORDER_MONTH_PERIOD'] > df_evolution_source_tab['FIRST_PURCHASE_MONTH_PERIOD_GLOBAL']],
                ['Nuevo en Mes', 'Recurrente en Mes'], default='Indeterminado' 
            )
            df_evo_classified_tab = df_evolution_source_tab[df_evolution_source_tab['CUSTOMER_TYPE_FOR_MONTH'] != 'Indeterminado']

            if not df_evo_classified_tab.empty:
                monthly_counts_tab = df_evo_classified_tab.groupby(
                    ['ORDER_MONTH_PERIOD', 'CUSTOMER_TYPE_FOR_MONTH']
                )['USER_EMAIL'].nunique().reset_index()
                monthly_counts_tab.rename(columns={'ORDER_MONTH_PERIOD': 'Mes', 'CUSTOMER_TYPE_FOR_MONTH': 'Tipo de Cliente', 'USER_EMAIL': 'Número de Clientes'}, inplace=True)
                monthly_counts_tab['Mes'] = monthly_counts_tab['Mes'].dt.to_timestamp()

                if not monthly_counts_tab.empty:
                    pivot_df_tab = monthly_counts_tab.pivot_table(index='Mes', columns='Tipo de Cliente', values='Número de Clientes', fill_value=0).reset_index()
                    if 'Nuevo en Mes' not in pivot_df_tab.columns: pivot_df_tab['Nuevo en Mes'] = 0
                    if 'Recurrente en Mes' not in pivot_df_tab.columns: pivot_df_tab['Recurrente en Mes'] = 0
                    pivot_df_tab.rename(columns={'Nuevo en Mes': 'Num_Nuevos', 'Recurrente en Mes': 'Num_Recurrentes'}, inplace=True)
                    pivot_df_tab['Total_Clientes_Mes'] = pivot_df_tab['Num_Nuevos'] + pivot_df_tab['Num_Recurrentes']
                    pivot_df_tab['Porc_Nuevos'] = np.where(pivot_df_tab['Total_Clientes_Mes'] > 0, (pivot_df_tab['Num_Nuevos'] / pivot_df_tab['Total_Clientes_Mes']) * 100, 0)
                    pivot_df_tab['Porc_Recurrentes'] = np.where(pivot_df_tab['Total_Clientes_Mes'] > 0, (pivot_df_tab['Num_Recurrentes'] / pivot_df_tab['Total_Clientes_Mes']) * 100, 0)
                    
                    data_bar_final_tab = pd.merge(monthly_counts_tab, pivot_df_tab[['Mes', 'Total_Clientes_Mes', 'Num_Nuevos', 'Porc_Nuevos', 'Num_Recurrentes', 'Porc_Recurrentes']], on='Mes', how='left')
                    
                    fig_stacked_bar_tab_title = (
                        f"Evolución Mensual: Nuevos vs. Recurrentes<br>"
                        f"<sub style='font-size:0.75em;'>Período: {subtitle_period_string_tab_vision}. 'Nuevo en Mes': 1ª compra global en ese mes. 'Recurrente': 1ª compra global anterior. De `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
                    )
                    fig_stacked_bar_tab = px.bar(
                        data_bar_final_tab, x='Mes', y='Número de Clientes', color='Tipo de Cliente', title=fig_stacked_bar_tab_title,
                        barmode='stack', custom_data=['Total_Clientes_Mes', 'Num_Nuevos', 'Porc_Nuevos', 'Num_Recurrentes', 'Porc_Recurrentes']
                    )
                    fig_stacked_bar_tab.update_traces(
                        hovertemplate=("<b>%{x|%b %Y}</b><br><br>" + "Total Clientes: %{customdata[0]} (100%)<br>" + "Clientes Nuevos: %{customdata[1]} (%{customdata[2]:.1f}%)<br>" + "Clientes Recurrentes: %{customdata[3]} (%{customdata[4]:.1f}%)" + "<extra></extra>")
                    )
                    fig_stacked_bar_tab.update_layout(title_font_size=15, margin=dict(t=60, b=20, l=20, r=20), height=350, legend_title_text='Tipo Cliente')
                    safe_plotly_chart(fig_stacked_bar_tab)
                else:
                    st.caption("No hay datos de evolución nuevos vs recurrentes.")
            else:
                st.caption("No se pudieron clasificar clientes para evolución nuevos vs recurrentes.")
        else:
            st.caption("Datos insuficientes para evolución nuevos vs recurrentes.")
        
        st.markdown("---")
        # --- Sección 3: Análisis Adicionales ---
        st.markdown(f"##### Análisis Adicionales para: {selected_period_label_tab_vision}")
        st.markdown(f"<div class='dashboard-subtext' style='margin-bottom:1rem;'>{subtitle_period_string_tab_vision}</div>", unsafe_allow_html=True)

        # col_acq, col_geo = st.columns(2) # Eliminado para disposición vertical

        # Gráfico de Canal de Adquisición
        st.markdown("###### 📣 Canales de Sesión en GA Orders (Período Global)")
        
        # Asegurarse que el dataframe base no está vacío
        if df_ga_orders_filtered.empty:
            st.caption("No hay datos en GA Orders para el período global seleccionado.")
        elif 'Session_primary_channel_group' not in df_ga_orders_filtered.columns:
            st.caption("La columna 'Session_primary_channel_group' no existe en los datos de GA Orders.")
        elif df_ga_orders_filtered['Session_primary_channel_group'].isnull().all():
            st.caption("La columna 'Session_primary_channel_group' en GA Orders está completamente vacía (todos los valores son nulos) para el período seleccionado.")
        else:
            # Contar la frecuencia de cada canal en el período filtrado globalmente
            ga_channel_counts_in_period = df_ga_orders_filtered['Session_primary_channel_group'].value_counts().reset_index()
            ga_channel_counts_in_period.columns = ['Canal de Sesión (GA)', 'Conteo de Sesiones con Transacción']
            
            # Filtrar filas donde el canal es nulo (si alguna quedara después del value_counts)
            ga_channel_counts_in_period = ga_channel_counts_in_period.dropna(subset=['Canal de Sesión (GA)'])

            if not ga_channel_counts_in_period.empty:
                fig_ga_channel_title = (
                    f"Distribución de Canales de Sesión (GA Orders)<br>"
                    f"<sub style='font-size:0.75em;'>Período Global: {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}. Fuente: `GA_ORDERS.csv`, columna `Session_primary_channel_group`.</sub>"
                )
                fig_ga_channels = px.bar(
                    ga_channel_counts_in_period, 
                    x='Canal de Sesión (GA)', 
                    y='Conteo de Sesiones con Transacción', 
                    title=fig_ga_channel_title,
                    text='Conteo de Sesiones con Transacción'
                )
                fig_ga_channels.update_traces(textposition='outside')
                fig_ga_channels.update_layout(
                    title_font_size=15, 
                    margin=dict(t=70, b=20, l=20, r=20),
                    height=400,
                    xaxis_title="Canal de Sesión (GA)",
                    yaxis_title="Conteo de Sesiones con Transacción"
                )
                safe_plotly_chart(fig_ga_channels)
            else:
                st.caption(f"No se encontraron datos de 'Session_primary_channel_group' válidos en GA Orders para el período global seleccionado: {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}.")

        # Gráfico de Geolocalización
        
        # 1. Mapa de Estados Unidos
        st.markdown("###### Mapa de Pedidos en Estados Unidos (por Estado)")
        df_us_items = df_items_for_tab_vision[df_items_for_tab_vision['SHIPPING_COUNTRY'].isin(['United States', 'US', 'USA'])] 
        
        if not df_us_items.empty and 'SHIPPING_STATE' in df_us_items.columns and df_us_items['SHIPPING_STATE'].notna().any():
            us_state_orders = df_us_items.groupby('SHIPPING_STATE')['ORDER_GID'].nunique().reset_index()
            us_state_orders.columns = ['Estado', 'Número de Pedidos'] # 'Estado' será usado para locations
            
            if 'SHIPPING_CITY' in df_us_items.columns:
                top_cities_per_state_list = []
                for state_name, group in df_us_items.groupby('SHIPPING_STATE'): # Usar SHIPPING_STATE
                    top_3_cities = group['SHIPPING_CITY'].value_counts().nlargest(3).index.tolist()
                    top_cities_str = ", ".join(top_3_cities)
                    top_cities_per_state_list.append({'Estado': state_name, 'Top Ciudades': top_cities_str})
                
                if top_cities_per_state_list:
                    top_cities_df = pd.DataFrame(top_cities_per_state_list)
                    us_state_orders = pd.merge(us_state_orders, top_cities_df, on='Estado', how='left')
                    us_state_orders['Top Ciudades'].fillna('N/A', inplace=True)
                else:
                    us_state_orders['Top Ciudades'] = 'N/A' 
            else:
                 us_state_orders['Top Ciudades'] = 'N/A'


            fig_us_map_title = (
                f"Mapa de Pedidos en EEUU por Estado<br>"
                f"<sub style='font-size:0.75em;'>Período: {selected_period_label_tab_vision}</sub>"
            )
            try:
                fig_us_map = px.choropleth(us_state_orders,
                                           locations='Estado', # Esta columna ahora viene de SHIPPING_STATE
                                           locationmode='USA-states',
                                           color='Número de Pedidos',
                                           scope='usa',
                                           hover_name='Estado',
                                           custom_data=['Top Ciudades'] if 'Top Ciudades' in us_state_orders else None,
                                           color_continuous_scale="Blues",
                                           title=fig_us_map_title)
                if 'Top Ciudades' in us_state_orders:
                     fig_us_map.update_traces(hovertemplate="<b>%{hovertext}</b><br>Pedidos: %{z}<br>Top Ciudades: %{customdata[0]}<extra></extra>")
                else:
                     fig_us_map.update_traces(hovertemplate="<b>%{hovertext}</b><br>Pedidos: %{z}<extra></extra>")

                fig_us_map.update_layout(title_font_size=15, margin=dict(t=60, b=0, l=0, r=0), height=450, geo=dict(bgcolor='rgba(0,0,0,0)'))
                safe_plotly_chart(fig_us_map)
            except Exception as e_us_map:
                st.warning(f"No se pudo generar el mapa de EEUU. Error: {e_us_map}")
                st.caption("Verifica que la columna 'SHIPPING_STATE' contenga nombres/abreviaturas de estados de EEUU válidos.")
        else:
            st.caption("No hay suficientes datos de pedidos en EEUU (o falta la columna 'SHIPPING_STATE') para generar el mapa de estados.")

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Gráfico de Barras por País (Global)
        st.markdown("###### Pedidos Globales por País")
        if 'SHIPPING_COUNTRY' in df_items_for_tab_vision.columns and not df_items_for_tab_vision['SHIPPING_COUNTRY'].isnull().all():
            all_country_orders = df_items_for_tab_vision.groupby('SHIPPING_COUNTRY')['ORDER_GID'].nunique().reset_index()
            all_country_orders.columns = ['País', 'Número de Pedidos']
            all_country_orders = all_country_orders.sort_values('Número de Pedidos', ascending=False).head(15) 

            if not all_country_orders.empty:
                fig_country_bar_title = (
                    f"Top 15 Países por Número de Pedidos<br>"
                    f"<sub style='font-size:0.75em;'>Período: {selected_period_label_tab_vision}</sub>"
                )
                fig_country_bar = px.bar(all_country_orders, x='País', y='Número de Pedidos', title=fig_country_bar_title)
                fig_country_bar.update_layout(title_font_size=15, margin=dict(t=60, b=20, l=20, r=20), height=400)
                safe_plotly_chart(fig_country_bar)
            else:
                st.caption("No hay datos de pedidos por país para el gráfico de barras.")
        else:
            st.caption("La columna 'SHIPPING_COUNTRY' no está disponible para el gráfico de barras por país.")


        # 3. Opcional: Top Ciudades en EEUU (Gráfico de Barras)
        st.markdown("###### Top Ciudades en Estados Unidos por Pedidos")
        if not df_us_items.empty and 'SHIPPING_CITY' in df_us_items.columns and df_us_items['SHIPPING_CITY'].notna().any() and 'SHIPPING_STATE' in df_us_items.columns:
            us_city_orders = df_us_items.groupby(['SHIPPING_STATE', 'SHIPPING_CITY'])['ORDER_GID'].nunique().reset_index(name='Número de Pedidos') # Usar SHIPPING_STATE
            us_city_orders['Lugar'] = us_city_orders['SHIPPING_CITY'] + ", " + us_city_orders['SHIPPING_STATE'] # Usar SHIPPING_STATE
            top_us_cities = us_city_orders.sort_values('Número de Pedidos', ascending=False).head(15)

            if not top_us_cities.empty:
                fig_us_city_title = (
                    f"Top 15 Ciudades en EEUU por Pedidos<br>"
                    f"<sub style='font-size:0.75em;'>Período: {selected_period_label_tab_vision}</sub>"
                )
                fig_us_city_bar = px.bar(top_us_cities, x='Lugar', y='Número de Pedidos', title=fig_us_city_title)
                fig_us_city_bar.update_layout(title_font_size=15, margin=dict(t=60, b=20, l=20, r=20), height=400)
                safe_plotly_chart(fig_us_city_bar)
            else:
                st.caption("No hay datos de pedidos por ciudad en EEUU para mostrar.")
        else:
            st.caption("No hay suficientes datos de ciudades/estados en EEUU para el gráfico de barras.")


    else: # df_items_for_tab_vision is empty or dates are not valid
        st.info(f"No hay datos de órdenes para el período seleccionado: '{selected_period_label_tab_vision}'. Por favor, ajusta la selección de período o el filtro global de fechas si aplica.")

# Tab 2: Frecuencia y Recurrencia de Compra
with tab_frecuencia:
    st.markdown("### 🔄 Frecuencia y Recurrencia de Compra")
    st.markdown("<div class='dashboard-subtext'>Análisis de la frecuencia con la que los clientes compran y cuánto tiempo pasa entre sus compras, basado en el período seleccionado en el filtro global de fechas.</div>", unsafe_allow_html=True)

    if not df_items_filtered.empty:
        customer_orders_freq = df_items_filtered.groupby('USER_EMAIL')['ORDER_GID'].nunique().reset_index(name='NUM_ORDERS')
        
        st.markdown("#### Métricas Clave de Frecuencia y Recurrencia")
        col_freq1, col_freq2, col_freq3 = st.columns(3)

        with col_freq1:
            avg_orders_per_customer = customer_orders_freq['NUM_ORDERS'].mean()
            st.metric("📦 Pedidos Promedio por Cliente", 
                      f"{avg_orders_per_customer:.2f}" if pd.notna(avg_orders_per_customer) else "N/A", 
                      help="Calculado como el número total de pedidos únicos dividido por el número total de clientes únicos (`USER_EMAIL`) que realizaron compras. Todo dentro del período global seleccionado y basado en datos de `SHOPIFY_ORDERS_LINEITEMS.csv`.")

        df_multi_orders_freq = pd.DataFrame() # Initialize df_multi_orders_freq
        with col_freq2:
            multi_order_customers_emails = customer_orders_freq[customer_orders_freq['NUM_ORDERS'] > 1]['USER_EMAIL']
            if not multi_order_customers_emails.empty:
                df_multi_orders_freq = df_items_filtered[df_items_filtered['USER_EMAIL'].isin(multi_order_customers_emails)].copy() 
                df_multi_orders_freq.sort_values(['USER_EMAIL', 'ORDER_CREATE_DATE'], inplace=True)
                df_multi_orders_freq['TIME_DIFF'] = df_multi_orders_freq.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].diff().dt.days
                avg_time_between_purchases = df_multi_orders_freq['TIME_DIFF'].mean() 
                st.metric("⏱️ Tiempo Promedio entre Compras", 
                          f"{avg_time_between_purchases:.1f} días" if pd.notna(avg_time_between_purchases) else "N/A", 
                          help="Promedio de días transcurridos entre compras consecutivas. Se calcula solo para clientes que han realizado más de un pedido dentro del período global seleccionado. Basado en `SHOPIFY_ORDERS_LINEITEMS.csv`.")
            else:
                st.metric("⏱️ Tiempo Promedio entre Compras", "N/A", help="No hay suficientes clientes con múltiples pedidos en el período para calcular. Se requiere más de un pedido por cliente en el período global seleccionado.")
        
        with col_freq3:
            recurrent_customer_count_freq = len(multi_order_customers_emails)
            total_customer_count_freq = customer_orders_freq['USER_EMAIL'].nunique()
            if total_customer_count_freq > 0:
                percentage_recurrent_freq = (recurrent_customer_count_freq / total_customer_count_freq) * 100
                st.metric("🔁 % Clientes Recurrentes", 
                          f"{percentage_recurrent_freq:.1f}%", 
                          help="Porcentaje de clientes únicos (`USER_EMAIL`) que realizaron más de un pedido dentro del período global seleccionado, sobre el total de clientes únicos en ese mismo período. Basado en `SHOPIFY_ORDERS_LINEITEMS.csv`.")
            else:
                st.metric("🔁 % Clientes Recurrentes", "N/A", help="No hay clientes en el período seleccionado.")
        
        st.markdown("---")
        st.markdown("#### 📊 Distribuciones de Frecuencia y Recurrencia")
        
        # Gráfico 1: Distribución del Nº de Pedidos por Cliente
        if not customer_orders_freq.empty:
            # Definir bins y etiquetas para agrupar el número de pedidos
            bins_num_orders = [0, 1, 2, 3, 5, 10, float('inf')]
            labels_num_orders = ['1 Pedido', '2 Pedidos', '3 Pedidos', '4-5 Pedidos', '6-10 Pedidos', '11+ Pedidos']
            
            # Crear una nueva columna con las categorías de pedidos
            customer_orders_freq['PEDIDOS_AGRUPADOS'] = pd.cut(
                customer_orders_freq['NUM_ORDERS'],
                bins=bins_num_orders,
                labels=labels_num_orders,
                right=True,
                include_lowest=True # Asegura que el 0 (si existiera) se incluya en el primer bin si el bin empieza en 0.
                                     # En este caso, NUM_ORDERS empieza en 1, por lo que [0,1] captura el 1.
            )
            
            # Contar clientes por categoría agrupada
            order_counts_grouped = customer_orders_freq['PEDIDOS_AGRUPADOS'].value_counts().reset_index()
            order_counts_grouped.columns = ['Grupo de Pedidos', 'Número de Clientes']
            
            # Asegurar el orden correcto de las categorías para el gráfico
            order_counts_grouped['Grupo de Pedidos'] = pd.Categorical(
                order_counts_grouped['Grupo de Pedidos'],
                categories=labels_num_orders,
                ordered=True
            )
            order_counts_grouped = order_counts_grouped.sort_values('Grupo de Pedidos')

            fig_dist_num_orders_title = (
                "Distribución del Nº de Pedidos por Cliente (Agrupado)<br>"
                "<sub>Cuántos clientes (`USER_EMAIL`) caen en rangos de cantidad de pedidos realizados en el período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
            )
            fig_dist_num_orders = px.bar( # Cambiado a px.bar
                order_counts_grouped, 
                x="Grupo de Pedidos", 
                y="Número de Clientes",
                title=fig_dist_num_orders_title
            )
            fig_dist_num_orders.update_layout(title_font_size=15, height=350, xaxis_title="Grupo de Cantidad de Pedidos")
            safe_plotly_chart(fig_dist_num_orders)
        else:
            st.caption("No hay datos para la distribución del número de pedidos.")

        # Gráfico 2: Distribución del Tiempo Entre Compras
        if not df_multi_orders_freq.empty and 'TIME_DIFF' in df_multi_orders_freq.columns and df_multi_orders_freq['TIME_DIFF'].notna().any():
            fig_dist_time_between_title = (
                "Distribución del Tiempo Entre Compras (Días)<br>"
                "<sub>Frecuencia de días entre compras consecutivas para clientes con >1 pedido en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
            )
            fig_dist_time_between = px.histogram(df_multi_orders_freq.dropna(subset=['TIME_DIFF']), x="TIME_DIFF",
                                                 title=fig_dist_time_between_title,
                                                 labels={"TIME_DIFF": "Días entre Compras Consecutivas"})
            fig_dist_time_between.update_layout(bargap=0.1, title_font_size=15, height=350)
            safe_plotly_chart(fig_dist_time_between)
        else:
            st.caption("No hay datos suficientes para la distribución del tiempo entre compras.")
        
        st.markdown("#### 📈 Segmentación por Frecuencia de Compra")
        if not customer_orders_freq.empty:
            bins = [0, 1, 3, 5, float('inf')]
            labels = ['1 Pedido (Comprador Único)', '2-3 Pedidos (Ocasional)', '4-5 Pedidos (Frecuente)', '6+ Pedidos (Leal)']
            customer_orders_freq['SEGMENTO_FRECUENCIA'] = pd.cut(customer_orders_freq['NUM_ORDERS'], bins=bins, labels=labels, right=True)
            
            segment_counts = customer_orders_freq['SEGMENTO_FRECUENCIA'].value_counts().reset_index()
            segment_counts.columns = ['Segmento por Frecuencia', 'Número de Clientes']
            segment_counts = segment_counts.sort_values(by='Segmento por Frecuencia', key=lambda x: x.map({label: i for i, label in enumerate(labels)}))


            fig_segment_freq_title = (
                "Segmentación de Clientes por Frecuencia de Pedidos<br>"
                "<sub>Clientes en segmentos (Comprador Único, Ocasional, etc.) según nº de pedidos en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
            )
            fig_segment_freq = px.bar(segment_counts, x="Segmento por Frecuencia", y="Número de Clientes",
                                      title=fig_segment_freq_title)
            fig_segment_freq.update_layout(title_font_size=15, height=400)
            safe_plotly_chart(fig_segment_freq)
        else:
            st.caption("No hay datos para segmentar por frecuencia.")


        st.markdown("---")
        st.markdown("#### 📅 Análisis de Cohortes (Retención Mensual)")
        st.markdown("""
        <div class='dashboard-subtext' style='font-size:0.88rem; line-height:1.4;'>
        Este análisis muestra qué porcentaje de clientes que hicieron su <i>primera compra global</i> en un mes específico ('cohorte')
        volvieron a comprar en los meses siguientes. La cohorte se define usando todos los datos históricos de órdenes, 
        y luego se observa su actividad de compra (retención) dentro del <b>período actualmente filtrado en el dashboard</b>.
        </div>
        """, unsafe_allow_html=True)
        
        # Se utiliza df_items (completo) para definir la COHORTE (mes de primera compra global)
        df_items_copy_cohort = df_items.copy() 
        df_items_copy_cohort['ORDER_MONTH'] = df_items_copy_cohort['ORDER_CREATE_DATE'].dt.to_period('M')
        df_items_copy_cohort['COHORT'] = df_items_copy_cohort.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].transform('min').dt.to_period('M')
        
        # Ahora, para ver la actividad de estas cohortes, filtramos por el df_items_filtered (período del dashboard)
        # y traemos la información de COHORTE original.
        df_cohort_data_activity_in_period = pd.merge(
            df_items_filtered.copy(), # Actividad en el período filtrado
            df_items_copy_cohort[['USER_EMAIL', 'COHORT']].drop_duplicates(subset=['USER_EMAIL']), # Cohorte global del cliente
            on='USER_EMAIL',
            how='left' # Nos quedamos solo con clientes activos en el período filtrado
        )
        # Necesitamos el mes de la orden también para la actividad en el período
        df_cohort_data_activity_in_period['ORDER_MONTH_ACTIVITY'] = df_cohort_data_activity_in_period['ORDER_CREATE_DATE'].dt.to_period('M')

        if not df_cohort_data_activity_in_period.empty and 'COHORT' in df_cohort_data_activity_in_period.columns:
            df_cohort_counts = df_cohort_data_activity_in_period.groupby(['COHORT', 'ORDER_MONTH_ACTIVITY']) \
                                            .agg(n_customers=('USER_EMAIL', 'nunique')) \
                                            .reset_index(drop=False)
            
            if not df_cohort_counts.empty:
                df_cohort_counts['PERIOD_NUMBER'] = (df_cohort_counts['ORDER_MONTH_ACTIVITY'] - df_cohort_counts['COHORT']).apply(lambda x: x.n if pd.notnull(x) else -1)
                # Filtrar periodos negativos que podrían surgir si la cohorte es posterior al mes de actividad (no debería pasar con 'left' merge y transform)
                df_cohort_counts = df_cohort_counts[df_cohort_counts['PERIOD_NUMBER'] >= 0]

                cohort_pivot = df_cohort_counts.pivot_table(index='COHORT',
                                                            columns='PERIOD_NUMBER',
                                                            values='n_customers')
                if not cohort_pivot.empty:
                    # Cohort size es el número de clientes únicos en la cohorte original (primera compra global)
                    cohort_size_df = df_items_copy_cohort.groupby('COHORT')['USER_EMAIL'].nunique().reset_index(name='TOTAL_CUSTOMERS_IN_COHORT')
                    cohort_pivot_with_size = cohort_pivot.reset_index().merge(cohort_size_df, on='COHORT', how='left').set_index('COHORT')
                    
                    cohort_matrix = cohort_pivot_with_size.iloc[:, :-1].divide(cohort_pivot_with_size['TOTAL_CUSTOMERS_IN_COHORT'], axis=0)
                    
                    # Preparar datos para el gráfico de líneas de retención promedio
                    if not cohort_matrix.empty:
                        # Calcular la retención promedio por cada período
                        avg_retention_curve = cohort_matrix.mean(axis=0) # Promedio por columna (PERIOD_NUMBER)
                        avg_retention_curve.index = avg_retention_curve.index.astype(int) # Asegurar que el índice es entero
                        avg_retention_curve = avg_retention_curve.sort_index()
                        avg_retention_df = avg_retention_curve.reset_index()
                        avg_retention_df.columns = ['Meses Desde Primera Compra', 'Tasa de Retención Promedio']
                        
                        # Convertir tasa a porcentaje para visualización
                        avg_retention_df['Tasa de Retención Promedio'] = avg_retention_df['Tasa de Retención Promedio'] * 100

                        # Descripción detallada del gráfico
                        st.markdown("""
                        <div class='dashboard-subtext' style='font-size:0.88rem; line-height:1.4; margin-bottom:0.5rem;'>
                        Este gráfico ilustra la <b>Curva de Retención Promedio Global</b>. Representa el porcentaje promedio de clientes que realizan compras adicionales en los meses siguientes a su mes de primera compra (definido como 'Mes 0').
                        <ul>
                            <li>El <b>eje X</b> ('Meses Desde Primera Compra') indica el número de meses transcurridos desde la primera compra del cliente.</li>
                            <li>El <b>eje Y</b> ('Tasa de Retención Promedio (%)') muestra el porcentaje de clientes, promediado entre todas las cohortes, que estuvieron activos (realizaron una compra) durante ese mes específico posterior a su adquisición.</li>
                            <li>Este análisis considera la actividad de compra de las cohortes dentro del <b>período global filtrado</b> en el dashboard.</li>
                        </ul>
                        Una curva descendente es típica, pero su pendiente y los puntos donde se estabiliza ofrecen información clave sobre la lealtad del cliente a largo plazo.
                        </div>
                        """, unsafe_allow_html=True)

                        fig_avg_retention_title = "Curva de Retención Promedio Global" # Título simplificado
                        
                        fig_avg_retention = px.line(
                            avg_retention_df, 
                            x='Meses Desde Primera Compra', 
                            y='Tasa de Retención Promedio',
                            title=fig_avg_retention_title,
                            markers=True,
                            labels={'Tasa de Retención Promedio': 'Tasa de Retención Promedio (%)'}
                        )
                        fig_avg_retention.update_layout(
                            yaxis_ticksuffix="%",
                            xaxis_dtick=1 # Mostrar cada mes en el eje X
                        )
                        safe_plotly_chart(fig_avg_retention)
                    else:
                        st.info("No hay suficientes datos para generar la curva de retención promedio (matriz de cohortes vacía).")
                else:
                    st.info("No hay suficientes datos para el análisis de cohortes (pivot vacío) con los filtros actuales.")
            else:
                st.info("No hay suficientes datos para el análisis de cohortes (counts vacíos) con los filtros actuales.")
        else:
            st.info("No hay datos de órdenes o cohortes para el período seleccionado (verifique `df_cohort_data_activity_in_period`).")
            if "Análisis de Cohortes (datos insuficientes o configuración compleja)" not in missing_data_points:
                missing_data_points.append("Análisis de Cohortes (datos insuficientes o configuración compleja)")
    else:
        st.info("No hay órdenes en el período seleccionado globalmente para analizar frecuencia y recurrencia.")

# Tab 3: Valor del Cliente
with tab_valor:
    st.markdown("### 💰 Valor del Cliente (LTV)")
    st.markdown("<div class='dashboard-subtext'>Entendiendo el valor que los clientes aportan a lo largo del tiempo, basado en el período seleccionado en el filtro global de fechas.</div>", unsafe_allow_html=True)

    if not df_items_filtered.empty:
        ltv_data = df_items_filtered.groupby('USER_EMAIL')['ORDER_TOTAL_PRICE'].sum().reset_index(name='LTV')
        
        st.markdown("#### 💸 LTV Promedio General (en período)")
        avg_ltv = ltv_data['LTV'].mean()
        st.metric("LTV Promedio (en período)", f"${avg_ltv:,.2f}" if pd.notna(avg_ltv) else "N/A", 
                  help="Valor de Vida del Cliente promedio. Se calcula sumando el `ORDER_TOTAL_PRICE` de todos los pedidos para cada cliente (`USER_EMAIL`) dentro del período global seleccionado, y luego promediando estas sumas entre todos los clientes. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.")

        st.markdown("#### 📊 LTV Promedio por Canal de Adquisición (Primer Pedido)")
        if not df_ga_orders.empty:
            first_order_info_valor = df_items.loc[df_items.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].idxmin()]
            merged_first_orders_valor = pd.merge(first_order_info_valor, df_ga_orders, 
                                           left_on='ORDER_GID_STR', right_on='Transaction_ID', 
                                           how='left')
            
            if not merged_first_orders_valor.empty and 'Session_primary_channel_group' in merged_first_orders_valor.columns:
                ltv_con_canal_adq = pd.merge(ltv_data, 
                                             merged_first_orders_valor[['USER_EMAIL', 'Session_primary_channel_group']].drop_duplicates(subset=['USER_EMAIL']),
                                             on='USER_EMAIL',
                                             how='left')
                
                if not ltv_con_canal_adq.empty and 'Session_primary_channel_group' in ltv_con_canal_adq.columns:
                    ltv_by_channel = ltv_con_canal_adq.groupby('Session_primary_channel_group')['LTV'].mean().reset_index()
                    ltv_by_channel = ltv_by_channel.sort_values('LTV', ascending=False)
                    fig_ltv_channel_title = (
                        "LTV Promedio por Canal de Adquisición del Cliente (1ª Compra Global)<br>"
                        "<sub>LTV (gasto total en período global) promediado por canal de 1ª compra global. Canales de `GA_ORDERS.csv`, LTV de `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
                    )
                    fig = px.bar(ltv_by_channel, x='Session_primary_channel_group', y='LTV', 
                                 title=fig_ltv_channel_title,
                                 labels={'Session_primary_channel_group': 'Canal de Adquisición', 'LTV': 'LTV Promedio ($)'})
                    safe_plotly_chart(fig)
                else:
                    st.info("No se pudo calcular LTV por canal de adquisición (datos de canal o LTV insuficientes).")
            else:
                st.warning("No se pudo determinar el canal de adquisición para LTV. Verifica la unión entre 'SHOPIFY_ORDERS_LINEITEMS' y 'GA_ORDERS'.")
                if "LTV por Canal de Adquisición (datos insuficientes o unión fallida)" not in missing_data_points:
                    missing_data_points.append("LTV por Canal de Adquisición (datos insuficientes o unión fallida)")
        else:
            st.info("Datos de GA Orders no disponibles para LTV por canal.")

        st.markdown("#### 🏆 Segmentación de Clientes por Ranking de Facturación (en período)")
        if not ltv_data.empty:
            ltv_data_sorted = ltv_data.sort_values('LTV', ascending=False)
            # Asegurar que hay suficientes valores únicos para qcut, o usar rangos fijos si no.
            try:
                ltv_data_sorted['RANK'] = pd.qcut(ltv_data_sorted['LTV'], q=4, labels=["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"], duplicates='drop')
            except ValueError: # Ocurre si no hay suficientes cuantiles distintos
                 ltv_data_sorted['RANK'] = pd.cut(ltv_data_sorted['LTV'], bins=4, labels=["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"], duplicates='drop', include_lowest=True)


            segment_summary = ltv_data_sorted.groupby('RANK', observed=False).agg(
                num_customers=('USER_EMAIL', 'count'),
                total_revenue=('LTV', 'sum'),
                avg_revenue_per_customer=('LTV', 'mean')
            ).reset_index()
            st.write("Segmentación de Clientes por Facturación (LTV en período global):")
            st.dataframe(segment_summary)
            st.markdown("<div class='dashboard-subtext' style='font-size:0.8rem; text-align:center; margin-top:-5px;'>Los clientes se agrupan en cuatro categorías (Alto, Medio-Alto, Medio-Bajo, Bajo) según su gasto total (`LTV`) durante el período global seleccionado. LTV se calcula como la suma de `ORDER_TOTAL_PRICE` por cliente desde `SHOPIFY_ORDERS_LINEITEMS.csv`. La tabla muestra el número de clientes, los ingresos totales y el ingreso promedio por cliente para cada segmento.</div>", unsafe_allow_html=True)
        else:
            st.info("No hay datos de LTV para segmentar clientes.")

        st.markdown("#### 💎 LTV Promedio por Segmento de Frecuencia de Compra")
        # Reutilizar customer_orders_freq de la pestaña de Frecuencia (basado en df_items_filtered)
        if 'customer_orders_freq' not in locals(): # Asegurar que existe, si se accede a esta pestaña directamente
            if not df_items_filtered.empty:
                 customer_orders_freq = df_items_filtered.groupby('USER_EMAIL')['ORDER_GID'].nunique().reset_index(name='NUM_ORDERS')
            else:
                 customer_orders_freq = pd.DataFrame(columns=['USER_EMAIL', 'NUM_ORDERS'])

        if not customer_orders_freq.empty and not ltv_data.empty:
            # Definir segmentos de frecuencia (igual que en la pestaña de Frecuencia)
            bins_ltv_freq = [0, 1, 3, 5, float('inf')]
            labels_ltv_freq = ['1 Pedido (Comprador Único)', '2-3 Pedidos (Ocasional)', '4-5 Pedidos (Frecuente)', '6+ Pedidos (Leal)']
            
            # Crear copia para no afectar el DataFrame original usado en otra pestaña
            customer_orders_freq_copy = customer_orders_freq.copy()
            customer_orders_freq_copy['SEGMENTO_FRECUENCIA'] = pd.cut(customer_orders_freq_copy['NUM_ORDERS'], bins=bins_ltv_freq, labels=labels_ltv_freq, right=True)
            
            # Unir con datos de LTV
            ltv_con_segmento_frecuencia = pd.merge(ltv_data, customer_orders_freq_copy[['USER_EMAIL', 'SEGMENTO_FRECUENCIA']], on='USER_EMAIL', how='left')
            
            if not ltv_con_segmento_frecuencia.empty and 'SEGMENTO_FRECUENCIA' in ltv_con_segmento_frecuencia.columns and ltv_con_segmento_frecuencia['SEGMENTO_FRECUENCIA'].notna().any():
                avg_ltv_by_freq_segment = ltv_con_segmento_frecuencia.groupby('SEGMENTO_FRECUENCIA', observed=False)['LTV'].mean().reset_index()
                avg_ltv_by_freq_segment = avg_ltv_by_freq_segment.sort_values(by='SEGMENTO_FRECUENCIA', key=lambda x: x.map({label: i for i, label in enumerate(labels_ltv_freq)}))


                fig_ltv_by_freq_title = (
                    "LTV Promedio por Segmento de Frecuencia de Compra<br>"
                    "<sub>LTV promedio (gasto total en período global) por segmento de frecuencia de compra (Nº pedidos en mismo período). Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
                )
                fig_ltv_by_freq = px.bar(avg_ltv_by_freq_segment, x='SEGMENTO_FRECUENCIA', y='LTV',
                                         title=fig_ltv_by_freq_title,
                                         labels={'SEGMENTO_FRECUENCIA': 'Segmento por Frecuencia', 'LTV': 'LTV Promedio ($)'})
                fig_ltv_by_freq.update_layout(title_font_size=15, height=400)
                safe_plotly_chart(fig_ltv_by_freq)
            else:
                st.info("No se pudo calcular el LTV por segmento de frecuencia (datos insuficientes o segmentos no aplicables).")
        else:
            st.info("Datos de frecuencia de clientes o LTV no disponibles para este análisis.")


        st.markdown("#### 📈 Tendencias del AOV (Valor Promedio del Pedido)")
        if not df_items_filtered.empty and df_items_filtered['ORDER_GID'].nunique() > 0 : # Asegurarse que hay ordenes
            aov_trend = df_items_filtered.groupby(pd.Grouper(key='ORDER_CREATE_DATE', freq='M')).agg(
                total_revenue=('ORDER_TOTAL_PRICE', 'sum'),
                total_orders=('ORDER_GID', 'nunique')
            ).reset_index()
            # Evitar división por cero
            aov_trend['AOV'] = np.where(aov_trend['total_orders'] > 0, aov_trend['total_revenue'] / aov_trend['total_orders'], np.nan)
            aov_trend = aov_trend.dropna(subset=['AOV']) 
            
            if not aov_trend.empty:
                fig_aov_title = (
                    "Tendencia Mensual del AOV (Valor Promedio del Pedido)<br>"
                    "<sub>AOV mensual = Ingresos totales del mes / Nº de pedidos únicos del mes. Período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
                )
                fig = px.line(aov_trend, x='ORDER_CREATE_DATE', y='AOV', title=fig_aov_title)
                fig.update_xaxes(title_text='Mes')
                fig.update_yaxes(title_text='AOV ($)')
                safe_plotly_chart(fig)
            else:
                st.info("No hay datos suficientes para mostrar la tendencia del AOV.")
        else:
            st.info("No hay órdenes en el período seleccionado para calcular AOV.")
    else:
        st.info("No hay órdenes en el período seleccionado para analizar el valor del cliente.")


# Tab 4: Comportamiento de Compra
with tab_comportamiento:
    st.markdown("### 🛒 Comportamiento de Compra")
    st.markdown("<div class='dashboard-subtext'>Qué compran los clientes y cómo se relacionan los productos.</div>", unsafe_allow_html=True)

    if not df_items_filtered.empty:
        st.markdown("#### 🛍️ Productos Más Comprados por Clientes Recurrentes (en período)")
        customer_order_counts_period = df_items_filtered.groupby('USER_EMAIL')['ORDER_GID'].nunique()
        recurrent_customers_period = customer_order_counts_period[customer_order_counts_period > 1].index
        
        if not recurrent_customers_period.empty:
            df_recurrent_purchases = df_items_filtered[df_items_filtered['USER_EMAIL'].isin(recurrent_customers_period)]
            if not df_recurrent_purchases.empty:
                top_products_recurrent = df_recurrent_purchases.groupby('PRODUCT_NAME')['LINEITEM_QTY'].sum().nlargest(10).reset_index()
                fig_top_recurrent_title = (
                    "Top 10 Productos por Clientes Recurrentes (Cantidad Total de Unidades)<br>"
                    "<sub>Productos más comprados (suma de `LINEITEM_QTY`) por clientes con >1 pedido en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
                )
                fig = px.bar(top_products_recurrent, x='PRODUCT_NAME', y='LINEITEM_QTY', 
                             title=fig_top_recurrent_title)
                safe_plotly_chart(fig)
            else:
                st.info("No hay compras de clientes recurrentes en el período seleccionado.")
        else:
            st.info("No hay clientes recurrentes en el período seleccionado para este análisis.")

        st.markdown("#### 🔗 Cross-sell y Bundles (Pares de Productos Más Comunes en el Mismo Pedido)")
        if 'ORDER_GID' in df_items_filtered.columns and 'PRODUCT_NAME' in df_items_filtered.columns:
            from itertools import combinations
            order_item_counts = df_items_filtered.groupby('ORDER_GID')['PRODUCT_NAME'].nunique()
            multi_item_orders = order_item_counts[order_item_counts > 1].index
            
            if not multi_item_orders.empty:
                df_multi_item_orders = df_items_filtered[df_items_filtered['ORDER_GID'].isin(multi_item_orders)]
                
                frequent_pairs = {}
                for order_gid, group in df_multi_item_orders.groupby('ORDER_GID'):
                    products_in_order = sorted(list(set(group['PRODUCT_NAME']))) 
                    if len(products_in_order) >= 2:
                        for pair in combinations(products_in_order, 2):
                            frequent_pairs[pair] = frequent_pairs.get(pair, 0) + 1
                
                if frequent_pairs:
                    top_pairs_df = pd.DataFrame(list(frequent_pairs.items()), columns=['Product_Pair', 'Frequency'])
                    top_pairs_df = top_pairs_df.sort_values('Frequency', ascending=False).head(10)
                    top_pairs_df['Product_Pair_Display'] = top_pairs_df['Product_Pair'].apply(lambda x: f"{x[0]} & {x[1]}")
                    
                    fig_pairs_title = (
                        "Top 10 Pares de Productos Comprados Juntos (Frecuencia en Pedidos)<br>"
                        "<sub>Pares de productos distintos que más frecuentemente aparecen juntos en el mismo pedido (`ORDER_GID`) en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
                    )
                    fig_pairs = px.bar(top_pairs_df, x='Product_Pair_Display', y='Frequency',
                                     title=fig_pairs_title)
                    fig_pairs.update_xaxes(title_text="Par de Productos")
                    safe_plotly_chart(fig_pairs)
                else:
                    st.info("No se encontraron pares de productos comprados juntos con frecuencia en el período.")
            else:
                st.info("No hay pedidos con múltiples productos diferentes en el período seleccionado para análisis de cross-sell.")
        else:
            if "Cross-sell y Bundles (columnas 'ORDER_GID' o 'PRODUCT_NAME' faltantes, o análisis complejo)" not in missing_data_points:
                missing_data_points.append("Cross-sell y Bundles (columnas 'ORDER_GID' o 'PRODUCT_NAME' faltantes, o análisis complejo)")
            st.warning("Datos insuficientes para análisis de cross-sell.")

        st.markdown("#### ☀️ Estacionalidad de Compras")
        # Create a copy to avoid SettingWithCopyWarning
        df_items_filtered_seasonal = df_items_filtered.copy()
        df_items_filtered_seasonal['MONTH'] = df_items_filtered_seasonal['ORDER_CREATE_DATE'].dt.strftime('%B')
        df_items_filtered_seasonal['DAY_OF_WEEK'] = df_items_filtered_seasonal['ORDER_CREATE_DATE'].dt.day_name()
        
        months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        sales_by_month = df_items_filtered_seasonal.groupby('MONTH')['ORDER_TOTAL_PRICE'].sum().reindex(months_order).reset_index()
        sales_by_day = df_items_filtered_seasonal.groupby('DAY_OF_WEEK')['ORDER_TOTAL_PRICE'].sum().reindex(days_order).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            if not sales_by_month.empty:
                fig_month_title = (
                    "Estacionalidad: Ingresos Totales por Mes<br>"
                    "<sub>Suma de `ORDER_TOTAL_PRICE` por mes, en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
                )
                fig_month = px.bar(sales_by_month, x='MONTH', y='ORDER_TOTAL_PRICE', title=fig_month_title)
                safe_plotly_chart(fig_month)
            else:
                st.info("No hay datos de ventas mensuales en el período.")
        with col2:
            if not sales_by_day.empty:
                fig_day_title = (
                    "Estacionalidad: Ingresos Totales por Día de la Semana<br>"
                    "<sub>Suma de `ORDER_TOTAL_PRICE` por día de la semana, en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
                )
                fig_day = px.bar(sales_by_day, x='DAY_OF_WEEK', y='ORDER_TOTAL_PRICE', title=fig_day_title)
                safe_plotly_chart(fig_day)
            else:
                st.info("No hay datos de ventas por día de la semana en el período.")
        if "Estacionalidad por fechas especiales (requiere configuración manual de fechas)" not in missing_data_points:
            missing_data_points.append("Estacionalidad por fechas especiales (requiere configuración manual de fechas)")
    else:
        st.info("No hay órdenes en el período seleccionado para analizar el comportamiento de compra.")


# Tab 5: Traffic Analytics
with tab_trafico:
    st.markdown("### 📈 Traffic Analytics")
    st.markdown("<div class='dashboard-subtext'>Análisis del tráfico web y su efectividad.</div>", unsafe_allow_html=True)

    if not df_sessions_filtered.empty:
        st.markdown("#### 📉 Tasa de Rebote (Bounce Rate)")
        if 'BOUNCE_RATE' in df_sessions_filtered.columns:
            avg_bounce_rate = df_sessions_filtered['BOUNCE_RATE'].mean() * 100
            st.metric("Tasa de Rebote Promedio", f"{avg_bounce_rate:.2f}%" if pd.notna(avg_bounce_rate) else "N/A", 
                      help="Porcentaje promedio de sesiones que terminaron después de ver una sola página (rebote). Se calcula como la media de la columna `BOUNCE_RATE` (donde 1 es un rebote) del archivo `GA_SESSIONS.csv` para el período global seleccionado, multiplicado por 100.")
        else:
            st.warning("Columna 'BOUNCE_RATE' no encontrada en `GA_SESSIONS.csv`. No se puede calcular la tasa de rebote.")

        st.markdown("#### ⏳ Duración Promedio de Sesión")
        if 'AVG_SESSION_DURATION' in df_sessions_filtered.columns:
            avg_duration = df_sessions_filtered['AVG_SESSION_DURATION'].mean()
            st.metric("Duración Promedio de Sesión (segundos)", f"{avg_duration:.2f}s" if pd.notna(avg_duration) else "N/A", 
                      help="Tiempo promedio (en segundos) que los usuarios pasan en el sitio por sesión. Se calcula como la media de la columna `AVG_SESSION_DURATION` del archivo `GA_SESSIONS.csv` para el período global seleccionado.")
        else:
            st.warning("Columna 'AVG_SESSION_DURATION' no encontrada en `GA_SESSIONS.csv`. No se puede calcular la duración promedio.")
    else:
        st.info("No hay datos de sesiones en el período seleccionado.")

    if not df_ads_filtered.empty:
        st.markdown("#### 👉 CTR (Click-Through Rate) Promedio de Anuncios")
        if 'Ads_CTR' in df_ads_filtered.columns:
            avg_ctr = df_ads_filtered['Ads_CTR'].mean() * 100
            st.metric("CTR Promedio de Anuncios", f"{avg_ctr:.2f}%" if pd.notna(avg_ctr) else "N/A", 
                      help="Tasa de Clics (Click-Through Rate) promedio de las campañas de anuncios. Se calcula como la media de la columna `Ads_CTR` (definida como Clics / Impresiones) del archivo `GA_ADS_CAMPAIGNS.csv` para el período global seleccionado, multiplicado por 100 para expresar como porcentaje.")
        else:
            st.warning("Columna 'Ads_CTR' no encontrada en `GA_ADS_CAMPAIGNS.csv`.")
            if "CTR de Anuncios (columna 'Ads_CTR' no encontrada)" not in missing_data_points:
                missing_data_points.append("CTR de Anuncios (columna 'Ads_CTR' no encontrada)")
    else:
        st.info("No hay datos de anuncios en el período seleccionado para CTR.")

    st.markdown("#### 🛒 Tasa de Abandono de Carrito (CAR)")
    st.warning("El cálculo de la Tasa de Abandono de Carrito (CAR) requiere datos específicos sobre eventos de 'añadir al carrito' y 'compras completadas' que no están disponibles o no se han configurado para su cálculo con los CSVs actuales. Formula: (Carritos Creados - Transacciones) / Carritos Creados.")
    st.markdown("<div class='dashboard-subtext' style='font-size:0.8rem; text-align:center; margin-top:5px;'>Idealmente, se usarían datos de `ADD_TO_CART_EVENTS` y `PURCHASE_EVENTS` del dataset de sesiones o similar.</div>", unsafe_allow_html=True)

# Tab 6: Costo de Adquisición
with tab_costo:
    st.markdown("### 💸 Costo de Adquisición")
    st.markdown("<div class='dashboard-subtext'>Análisis de cuánto cuesta adquirir nuevos clientes.</div>", unsafe_allow_html=True)

    if not df_ads_filtered.empty:
        st.markdown("#### 📊 CAC (Costo de Adquisición de Cliente) por Canal (Curva de Evolución)")
        if 'Ads_cost' in df_ads_filtered.columns and 'Total_users' in df_ads_filtered.columns and 'Primary_channel_group' in df_ads_filtered.columns:
            df_ads_filtered_cac = df_ads_filtered.copy()
            # Asegurar que Total_users no es cero para evitar división por cero
            df_ads_filtered_cac['CAC'] = np.where(df_ads_filtered_cac['Total_users'] > 0, df_ads_filtered_cac['Ads_cost'] / df_ads_filtered_cac['Total_users'], np.nan)
            df_ads_filtered_cac['CAC'] = df_ads_filtered_cac['CAC'].replace([np.inf, -np.inf], np.nan) 

            cac_by_channel_time = df_ads_filtered_cac.groupby([pd.Grouper(key='Date', freq='M'), 'Primary_channel_group'])['CAC'].mean().reset_index()
            cac_by_channel_time = cac_by_channel_time.dropna(subset=['CAC'])

            if not cac_by_channel_time.empty:
                fig_cac_title = (
                    "Evolución Mensual del CAC por Canal de Anuncios<br>"
                    "<sub>CAC mensual por canal = Costo total de anuncios (`Ads_cost`) del mes / Nuevos usuarios (`Total_users`) del mes. Fuente: `GA_ADS_CAMPAIGNS.csv`. Período global.</sub>"
                )
                fig = px.line(cac_by_channel_time, x='Date', y='CAC', color='Primary_channel_group',
                              title=fig_cac_title,
                              labels={'Date': 'Mes', 'CAC': 'CAC Promedio ($)', 'Primary_channel_group': 'Canal Primario'})
                safe_plotly_chart(fig)
            else:
                st.info("No hay suficientes datos para mostrar la evolución del CAC por canal.")
        else:
            st.warning("Columnas necesarias ('Ads_cost', 'Total_users', 'Primary_channel_group') no encontradas en `GA_ADS_CAMPAIGNS.csv` para calcular CAC.")
            if "CAC por Canal (columnas requeridas faltantes en datos de anuncios)" not in missing_data_points:
                missing_data_points.append("CAC por Canal (columnas requeridas faltantes en datos de anuncios)")
    else:
        st.info("No hay datos de anuncios en el período seleccionado para analizar costos de adquisición.")

    st.markdown("#### 🎯 Costo por Lead (CPL)")
    st.warning("El cálculo del Costo por Lead (CPL) requiere datos sobre la generación de leads (ej. 'LEADS_GENERATED') y el costo asociado a esa generación, que no están disponibles en los CSVs actuales. Formula: Costo de Campaña / Número de Leads Generados.")
    st.markdown("<div class='dashboard-subtext' style='font-size:0.8rem; text-align:center; margin-top:5px;'>Se necesitaría la columna `LEADS_GENERATED` en `GA_ADS_CAMPAIGNS.csv` o un dataset similar.</div>", unsafe_allow_html=True)

# Tab 7: NPS
with tab_nps:
    st.markdown("### 😊 NPS (Net Promoter Score)")
    st.markdown("<div class='dashboard-subtext'>Midiendo la lealtad y satisfacción del cliente.</div>", unsafe_allow_html=True)
    st.warning("""
    Los datos de Net Promoter Score (NPS) generalmente se recopilan a través de encuestas directas a los clientes
    y no suelen estar presentes en los conjuntos de datos transaccionales o de análisis web estándar.
    """)
    st.markdown("""
    <div class='dashboard-subtext' style='font-size:0.8rem; text-align:left; margin-top:5px;'>
    Para mostrar análisis de NPS aquí, necesitarías cargar un archivo CSV que contenga:
    <ul>
        <li>Identificador de cliente (ej. <code>USER_EMAIL</code>)</li>
        <li>Puntuación NPS (0-10)</li>
        <li>Fecha de la encuesta</li>
        <li>Opcionalmente, el canal por el cual se obtuvo la respuesta.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# Footer mejorado
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; padding: 1rem; color: #666;'>
        <p style='margin-bottom: 0.5rem;'>Customer Analytics Dashboard</p>
        <p style='font-size: 0.9rem; margin: 0;'>Última actualización de datos en dashboard: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    """,
    unsafe_allow_html=True
) 

# Sección de Próximas Métricas y Mejoras (reemplaza "Notas sobre Datos Faltantes")
# Lista de puntos específicos a mostrar con el nuevo tono
planned_metrics_messages = [
    "LTV por canal de adquisición",
    "Tasa de Rebote (Bounce Rate)",
    "Duración Promedio de Sesión",
    "Tasa de Abandono de Carrito (CAR)",
    "Costo por Lead (CPL)",
    "Net Promoter Score (NPS)"
]

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Próximas Métricas y Mejoras")
st.sidebar.info("Estamos trabajando continuamente para enriquecer este dashboard. Algunas métricas y análisis adicionales que estarán disponibles pronto incluyen:")
for message in planned_metrics_messages:
    st.sidebar.markdown(f"- {message}")