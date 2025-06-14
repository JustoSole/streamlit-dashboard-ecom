# Standard library imports
import json
import logging
import warnings
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import time

# Third-party imports
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

# BigQuery imports
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core import exceptions

# BigQuery configuration - SOLO BigQuery, sin fallback
try:
    import bigquery_config as bq_config
except ImportError:
    st.error("❌ Error crítico: No se encontró el archivo bigquery_config.py")
    st.error("Por favor, asegúrate de que el archivo bigquery_config.py existe en el directorio del dashboard.")
    st.stop()

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# —————————————————————————————————————————
# CONFIGURACIÓN Y CONSTANTES
# —————————————————————————————————————————

# Configuración de la página
st.set_page_config(
    layout="wide",
    page_title="Customer Analytics Dashboard",
    page_icon="👥",
    initial_sidebar_state="expanded"
)

# Configuración del Dashboard
class DashboardConfig:
    """Configuración centralizada del dashboard."""
    
    # Configuración de logging
    LOG_FILE: str = "dashboard.log"
    
    # BigQuery Configuration - SOLO BigQuery
    USE_BIGQUERY: bool = True  # Siempre True - Solo BigQuery
    BIGQUERY_PROJECT_ID: str = bq_config.BIGQUERY_PROJECT_ID
    BIGQUERY_DATASET: str = bq_config.BIGQUERY_DATASET
    SERVICE_ACCOUNT_FILE: str = bq_config.SERVICE_ACCOUNT_FILE
    
    # BigQuery table names
    BIGQUERY_TABLES: Dict[str, str] = bq_config.BIGQUERY_TABLES
    
    # Cache settings
    CACHE_TTL: int = bq_config.CACHE_TTL
    
    # Query limits
    MAX_QUERY_ROWS: int = getattr(bq_config, 'MAX_QUERY_ROWS', 100000)
    
    # Configuraciones de visualización
    DEFAULT_CHART_HEIGHT: int = 400
    DEFAULT_COLUMNS: int = 3
    MAX_RESULTS_DISPLAY: int = 15
    
    # Configuraciones de segmentación
    FREQUENCY_BINS: List[Union[int, float]] = [0, 1, 3, 5, float('inf')]
    FREQUENCY_LABELS: List[str] = [
        '1 Pedido (Comprador Único)', 
        '2-3 Pedidos (Ocasional)', 
        '4-5 Pedidos (Frecuente)', 
        '6+ Pedidos (Leal)'
    ]
    
    ORDER_GROUP_BINS: List[Union[int, float]] = [0, 1, 2, 3, 5, 10, float('inf')]
    ORDER_GROUP_LABELS: List[str] = [
        '1 Pedido', '2 Pedidos', '3 Pedidos', 
        '4-5 Pedidos', '6-10 Pedidos', '11+ Pedidos'
    ]
    
    # Ordenamiento temporal
    LEVEL_ORDER: List[str] = ['Principiante', 'Intermedio', 'Avanzado', 'Sin Especificar']
    MONTHS_ORDER: List[str] = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    DAYS_ORDER: List[str] = [
        "Monday", "Tuesday", "Wednesday", "Thursday", 
        "Friday", "Saturday", "Sunday"
    ]
    
    # Paletas de colores para consistencia - Usando azul uniforme para mejor UX
    COLOR_PALETTES: Dict[str, Union[str, List[str]]] = {
        'sport_recurrence': 'Blues',
        'level_recurrence': 'Blues',
        'padel': ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78'],
        'pickleball': ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'],
        'otros': ['#1e40af', '#3b82f6', '#60a5fa', '#93c5fd'],
        'comparative': ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
    }
    
    # Configuraciones de productos
    PRODUCT_CATEGORIES: Dict[str, List[str]] = {
        'Paletas/Raquetas': ['paddle', 'paleta', 'racket', 'raqueta'],
        'Pelotas': ['ball', 'pelota', 'bola'],
        'Ropa': ['shirt', 'shorts', 'polo', 'camiseta', 'pantalón', 'short', 'apparel', 'clothing'],
        'Calzado': ['shoe', 'sneaker', 'zapato', 'calzado'],
        'Bolsas/Fundas': ['bag', 'case', 'cover', 'bolsa', 'funda', 'mochila'],
        'Accesorios': ['grip', 'overgrip', 'string', 'cuerda', 'cordaje', 'wristband', 'headband'],
        'Equipos de Cancha': ['net', 'red', 'court', 'cancha']
    }
    
    SPORT_KEYWORDS: Dict[str, List[str]] = {
        'Pickleball': ['pickleball', 'pickle ball', 'pickball'],
        'Pádel': ['padel', 'pádel', 'paddle tennis', 'pop tennis']
    }
    
    LEVEL_KEYWORDS: Dict[str, List[str]] = {
        'Principiante': ['beginner', 'starter', 'principiante', 'inicial', 'básico', 'basic'],
        'Intermedio': ['intermediate', 'intermedio', 'medio', 'recreational', 'recreativo', 'club', 'sport', 'game'],
        'Avanzado': ['advanced', 'avanzado', 'pro', 'professional', 'profesional', 'elite', 'competition', 'competición', 'tour', 'championship', 'master', 'premium']
    }

# Instance of configuration
config = DashboardConfig()

# Global state management
class GlobalState:
    """Gestión del estado global del dashboard."""
    
    def __init__(self):
        self.missing_data_points: List[str] = []
        self.error_messages: List[str] = []
        
    def add_missing_data_point(self, point: str) -> None:
        """Agrega un punto de datos faltante."""
        if point not in self.missing_data_points:
            self.missing_data_points.append(point)
            logger.warning(f"Missing data point identified: {point}")
    
    def add_error_message(self, message: str) -> None:
        """Agrega un mensaje de error."""
        self.error_messages.append(message)
        logger.error(message)
    
    def clear_errors(self) -> None:
        """Limpia los mensajes de error."""
        self.error_messages.clear()

# Global state instance
state = GlobalState()

# Backward compatibility - DEPRECATED (mantenido para compatibilidad)
missing_data_points = state.missing_data_points
FREQUENCY_BINS = config.FREQUENCY_BINS
FREQUENCY_LABELS = config.FREQUENCY_LABELS
ORDER_GROUP_BINS = config.ORDER_GROUP_BINS
ORDER_GROUP_LABELS = config.ORDER_GROUP_LABELS
LEVEL_ORDER = config.LEVEL_ORDER
MONTHS_ORDER = config.MONTHS_ORDER
DAYS_ORDER = config.DAYS_ORDER
COLOR_PALETTES = config.COLOR_PALETTES

# —————————————————————————————————————————
# UTILIDADES Y HELPERS
# —————————————————————————————————————————

def log_function_call(func_name: str, **kwargs) -> None:
    """Registra llamadas a funciones importantes."""
    logger.info(f"Executing {func_name} with parameters: {kwargs}")

def handle_dataframe_error(df: pd.DataFrame, context: str) -> bool:
    """Maneja errores comunes de DataFrames."""
    if df is None:
        state.add_error_message(f"DataFrame is None in context: {context}")
        return False
    if df.empty:
        logger.warning(f"Empty DataFrame in context: {context}")
        return False
    return True

def validate_required_columns(df: pd.DataFrame, required_cols: List[str], context: str) -> bool:
    """Valida que un DataFrame tenga las columnas requeridas."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns {missing_cols} in context: {context}"
        state.add_error_message(error_msg)
        return False
    return True

def safe_division(numerator: float, denominator: float, default: float = None) -> float:
    """Realiza división segura evitando división por cero. Retorna None por defecto para evitar inventar datos."""
    try:
        if denominator == 0:
            return default
        if pd.isna(numerator) or pd.isna(denominator):
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def format_currency(amount: float) -> str:
    """Formatea cantidades monetarias."""
    try:
        return f"${amount:,.2f}" if pd.notna(amount) else "N/A"
    except (TypeError, ValueError):
        return "N/A"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Formatea porcentajes."""
    try:
        return f"{value:.{decimals}f}%" if pd.notna(value) else "N/A"
    except (TypeError, ValueError):
        return "N/A"

def normalize_datetime(dt: Union[datetime, pd.Timestamp, str]) -> pd.Timestamp:
    """Normaliza fechas para evitar problemas de zona horaria en comparaciones."""
    try:
        if dt is None:
            return None
        
        # Convertir a pandas Timestamp
        if isinstance(dt, str):
            ts = pd.to_datetime(dt)
        elif isinstance(dt, datetime):
            ts = pd.Timestamp(dt)
        elif isinstance(dt, pd.Timestamp):
            ts = dt
        else:
            ts = pd.to_datetime(dt)
        
        # Remover zona horaria si existe para comparaciones consistentes
        if hasattr(ts, 'tz') and ts.tz is not None:
            ts = ts.tz_localize(None)
            
        return ts
    except Exception as e:
        logger.warning(f"Error normalizing datetime {dt}: {e}")
        return None

# —————————————————————————————————————————
# FUNCIONES DE CONEXIÓN A BIGQUERY - OPTIMIZADAS
# —————————————————————————————————————————

@st.cache_resource
def get_bigquery_client() -> Optional[bigquery.Client]:
    """Inicializa y retorna el cliente de BigQuery con cache."""
    try:
        log_function_call("get_bigquery_client")
        
        if not Path(config.SERVICE_ACCOUNT_FILE).exists():
            error_msg = f"Service account file not found: {config.SERVICE_ACCOUNT_FILE}"
            state.add_error_message(error_msg)
            return None
        
        # Cargar credenciales
        with open(config.SERVICE_ACCOUNT_FILE) as source:
            info = json.load(source)
        
        # Usar los scopes configurados (incluye Drive para tablas vinculadas a Google Sheets)
        scopes = bq_config.BIGQUERY_SCOPES
        
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=scopes
        )
        
        # Inicializar cliente con configuración específica
        client = bigquery.Client(
            credentials=credentials,
            project=config.BIGQUERY_PROJECT_ID,
            location="US"  # Especificar ubicación para evitar errores
        )
        
        # Verificar conexión con query simple
        test_query = "SELECT 1 as test_connection"
        client.query(test_query).result()
        logger.info("✅ BigQuery client initialized and tested successfully")
        return client
        
    except FileNotFoundError:
        error_msg = f"❌ Service account file not found: {config.SERVICE_ACCOUNT_FILE}"
        state.add_error_message(error_msg)
        logger.error(error_msg)
        return None
    except json.JSONDecodeError:
        error_msg = f"❌ Invalid JSON in service account file: {config.SERVICE_ACCOUNT_FILE}"
        state.add_error_message(error_msg)
        logger.error(error_msg)
        return None
    except exceptions.Forbidden as e:
        error_msg = f"❌ BigQuery access denied - Check service account permissions: {e.message}"
        state.add_error_message(error_msg)
        logger.error(error_msg)
        
        # Sugerencias específicas para el error 403
        st.error("🔒 **Error de Permisos de BigQuery**")
        st.error("**Verifica que el service account tenga estos roles:**")
        st.error("• BigQuery Data Viewer")
        st.error("• BigQuery Job User") 
        st.error("• BigQuery Read Session User (opcional)")
        st.error(f"**Service Account:** `bigquery-streamlit@racket-central-gcp.iam.gserviceaccount.com`")
        return None
    except Exception as e:
        error_msg = f"❌ Error initializing BigQuery client: {str(e)}"
        state.add_error_message(error_msg)
        logger.error(error_msg)
        return None

def get_default_date_range() -> Tuple[str, str]:
    """Obtiene el rango de fechas por defecto (YTD - Year to Date) para inicializar el dashboard."""
    end_date = datetime.now()
    start_date = datetime(end_date.year, 1, 1)  # 1 de enero del año actual (YTD)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def load_table_from_bigquery_optimized(client: bigquery.Client, table_name: str, 
                                      date_column: Optional[str] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None,
                                      columns: Optional[List[str]] = None,
                                      limit: Optional[int] = None,
                                      max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Carga una tabla desde BigQuery con optimizaciones de rendimiento y retry logic."""
    
    for attempt in range(max_retries):
        try:
            log_function_call("load_table_from_bigquery_optimized", table=table_name, attempt=attempt+1)
            
            # Usar rango de fechas por defecto si no se proporciona
            if date_column and not start_date:
                start_date, end_date = get_default_date_range()
                logger.info(f"Using default date range: {start_date} to {end_date}")
            
            # Seleccionar columnas específicas en lugar de *
            if columns:
                columns_str = ", ".join(columns)
            else:
                columns_str = "*"
            
            # Construir query optimizada
            if '.' in table_name: # Handle fully qualified table names
                full_table_name = f"`{table_name}`"
            else:
                full_table_name = f"`{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{table_name}`"
            
            query = f"SELECT {columns_str} FROM {full_table_name}"
            
            # Agregar filtros de fecha DIRECTAMENTE en la query (optimización principal)
            conditions = []
            if date_column and start_date and end_date:
                conditions.append(f"{date_column} >= '{start_date}'")
                conditions.append(f"{date_column} <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Agregar ordenamiento por fecha si existe
            if date_column:
                query += f" ORDER BY {date_column} DESC"
            
            # Aplicar límite por defecto para evitar consultas masivas
            effective_limit = limit or config.MAX_QUERY_ROWS
            query += f" LIMIT {effective_limit}"
            
            logger.info(f"Executing optimized BigQuery: {query}")
            
            # Ejecutar query optimizada
            df = client.query(query).to_dataframe()
            
            if df.empty:
                logger.warning(f"Empty result from BigQuery table: {table_name}")
            else:
                logger.info(f"Loaded {len(df)} rows from BigQuery table: {table_name} (optimized)")
            
            return df
            
        except exceptions.NotFound:
            error_msg = f"BigQuery table not found: {table_name}"
            state.add_error_message(error_msg)
            return None
        except exceptions.Forbidden as e:
            error_msg = f"Access denied to BigQuery table {table_name}: {e.message}"
            state.add_error_message(error_msg)
            return None
        except Exception as e:
            error_str = str(e)
            
            # Manejo específico para errores de Google Sheets sobrecargado
            if "Resources exceeded" in error_str and "Google Sheets service overloaded" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Espera incremental: 2, 4, 6 segundos
                    logger.warning(f"Google Sheets overloaded for {table_name}, retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"⚠️ {table_name}: Google Sheets sobrecargado después de {max_retries} intentos. Tabla omitida temporalmente."
                    state.add_error_message(error_msg)
                    logger.error(error_msg)
                    return None
            else:
                error_msg = f"Error loading table {table_name} from BigQuery: {error_str}"
                state.add_error_message(error_msg)
                return None
    
    return None

def load_table_from_bigquery(client: bigquery.Client, table_name: str, 
                             date_column: Optional[str] = None,
                             start_date: Optional[pd.Timestamp] = None,
                             end_date: Optional[pd.Timestamp] = None,
                             limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Carga una tabla desde BigQuery con filtros opcionales - FUNCIÓN LEGACY."""
    # Convertir fechas de pandas a string si se proporcionan
    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None
    
    # Llamar a la función optimizada
    return load_table_from_bigquery_optimized(
        client, table_name, date_column, start_date_str, end_date_str, limit=limit
    )

def execute_bigquery_query(client: bigquery.Client, query: str) -> Optional[pd.DataFrame]:
    """Ejecuta una query personalizada en BigQuery."""
    try:
        log_function_call("execute_bigquery_query")
        logger.info(f"Executing custom BigQuery: {query[:200]}...")
        
        result = client.query(query).to_dataframe()
        logger.info(f"Custom query returned {len(result)} rows")
        return result
        
    except Exception as e:
        error_msg = f"Error executing custom BigQuery: {str(e)}"
        state.add_error_message(error_msg)
        return None

# —————————————————————————————————————————
# ESTILOS CSS
# —————————————————————————————————————————

def apply_custom_css() -> None:
    """Aplica todos los estilos CSS personalizados."""
    log_function_call("apply_custom_css")
    st.markdown("""
<style>
    /* Main dashboard font */
    body, .stApp {
        font-family: 'Source Sans Pro', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Headers styling */
    h1, h2, h3 {
        color: #005A5A;
        font-weight: 600;
    }
    
    /* Improve st.metric look and feel */
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #E0E0E0;
        transition: box-shadow 0.3s ease-in-out;
    }
    .stMetric:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }

    /* Style the help text within st.metric */
    .stMetric [data-testid="stMetricHelp"] p {
        font-style: italic;
        font-size: 0.82rem;
        color: #555E67;
        line-height: 1.3;
        margin-top: 5px;
    }
    .stMetric label {
        font-weight: 500;
        color: #4B5563;
    }
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 2.1rem;
        font-weight: 600;
        color: #008080;
    }
     .stMetric div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* General descriptive text under tabs or section titles */
    .dashboard-subtext {
        font-size: 0.9rem;
        color: #4A5568;
        margin-bottom: 1.2rem;
        font-style: normal;
        line-height: 1.5;
    }
    
    /* Style st.tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #D1D5DB;
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px 6px 0px 0px;
        padding: 0px 16px;
        margin-bottom: -1px;
        border: 1px solid transparent;
        font-weight: 500;
        color: #4B5563;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E5E7EB;
        color: #1F2937;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        color: #008080;
        border-color: #D1D5DB #D1D5DB #FFFFFF;
        font-weight: 600;
    }

    /* Warnings and infos */
    .stAlert {
        border-radius: 6px;
    }
    .stAlert[data-baseweb="alert"] > div:first-child {
        padding-top: 10px !important; 
    }
    .stAlert[data-baseweb="alert"] p {
        font-size: 0.88rem;
    }
    
    /* Sidebar styling for missing data points */
    [data-testid="stSidebar"] .stAlert[data-baseweb="alert"] p {
        font-size: 0.8rem;
    }
    [data-testid="stSidebar"] h3 {
        color: #006A6A;
        font-size: 1.1rem;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 0.95rem;
    }

    /* General st.info styling */
    .stInfo {
        background-color: #E0F2FE !important;
        border: 1px solid #7DD3FC !important;
        color: #0C5475 !important;
        border-radius: 6px !important;
    }
    .stInfo p {
         color: #0C5475 !important;
         font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# —————————————————————————————————————————
# FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# —————————————————————————————————————————

def load_csv_safely(file_path: str, required_columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Carga un archivo CSV de forma segura con validaciones."""
    try:
        if not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            state.add_error_message(error_msg)
            return None
        
        logger.info(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            logger.warning(f"Empty CSV file loaded: {file_path}")
            return df
        
        if required_columns and not validate_required_columns(df, required_columns, f"CSV load: {file_path}"):
            return None
        
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
        
    except FileNotFoundError:
        error_msg = f"CSV file not found: {file_path}"
        state.add_error_message(error_msg)
        return None
    except pd.errors.EmptyDataError:
        error_msg = f"Empty CSV file: {file_path}"
        state.add_error_message(error_msg)
        return None
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing CSV {file_path}: {str(e)}"
        state.add_error_message(error_msg)
        return None
    except Exception as e:
        error_msg = f"Unexpected error loading {file_path}: {str(e)}"
        state.add_error_message(error_msg)
        return None

def process_shopify_data(df_items: pd.DataFrame, df_customers: Optional[pd.DataFrame], df_orders: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Procesa y limpia los datos de Shopify, fusionando con datos de clientes y órdenes."""
    if not handle_dataframe_error(df_items, "process_shopify_data"):
        return pd.DataFrame()
    
    try:
        df_processed = df_items.copy()

        # Handle upstream column name changes from BigQuery
        if 'ORDER_ID' in df_processed.columns:
            df_processed.rename(columns={'ORDER_ID': 'ORDER_GID'}, inplace=True)

        # Merge with orders data to get specific shipping info (city, state, lat, lon)
        if df_orders is not None and not df_orders.empty:
            order_cols = ['ORDER_ID', 'SHIPPING_CITY', 'SHIPPING_STATE', 'SHIPPING_COUNTRY', 'SHIPPING_LATITUDE', 'SHIPPING_LONGITUDE']
            
            # Use only existing columns to avoid errors
            cols_to_use = [col for col in order_cols if col in df_orders.columns]

            if 'ORDER_ID' in cols_to_use:
                # Create a mapping from ORDER_GID to ORDER_ID for the merge
                df_orders_for_merge = df_orders[cols_to_use].copy()
                df_orders_for_merge = df_orders_for_merge.rename(columns={'ORDER_ID': 'ORDER_GID'})
                df_processed = pd.merge(df_processed, df_orders_for_merge, on='ORDER_GID', how='left')
            else:
                 logger.warning("Cannot merge order shipping data, ORDER_ID is missing from df_orders.")
        else:
            logger.warning("Orders data (df_orders) not provided. City/State/Coordinates will be missing.")

        # RENAME COLUMNS to match dashboard expectations
        rename_map = {
            'ORDER_GID': 'ORDER_GID',  # Already correct
            'LINEITEM_QTY': 'LINEITEM_QTY',  # Already correct
            'PRICE': 'PRICE'  # Already correct
        }
        df_processed = df_processed.rename(columns=rename_map)

        # Calculate line item total
        df_processed['DISTRIBUTED_SALE'] = df_processed['PRICE'] * df_processed['LINEITEM_QTY']

        # Calculate ORDER_TOTAL_PRICE by summing line items for each order
        order_total_price_df = df_processed.groupby('ORDER_GID')['DISTRIBUTED_SALE'].sum().reset_index()
        order_total_price_df.rename(columns={'DISTRIBUTED_SALE': 'ORDER_TOTAL_PRICE'}, inplace=True)
        
        # Merge the order-level total price back to the item-level data
        df_processed = pd.merge(df_processed, order_total_price_df, on='ORDER_GID', how='left')

        # Merge with customer data to get email and shipping info
        if df_customers is not None and not df_customers.empty:
            # Seleccionar y renombrar columnas de clientes
            customer_cols = ['CUSTOMER_ID', 'CUSTOMER_EMAIL', 'SHIPPING_COUNTRY', 'SHIPPING_STATE', 'SHIPPING_CITY']
            
            # Verificar que las columnas existen en df_customers
            missing_cols = [col for col in customer_cols if col not in df_customers.columns]
            if missing_cols:
                logger.warning(f"Las siguientes columnas de cliente faltan y no se incluirán: {missing_cols}")
                customer_cols = [col for col in customer_cols if col in df_customers.columns]

            if 'CUSTOMER_ID' in customer_cols:
                customers_to_merge = df_customers[customer_cols].copy()
                customers_to_merge = customers_to_merge.rename(columns={'CUSTOMER_EMAIL': 'USER_EMAIL'})
                
                df_processed = pd.merge(df_processed, customers_to_merge, on='CUSTOMER_ID', how='left')
            
                # Handle cases where customer email is still missing after merge
                if 'USER_EMAIL' in df_processed.columns and df_processed['USER_EMAIL'].isnull().any():
                    missing_count = df_processed['USER_EMAIL'].isnull().sum()
                    logger.warning(f"{missing_count} line items have no matching customer. These will be excluded from customer-specific analysis.")
                    df_processed.dropna(subset=['USER_EMAIL'], inplace=True) # Opcional: decidir si eliminar o no
            else:
                 logger.error("CUSTOMER_ID no está en el dataframe de clientes. No se puede hacer el merge.")
                 df_processed['USER_EMAIL'] = 'error_no_customer_id'

        else:
            logger.error("Customer data (df_customers) not provided or is empty. Cannot merge customer info.")
            # Crear columnas placeholder para evitar KeyErrors posteriores si no se hizo antes
            df_processed['USER_EMAIL'] = 'error_no_customer_data'
            if 'SHIPPING_COUNTRY' not in df_processed.columns:
                 df_processed['SHIPPING_COUNTRY'] = 'Unknown'
            if 'SHIPPING_STATE' not in df_processed.columns:
                 df_processed['SHIPPING_STATE'] = 'Unknown'
            if 'SHIPPING_CITY' not in df_processed.columns:
                 df_processed['SHIPPING_CITY'] = 'Unknown'
        
        # Procesamiento de fechas
        if 'ORDER_CREATE_DATE' in df_processed.columns:
            df_processed['ORDER_CREATE_DATE'] = pd.to_datetime(
                df_processed['ORDER_CREATE_DATE']
            ).dt.tz_localize(None)
        
        # Limpieza de datos de usuario
        if 'USER_EMAIL' in df_processed.columns:
            df_processed['USER_EMAIL'] = (
                df_processed['USER_EMAIL']
                .astype(str)
                .fillna('unknown_email')
                .str.lower()
                .str.strip()
            )
        
        # Limpieza de nombres de productos
        if 'PRODUCT_NAME' in df_processed.columns:
            df_processed['PRODUCT_NAME'] = (
                df_processed['PRODUCT_NAME']
                .astype(str)
                .fillna('Unknown Product')
                .str.strip()
            )
        
        # Limpieza de datos numéricos - NO INVENTAR DATOS CON fillna(0)
        numeric_columns = ['ORDER_TOTAL_PRICE', 'DISTRIBUTED_SALE', 'LINEITEM_QTY', 'PRICE']
        for col in numeric_columns:
            if col in df_processed.columns:
                # Solo convertir a numérico, NO llenar con ceros inventados
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                # Reportar si hay valores faltantes
                null_count = df_processed[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Columna {col}: {null_count} valores nulos encontrados (no se rellenan con datos inventados)")
                    state.add_missing_data_point(f"Valores nulos en columna {col} ({null_count} registros)")
        
        # Crear columna PRICE para compatibilidad (usar DISTRIBUTED_SALE como alternativa)
        if 'DISTRIBUTED_SALE' in df_processed.columns and 'PRICE' not in df_processed.columns:
            df_processed['PRICE'] = df_processed['DISTRIBUTED_SALE']
            logger.info("Created PRICE column from DISTRIBUTED_SALE for compatibility")
        
        # Crear ORDER_GID_STR para joins
        if 'ORDER_GID' in df_processed.columns:
            df_processed['ORDER_GID_STR'] = df_processed['ORDER_GID'].astype(str)
        
        logger.info(f"Processed Shopify data: {len(df_processed)} rows")
        return df_processed
        
    except Exception as e:
        error_msg = f"Error processing Shopify data: {str(e)}"
        state.add_error_message(error_msg)
        return pd.DataFrame()

def process_ga_sessions_data(df: pd.DataFrame) -> pd.DataFrame:
    """Procesa y limpia los datos de sesiones de GA con columnas disponibles."""
    if not handle_dataframe_error(df, "process_ga_sessions_data"):
        return pd.DataFrame()
    
    try:
        df_processed = df.copy()
        
        # RENAME COLUMNS from RAW_GA_CAMPAIGN_METRICS schema - mantener CAMPAIGN_DATE para consistencia
        rename_map = {
            'CAMPAIGN_USERS': 'TOTAL_USERS',
            'CAMPAIGN_SESSIONS': 'TOTAL_SESSIONS',
            'CAMPAIGN_PURCHASE_EVENTS': 'TOTAL_PURCHASES',
            'CAMPAIGN_BOUNCE_RATE': 'BOUNCE_RATE',
            'CAMPAIGN_AVG_SESSION_DURATION': 'AVG_SESSION_DURATION'
        }
        df_processed = df_processed.rename(columns=rename_map)
        
        # Procesamiento de fechas
        if 'CAMPAIGN_DATE' in df_processed.columns:
            df_processed['CAMPAIGN_DATE'] = pd.to_datetime(
                df_processed['CAMPAIGN_DATE']
            ).dt.tz_localize(None)
        
        # Limpiar datos numéricos disponibles
        numeric_columns = ['TOTAL_USERS', 'TOTAL_SESSIONS', 'TOTAL_PURCHASES', 'BOUNCE_RATE', 'AVG_SESSION_DURATION']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Reportar métricas no disponibles (una sola vez por sesión)
        if not hasattr(process_ga_sessions_data, '_missing_reported'):
            state.add_missing_data_point("Eventos de carrito y checkout - no disponibles en schema actual")
            process_ga_sessions_data._missing_reported = True
        
        logger.info(f"Processed GA Sessions data: {len(df_processed)} rows with available columns")
        return df_processed
        
    except Exception as e:
        error_msg = f"Error processing GA Sessions data: {str(e)}"
        state.add_error_message(error_msg)
        return pd.DataFrame()

def process_ga_orders_data(df: pd.DataFrame) -> pd.DataFrame:
    """Procesa y limpia los datos de órdenes de GA con columnas disponibles."""
    if not handle_dataframe_error(df, "process_ga_orders_data"):
        return pd.DataFrame()
    
    try:
        df_processed = df.copy()

        # RENAME COLUMNS from RAW_GA_CAMPAIGN_TRANSACTIONS schema - mantener CAMPAIGN_DATE para consistencia
        rename_map = {
            'TRANSACTION_ID': 'Transaction_ID',
            'CAMPAIGN_PRIMARY_GROUP': 'Session_primary_channel_group',
            'CAMPAIGN_ID': 'Campaign_ID',
            'CAMPAIGN_SOURCE': 'Campaign'
        }
        df_processed = df_processed.rename(columns=rename_map)
        
        # Procesamiento de fechas
        if 'CAMPAIGN_DATE' in df_processed.columns:
            df_processed['CAMPAIGN_DATE'] = pd.to_datetime(df_processed['CAMPAIGN_DATE']).dt.tz_localize(None)
        
        # Limpiar Transaction_ID
        if 'Transaction_ID' in df_processed.columns:
            df_processed['Transaction_ID'] = df_processed['Transaction_ID'].astype(str)
            
        # Limpiar datos numéricos disponibles
        if 'Key_events' not in df_processed.columns:
            df_processed['Key_events'] = 0 # Placeholder since it's not in the source table
            df_processed['Key_events'] = pd.to_numeric(df_processed['Key_events'], errors='coerce')
        
        # Reportar métricas no disponibles (una sola vez por sesión)
        if not hasattr(process_ga_orders_data, '_missing_reported'):
            state.add_missing_data_point("Session Quality Score - no disponible en schema actual de GA_ORDERS")
            state.add_missing_data_point("Device Category - no disponible en schema actual de GA_ORDERS")
            process_ga_orders_data._missing_reported = True
        
        logger.info(f"Processed GA Orders data: {len(df_processed)} rows with available columns")
        return df_processed
        
    except Exception as e:
        error_msg = f"Error processing GA Orders data: {str(e)}"
        state.add_error_message(error_msg)
        return pd.DataFrame()

def process_ads_data(df: pd.DataFrame) -> pd.DataFrame:
    """Procesa y limpia los datos de campañas publicitarias con columnas disponibles."""
    if not handle_dataframe_error(df, "process_ads_data"):
        return pd.DataFrame()
    
    try:
        df_processed = df.copy()

        # RENAME COLUMNS from RAW_GA_CAMPAIGN_METRICS schema - mantener CAMPAIGN_DATE para consistencia
        rename_map = {
            'CAMPAIGN_ID': 'Campaign_ID',
            'CAMPAIGN_NAME': 'Source', # Using Name as Source
            'CAMPAIGN_USERS': 'Total_users',
            'CAMPAIGN_SESSIONS': 'Sessions',
            'CAMPAIGN_IMPRESSIONS': 'Ads_impressions',
            'CAMPAIGN_CLICKS': 'Ads_clicks',
            'CAMPAIGN_COST': 'Ads_cost',
            'CAMPAIGN_ROAS': 'Return_on_ad_spend',
            'CAMPAIGN_NEW_USERS': 'New_users'
        }
        df_processed = df_processed.rename(columns=rename_map)
        
        # Handle Primary_channel_group - use from data if available, otherwise default to 'Paid'
        if 'CAMPAIGN_PRIMARY_GROUP' in df_processed.columns:
            df_processed['Primary_channel_group'] = df_processed['CAMPAIGN_PRIMARY_GROUP']
        else:
            df_processed['Primary_channel_group'] = 'Paid' # Default since this column doesn't exist in RAW_GA_CAMPAIGN_METRICS
        
        # Procesamiento de fechas
        if 'CAMPAIGN_DATE' in df_processed.columns:
            df_processed['CAMPAIGN_DATE'] = pd.to_datetime(df_processed['CAMPAIGN_DATE']).dt.tz_localize(None)
        
        # Calcular Ads_CTR si no existe
        if 'Ads_clicks' in df_processed.columns and 'Ads_impressions' in df_processed.columns:
            df_processed['Ads_CTR'] = safe_division(df_processed['Ads_clicks'], df_processed['Ads_impressions'])
        
        # Limpiar datos numéricos disponibles
        numeric_columns = ['Total_users', 'Sessions', 'Ads_impressions', 'Ads_clicks', 'Ads_cost', 'Return_on_ad_spend', 'Ads_CTR', 'New_users']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Crear columna New_users para compatibilidad (usar Total_users como proxy)
        if 'Total_users' in df_processed.columns and 'New_users' not in df_processed.columns:
            df_processed['New_users'] = df_processed['Total_users']
        elif 'New_users' not in df_processed.columns:
            state.add_missing_data_point("Columna 'New_users' no disponible para Ads")
        
        # Reportar métricas no disponibles (una sola vez por sesión)
        if not hasattr(process_ads_data, '_missing_reported'):
            state.add_missing_data_point("Leads Generated - no disponible en schema actual de GA_ADS")
        
        logger.info(f"Processed Ads data: {len(df_processed)} rows with available columns")
        return df_processed
        
    except Exception as e:
        error_msg = f"Error processing Ads data: {str(e)}"
        state.add_error_message(error_msg)
        return pd.DataFrame()

def process_ft_customers_data(df: pd.DataFrame) -> pd.DataFrame:
    """Procesa y limpia los datos de la tabla FT_CUSTOMERS."""
    if not handle_dataframe_error(df, "process_ft_customers_data"):
        return pd.DataFrame()
    
    try:
        df_processed = df.copy()

        # RENAME COLUMNS from RAW_SHOPIFY_CUSTOMERS schema
        rename_map = {
            'CUSTOMER_CREATE_DATE': 'CUSTOMER_CREATION_DATE',
            'CUSTOMER_TOTAL_SPENT_AMOUNT': 'CUSTOMER_TOTAL_SPEND',
            'CUSTOMER_TOTAL_ORDERS': 'CUSTOMER_TOTAL_NUMBER_OF_ORDERS'
        }
        df_processed = df_processed.rename(columns=rename_map)
        
        # Procesamiento de fechas
        date_columns = ['CUSTOMER_CREATION_DATE']
        for col in date_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce').dt.tz_localize(None)
        
        # Limpiar email para compatibilidad
        if 'CUSTOMER_EMAIL' in df_processed.columns:
            df_processed['CUSTOMER_EMAIL'] = (
                df_processed['CUSTOMER_EMAIL']
                .astype(str)
                .fillna('unknown_email')
                .str.lower()
                .str.strip()
            )
            
            # Crear USER_EMAIL para compatibilidad con otras funciones
            df_processed['USER_EMAIL'] = df_processed['CUSTOMER_EMAIL']
        
        # Limpiar datos numéricos
        numeric_columns = [
            'CUSTOMER_TOTAL_SPEND', 'CUSTOMER_TOTAL_NUMBER_OF_ORDERS'
        ]
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Crear columnas de compatibilidad para análisis existentes
        if 'CUSTOMER_CREATION_DATE' in df_processed.columns:
            df_processed['ORDER_CREATE_DATE'] = df_processed['CUSTOMER_CREATION_DATE']
            
        if 'CUSTOMER_TOTAL_SPEND' in df_processed.columns:
            df_processed['PRICE'] = df_processed['CUSTOMER_TOTAL_SPEND']
            df_processed['ORDER_TOTAL_PRICE'] = df_processed['CUSTOMER_TOTAL_SPEND']
        
        logger.info(f"Processed FT_CUSTOMERS data: {len(df_processed)} rows with available columns")
        return df_processed
        
    except Exception as e:
        error_msg = f"Error processing FT_CUSTOMERS data: {str(e)}"
        state.add_error_message(error_msg)
        return pd.DataFrame()

@st.cache_data(ttl=config.CACHE_TTL, show_spinner="🔄 Cargando datos desde BigQuery...")
def load_real_data(source: str = "BigQuery") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Carga datos reales desde la fuente especificada (BigQuery o CSV)."""
    log_function_call(load_real_data.__name__, source=source)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
        
    if source == "BigQuery" and config.USE_BIGQUERY:
        logger.info("Loading data from BigQuery with progress indicators...")
        data = load_data_from_bigquery_with_progress(progress_bar, status_text)
    else:
        logger.info("Loading data from local CSV files...")
        data = load_data_from_csv()
        progress_bar.progress(100)
        status_text.text("Carga de CSV completada.")
        
        progress_bar.empty()
        status_text.empty()
    return data

def load_data_from_bigquery_with_progress(progress_bar, status_text) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Carga datos desde BigQuery con una barra de progreso e información de estado."""
    start_time = time.time()
    
    client = get_bigquery_client()
    if not client:
        st.error("Error al conectar con BigQuery. Revisa las credenciales.")
        return None, None, None, None, None
    
    start_date, end_date = get_default_date_range()
    logger.info(f"Loading data with default date range: {start_date} to {end_date}")
    
    # 1. Load FT_CUSTOMERS
    status_text.text("Cargando clientes (Shopify)...")
    progress_bar.progress(5)
    
    ft_customers_table = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['ft_customers']}"
    logger.info(f"Loading ft_customers from BigQuery table: {ft_customers_table}")
    
    df_ft_customers = load_table_from_bigquery_optimized(
        client, ft_customers_table,
        date_column=None,  # No date filter - load all data
        start_date=None,
        end_date=None,
        columns=['CUSTOMER_ID', 'CUSTOMER_CREATE_DATE', 'CUSTOMER_DISPLAY_NAME', 'CUSTOMER_EMAIL', 
                 'CUSTOMER_TOTAL_SPENT_AMOUNT', 'CUSTOMER_TOTAL_ORDERS', 'CUSTOMER_LAST_ORDER_ID'],
        limit=config.MAX_QUERY_ROWS
    )
    
    if df_ft_customers is not None:
        logger.info(f"✅ ft_customers: Raw data loaded - {len(df_ft_customers)} rows")
        df_ft_customers = process_ft_customers_data(df_ft_customers)
        logger.info(f"✅ ft_customers: Processed {len(df_ft_customers)} → {len(df_ft_customers)} rows in {time.time() - start_time:.2f}s")
    else:
        logger.error("Failed to load ft_customers data.")
        df_ft_customers = pd.DataFrame()

    # 2. Load SHOPIFY_ORDERS for shipping data
    status_text.text("Cargando datos de envío (Shopify)...")
    progress_bar.progress(15)

    orders_table_name = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['shopify_orders']}"
    logger.info(f"Loading orders from BigQuery table: {orders_table_name}")

    df_orders = load_table_from_bigquery_optimized(
        client, orders_table_name,
        date_column=None,  # No date filter - load all data
        start_date=None,
        end_date=None,
        columns=['ORDER_ID', 'SHIPPING_CITY', 'SHIPPING_STATE', 'SHIPPING_LATITUDE', 'SHIPPING_LONGITUDE'],
        limit=config.MAX_QUERY_ROWS
    )
    if df_orders is None:
        logger.error("Failed to load shopify_orders data.")
        df_orders = pd.DataFrame()

    # 3. Load ITEMS (Shopify Orders Line Items)
    status_text.text("Cargando ítems de pedidos (Shopify)...")
    progress_bar.progress(25)
    
    items_table = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['items']}"
    logger.info(f"Loading items from BigQuery table: {items_table}")
    
    df_items = load_table_from_bigquery_optimized(
        client, items_table,
        date_column=None,  # No date filter - load all data
        start_date=None,
        end_date=None,
        columns=['ORDER_ID', 'CUSTOMER_ID', 'ORDER_CREATE_DATE', 'PRODUCT_NAME', 
                 'LINEITEM_QTY', 'PRICE', 'PRODUCT_TYPE', 'PRODUCT_VENDOR'],
        limit=config.MAX_QUERY_ROWS
    )
    
    if df_items is not None:
        logger.info(f"✅ items: Raw data loaded - {len(df_items)} rows")
        df_items = process_shopify_data(df_items, df_ft_customers, df_orders) # Pass customer and order data
        logger.info(f"✅ items: Processed {len(df_items)} → {len(df_items)} rows in {time.time() - start_time:.2f}s")
    else:
        logger.error("Failed to load items data.")
        df_items = pd.DataFrame()
        
    # 4. Load SESSIONS (GA Metrics)
    status_text.text("Cargando métricas de sesión (GA)...")
    progress_bar.progress(50) # Adjusted progress
    
    sessions_table = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['sessions']}"
    logger.info(f"Loading sessions from BigQuery table: {sessions_table}")
    
    df_sessions = load_table_from_bigquery_optimized(
        client, sessions_table,
        date_column=None,  # No date filter - load all data
        start_date=None,
        end_date=None,
        columns=['CAMPAIGN_DATE', 'CAMPAIGN_USERS', 'CAMPAIGN_SESSIONS', 'CAMPAIGN_PURCHASE_EVENTS', 
                 'CAMPAIGN_BOUNCE_RATE', 'CAMPAIGN_AVG_SESSION_DURATION'],
        limit=config.MAX_QUERY_ROWS
    )
    
    if df_sessions is not None:
        logger.info(f"✅ sessions: Raw data loaded - {len(df_sessions)} rows")
        df_sessions = process_ga_sessions_data(df_sessions)
        logger.info(f"✅ sessions: Processed {len(df_sessions)} → {len(df_sessions)} rows in {time.time() - start_time:.2f}s")
    else:
        logger.error("Failed to load sessions data.")
        df_sessions = pd.DataFrame()
        
    # 5. Load GA_ORDERS
    status_text.text("Cargando órdenes de GA...")
    progress_bar.progress(65) # Adjusted progress
    
    ga_orders_table = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['ga_orders']}"
    logger.info(f"Loading ga_orders from BigQuery table: {ga_orders_table}")
    
    df_ga_orders = load_table_from_bigquery_optimized(
        client, ga_orders_table,
        date_column=None,  # No date filter - load all data
        start_date=None,
        end_date=None,
        columns=['CAMPAIGN_DATE', 'TRANSACTION_ID', 'CAMPAIGN_PRIMARY_GROUP', 
                 'CAMPAIGN_ID', 'CAMPAIGN_SOURCE'],
        limit=config.MAX_QUERY_ROWS
    )
    
    if df_ga_orders is not None:
        logger.info(f"✅ ga_orders: Raw data loaded - {len(df_ga_orders)} rows")
        df_ga_orders = process_ga_orders_data(df_ga_orders)
        logger.info(f"✅ ga_orders: Processed {len(df_ga_orders)} → {len(df_ga_orders)} rows in {time.time() - start_time:.2f}s")
    else:
        logger.error("Failed to load ga_orders data.")
        df_ga_orders = pd.DataFrame()
        
    # 6. Load ADS
    status_text.text("Cargando anuncios de GA...")
    progress_bar.progress(85) # Adjusted progress
    
    ads_table = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['ga_ads']}"
    logger.info(f"Loading ads from BigQuery table: {ads_table}")
    
    df_ads = load_table_from_bigquery_optimized(
        client, ads_table,
        date_column=None,  # No date filter - load all data
        start_date=None,
        end_date=None,
        columns=['CAMPAIGN_DATE', 'CAMPAIGN_ID', 'CAMPAIGN_NAME', 'CAMPAIGN_USERS', 
                 'CAMPAIGN_SESSIONS', 'CAMPAIGN_IMPRESSIONS', 'CAMPAIGN_CLICKS', 'CAMPAIGN_COST', 
                 'CAMPAIGN_ROAS', 'CAMPAIGN_NEW_USERS'],
        limit=config.MAX_QUERY_ROWS
    )
    
    if df_ads is not None:
        logger.info(f"✅ ads: Raw data loaded - {len(df_ads)} rows")
        df_ads = process_ads_data(df_ads)
        logger.info(f"✅ ads: Processed {len(df_ads)} → {len(df_ads)} rows in {time.time() - start_time:.2f}s")
    else:
        logger.error("Failed to load ads data.")
        df_ads = pd.DataFrame()
    
    logger.info(f"BigQuery data loading completed in {time.time() - start_time:.2f}s")
    
    return df_items, df_sessions, df_ga_orders, df_ads, df_ft_customers

def load_data_from_bigquery() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Función de compatibilidad - redirige a la versión optimizada."""
    return load_data_from_bigquery_with_progress(None, None)

# Función CSV marcada como DEPRECADA - Ya no se usa
def load_data_from_csv() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Carga los datos desde archivos CSV locales."""
    df_items = load_csv_safely('data/items.csv', required_columns=['ORDER_GID', 'USER_EMAIL', 'ORDER_CREATE_DATE'])
    df_sessions = load_csv_safely('data/sessions.csv', required_columns=['SESSION_ID', 'USER_EMAIL', 'EVENT_DATE'])
    df_ga_orders = load_csv_safely('data/ga_orders.csv', required_columns=['Transaction_ID', 'Date'])
    df_ads = load_csv_safely('data/ads.csv', required_columns=['Campaign_ID', 'Date'])
    df_ft_customers = load_csv_safely('data/ft_customers.csv', required_columns=['CUSTOMER_ID', 'CUSTOMER_EMAIL'])
    df_orders = load_csv_safely('data/shopify_orders.csv', required_columns=['ORDER_ID', 'SHIPPING_COUNTRY'])

    # Process all dataframes
    if df_items is not None and df_ft_customers is not None:
        df_items = process_shopify_data(df_items, df_ft_customers, df_orders)
    if df_sessions is not None:
        df_sessions = process_ga_sessions_data(df_sessions)
    if df_ga_orders is not None:
        df_ga_orders = process_ga_orders_data(df_ga_orders)
    if df_ads is not None:
        df_ads = process_ads_data(df_ads)
    if df_ft_customers is not None:
        df_ft_customers = process_ft_customers_data(df_ft_customers)

    return df_items, df_sessions, df_ga_orders, df_ads, df_ft_customers

def _check_missing_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """Registra las columnas que faltan en un DataFrame."""
    # Lista de columnas conocidas que no existen en los schemas actuales
    known_missing = {
        'ADD_TO_CART_EVENTS', 
        'CHECKOUT_INITIATED_EVENTS', 'SESSION_QUALITY_SCORE', 'DEVICE_CATEGORY',
        'LEADS_GENERATED', 'New_users'
    }
    
    for col in columns:
        if col not in df.columns and col not in known_missing:
            # Solo reportar columnas que realmente deberían existir
            state.add_missing_data_point(f"Columna inesperadamente faltante: {col}")

def get_date_range_info(df: pd.DataFrame, date_col: str) -> Dict[str, Union[pd.Timestamp, None]]:
    """Obtiene información del rango de fechas de un DataFrame."""
    if not handle_dataframe_error(df, f"get_date_range_info for column {date_col}"):
        return {'min_date': pd.Timestamp.min.tz_localize(None), 'max_date': pd.Timestamp.max.tz_localize(None)}
    
    if date_col not in df.columns:
        logger.warning(f"Date column {date_col} not found in DataFrame")
        return {'min_date': pd.Timestamp.min.tz_localize(None), 'max_date': pd.Timestamp.max.tz_localize(None)}
    
    return {'min_date': df[date_col].min(), 'max_date': df[date_col].max()}

def filter_dataframes_by_date(dataframes: Dict[str, pd.DataFrame], date_columns: Dict[str, str], 
                             start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """Filtra múltiples DataFrames por rango de fechas."""
    log_function_call("filter_dataframes_by_date", start_date=start_date, end_date=end_date)
    
    filtered_dfs = {}
    for name, df in dataframes.items():
        if not handle_dataframe_error(df, f"filter_dataframes_by_date for {name}"):
            filtered_dfs[name] = pd.DataFrame()
            continue
            
        if name in date_columns:
            date_col = date_columns[name]
            if date_col in df.columns:
                try:
                    filtered_dfs[name] = df[df[date_col].between(start_date, end_date)]
                    logger.info(f"Filtered {name}: {len(filtered_dfs[name])} rows after date filter")
                except Exception as e:
                    logger.error(f"Error filtering {name} by date: {str(e)}")
                    filtered_dfs[name] = df
            else:
                logger.warning(f"Date column {date_col} not found in {name}")
                filtered_dfs[name] = df
        else:
            filtered_dfs[name] = df
    
    return filtered_dfs

# —————————————————————————————————————————
# FUNCIONES DE CATEGORIZACIÓN Y ENRICHMENT
# —————————————————————————————————————————

def categorize_product(product_name: str) -> str:
    """Categoriza productos en categorías específicas para deportes de raqueta."""
    if pd.isna(product_name) or str(product_name).lower().strip() in ['unknown product', '', 'nan']:
        return 'Sin Clasificar'
    
    try:
        product_lower = str(product_name).lower().strip()
        
        # Usar configuración centralizada
        for category, keywords in config.PRODUCT_CATEGORIES.items():
            if any(keyword in product_lower for keyword in keywords):
                return category
        
        return 'Otros'
        
    except Exception as e:
        logger.error(f"Error categorizing product '{product_name}': {str(e)}")
        return 'Sin Clasificar'

def identify_sport_universe(product_name: str) -> str:
    """Identifica si un producto pertenece al universo de pickleball o pádel."""
    if pd.isna(product_name) or str(product_name).lower().strip() in ['unknown product', '', 'nan']:
        return 'Sin Clasificar'
    
    try:
        product_lower = str(product_name).lower().strip()
        
        # Usar configuración centralizada
        for sport, keywords in config.SPORT_KEYWORDS.items():
            if any(keyword in product_lower for keyword in keywords):
                return sport
        
        return 'Genérico/Ambos'
        
    except Exception as e:
        logger.error(f"Error identifying sport for product '{product_name}': {str(e)}")
        return 'Sin Clasificar'

def get_player_level_from_product(product_name: str) -> str:
    """Intenta extraer el nivel de jugador del nombre del producto."""
    if pd.isna(product_name) or str(product_name).lower().strip() in ['unknown product', '', 'nan']:
        return 'Sin Especificar'
    
    try:
        product_lower = str(product_name).lower().strip()
        
        # Usar configuración centralizada
        for level, keywords in config.LEVEL_KEYWORDS.items():
            if any(keyword in product_lower for keyword in keywords):
                return level
        
        return 'Sin Especificar'
        
    except Exception as e:
        logger.error(f"Error identifying player level for product '{product_name}': {str(e)}")
        return 'Sin Especificar'

def group_sports(sport: str) -> str:
    """Agrupa deportes en categorías principales."""
    if pd.isna(sport):
        return 'Otros/Genérico'
    
    sport_clean = str(sport).strip()
    
    if sport_clean == 'Pádel':
        return 'Pádel'
    elif sport_clean == 'Pickleball':
        return 'Pickleball'
    else:
        return 'Otros/Genérico'

def enrich_dataframe_with_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece un DataFrame con todas las categorías de producto."""
    if not handle_dataframe_error(df, "enrich_dataframe_with_categories"):
        return pd.DataFrame()
    
    try:
        log_function_call("enrich_dataframe_with_categories", rows=len(df))
        
        df_enriched = df.copy()
        
        # Verificar que existe la columna requerida
        if 'PRODUCT_NAME' not in df_enriched.columns:
            logger.warning("PRODUCT_NAME column not found, creating default categories")
            df_enriched['PRODUCT_CATEGORY'] = 'Sin Clasificar'
            df_enriched['SPORT_UNIVERSE'] = 'Sin Clasificar'
            df_enriched['PLAYER_LEVEL'] = 'Sin Especificar'
            df_enriched['SPORT_GROUPED'] = 'Otros/Genérico'
            return df_enriched
        
        # Aplicar categorización
        df_enriched['PRODUCT_CATEGORY'] = df_enriched['PRODUCT_NAME'].apply(categorize_product)
        df_enriched['SPORT_UNIVERSE'] = df_enriched['PRODUCT_NAME'].apply(identify_sport_universe)
        df_enriched['PLAYER_LEVEL'] = df_enriched['PRODUCT_NAME'].apply(get_player_level_from_product)
        df_enriched['SPORT_GROUPED'] = df_enriched['SPORT_UNIVERSE'].apply(group_sports)
        
        logger.info(f"Successfully enriched DataFrame with categories for {len(df_enriched)} rows")
        return df_enriched
        
    except Exception as e:
        error_msg = f"Error enriching DataFrame with categories: {str(e)}"
        state.add_error_message(error_msg)
        return df

# —————————————————————————————————————————
# FUNCIONES DE ANÁLISIS REUTILIZABLES
# —————————————————————————————————————————

def calculate_customer_metrics(df_items_filtered: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, int, float]]:
    """Calcula métricas básicas de clientes de forma reutilizable."""
    if df_items_filtered.empty:
        return {'customer_orders': pd.DataFrame(), 'metrics': {}}
    
    customer_orders = df_items_filtered.groupby('USER_EMAIL')['ORDER_GID'].nunique().reset_index(name='NUM_ORDERS')
    
    metrics = {
        'avg_orders_per_customer': customer_orders['NUM_ORDERS'].mean(),
        'total_customers': customer_orders['USER_EMAIL'].nunique(),
        'recurrent_customers': len(customer_orders[customer_orders['NUM_ORDERS'] > 1]),
        'recurrence_rate': len(customer_orders[customer_orders['NUM_ORDERS'] > 1]) / len(customer_orders) * 100 if len(customer_orders) > 0 else 0
    }
    
    return {'customer_orders': customer_orders, 'metrics': metrics}

def analyze_repurchase_behavior(df_items_filtered: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analiza el comportamiento de recompra de los clientes."""
    if df_items_filtered.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Protección contra datasets muy grandes
    if len(df_items_filtered) > 300000:  # Más de 300k filas para análisis detallado
        logger.warning(f"Large dataset for repurchase analysis ({len(df_items_filtered):,} rows), sampling")
        df_sample = df_items_filtered.sample(n=min(200000, len(df_items_filtered)), random_state=42)
        st.info(f"ℹ️ **Análisis de recompra optimizado**: Procesando {len(df_sample):,} registros de {len(df_items_filtered):,} totales.")
    else:
        df_sample = df_items_filtered
    
    df_enriched = enrich_dataframe_with_categories(df_sample)
    
    try:
        # Análisis de primera compra vs recompras
        customer_orders = df_enriched.groupby('USER_EMAIL').agg({
            'ORDER_CREATE_DATE': ['min', 'max', 'count'],
            'ORDER_GID': 'nunique',
            'PRODUCT_CATEGORY': 'first',
            'SPORT_UNIVERSE': 'first',
            'PLAYER_LEVEL': 'first'
        }).reset_index()
        
        # Aplanar columnas multinivel
        customer_orders.columns = [
            'USER_EMAIL', 'FIRST_PURCHASE_DATE', 'LAST_PURCHASE_DATE', 
            'TOTAL_LINE_ITEMS', 'TOTAL_ORDERS', 'FIRST_CATEGORY', 
            'FIRST_SPORT', 'FIRST_LEVEL'
        ]
        
        # Identificar clientes con recompras
        repurchase_customers = customer_orders[customer_orders['TOTAL_ORDERS'] > 1]['USER_EMAIL']
        repurchase_analysis = df_enriched[df_enriched['USER_EMAIL'].isin(repurchase_customers)].copy()
        first_purchases = df_enriched.loc[df_enriched.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].idxmin()]
        
        return customer_orders, repurchase_analysis, first_purchases
        
    except Exception as e:
        error_msg = f"Error in repurchase behavior analysis: {str(e)}"
        logger.error(error_msg)
        state.add_error_message(error_msg)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def calculate_time_to_repurchase(df_items_filtered: pd.DataFrame) -> pd.DataFrame:
    """Calcula el tiempo promedio hasta la recompra por categoría y deporte."""
    if df_items_filtered.empty:
        return pd.DataFrame()
    
    df_enriched = enrich_dataframe_with_categories(df_items_filtered)
    
    customer_order_dates = df_enriched.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].apply(list).reset_index()
    customer_order_dates = customer_order_dates[customer_order_dates['ORDER_CREATE_DATE'].apply(len) > 1]
    
    if customer_order_dates.empty:
        return pd.DataFrame()
    
    repurchase_times = []
    for _, row in customer_order_dates.iterrows():
        dates = sorted(row['ORDER_CREATE_DATE'])
        time_to_second = (dates[1] - dates[0]).days
        
        first_purchase = df_enriched[
            (df_enriched['USER_EMAIL'] == row['USER_EMAIL']) & 
            (df_enriched['ORDER_CREATE_DATE'] == dates[0])
        ]
        
        if not first_purchase.empty:
            repurchase_times.append({
                'USER_EMAIL': row['USER_EMAIL'],
                'DAYS_TO_REPURCHASE': time_to_second,
                'FIRST_CATEGORY': first_purchase['PRODUCT_CATEGORY'].iloc[0],
                'FIRST_SPORT': first_purchase['SPORT_UNIVERSE'].iloc[0]
            })
    
    return pd.DataFrame(repurchase_times)

# —————————————————————————————————————————
# FUNCIONES DE VISUALIZACIÓN REUTILIZABLES
# —————————————————————————————————————————

def optimize_text_position(fig: go.Figure) -> go.Figure:
    """Optimiza la posición del texto en los gráficos para mejor legibilidad."""
    try:
        # Para gráficos de barras, ajustar posición del texto basado en el valor y el espacio disponible
        for trace in fig.data:
            if hasattr(trace, 'textposition') and trace.textposition == 'auto':
                # Si ya está en 'auto', Plotly manejará automáticamente la posición
                fig.update_traces(
                    textfont=dict(size=11, color='black'),
                    # Configurar el texto para que se muestre en la mejor posición automáticamente
                    textposition='auto'
                )
            elif hasattr(trace, 'text') and trace.text is not None:
                # Para casos específicos donde necesitamos control manual
                if hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
                    # Analizar los valores para decidir la mejor posición
                    values = [v for v in trace.y if v is not None and not pd.isna(v)]
                    if values:
                        max_val = max(values)
                        min_val = min(values)
                        avg_val = sum(values) / len(values)
                        
                        # Si la mayoría de barras son pequeñas, usar 'outside'
                        # Si son grandes, usar 'inside' o 'auto'
                        if avg_val < max_val * 0.2:  # Valores pequeños
                            fig.update_traces(
                                textposition='outside',
                                textfont=dict(color='black', size=10)
                            )
                        else:  # Valores medianos a grandes
                            fig.update_traces(
                                textposition='auto',
                                textfont=dict(size=11)
                            )
        
        # Ajustar el layout para dar más espacio al texto si es necesario
        fig.update_layout(
            margin=dict(t=80, b=60, l=60, r=60),  # Más margen para el texto
            font=dict(size=11)
        )
        
        return fig
    except Exception as e:
        logger.warning(f"Error optimizing text position: {str(e)}")
        return fig

def safe_plotly_chart(fig: Optional[go.Figure], use_container_width: bool = True, 
                     error_message: str = "No hay datos suficientes para mostrar este gráfico con los filtros seleccionados.",
                     key: Optional[str] = None) -> None:
    """Función helper para evitar errores con data vacía en gráficos."""
    try:
        if fig is not None:
            # Optimizar posición del texto antes de mostrar
            fig = optimize_text_position(fig)
            # Generate unique key if not provided
            if key is None:
                key = f"chart_{hash(str(fig))}"
            st.plotly_chart(fig, use_container_width=use_container_width, key=key)
        else:
            st.info(error_message)
    except Exception as e:
        logger.error(f"Error displaying plotly chart: {str(e)}")
        st.error("Error al mostrar el gráfico. Por favor, revisa los datos.")

def create_metric_layout(metrics_data: List[Dict[str, Union[str, float, int]]], 
                        columns: int = None) -> None:
    """Crea un layout de métricas reutilizable."""
    if not metrics_data:
        logger.warning("No metrics data provided to create_metric_layout")
        return
    
    try:
        columns = columns or config.DEFAULT_COLUMNS
        cols = st.columns(columns)
        
        for i, metric in enumerate(metrics_data):
            if not isinstance(metric, dict):
                logger.warning(f"Invalid metric data at index {i}: {metric}")
                continue
                
            with cols[i % columns]:
                st.metric(
                    label=metric.get('label', 'N/A'),
                    value=metric.get('value', 'N/A'),
                    delta=metric.get('delta'),
                    help=metric.get('help')
                )
    except Exception as e:
        logger.error(f"Error creating metric layout: {str(e)}")
        st.error("Error al crear el layout de métricas.")

def create_bar_chart(data: pd.DataFrame, x: str, y: str, title: str, 
                    labels: Optional[Dict[str, str]] = None,
                    color: Optional[str] = None,
                    color_scale: Optional[str] = None,
                    text: Optional[str] = None,
                    height: int = None) -> Optional[go.Figure]:
    """Crea un gráfico de barras estándar con configuración consistente."""
    if not handle_dataframe_error(data, f"create_bar_chart for {title}"):
        return None
    
    try:
        height = height or config.DEFAULT_CHART_HEIGHT
        
        # Validar columnas requeridas
        required_cols = [x, y]
        if color:
            required_cols.append(color)
        if text:
            required_cols.append(text)
        
        if not validate_required_columns(data, required_cols, f"create_bar_chart: {title}"):
            return None
        
        fig = px.bar(
            data, x=x, y=y, title=title, 
            labels=labels or {}, 
            color=color,
            color_continuous_scale=color_scale,
            text=text
        )
        
        if text:
            fig.update_traces(texttemplate='%{text}', textposition='auto')
        
        fig.update_layout(
            title_font_size=14, 
            height=height,
            margin=dict(t=60, b=20, l=20, r=20)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating bar chart '{title}': {str(e)}")
        return None

def create_line_chart(data: pd.DataFrame, x: str, y: str, title: str,
                     labels: Optional[Dict[str, str]] = None,
                     markers: bool = True,
                     height: int = None) -> Optional[go.Figure]:
    """Crea un gráfico de líneas estándar con configuración consistente."""
    if not handle_dataframe_error(data, f"create_line_chart for {title}"):
        return None
    
    try:
        height = height or config.DEFAULT_CHART_HEIGHT
        
        # Validar columnas requeridas
        if not validate_required_columns(data, [x, y], f"create_line_chart: {title}"):
            return None
        
        fig = px.line(
            data, x=x, y=y, title=title,
            labels=labels or {},
            markers=markers
        )
        
        fig.update_layout(
            title_font_size=14,
            height=height,
            margin=dict(t=60, b=20, l=20, r=20)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating line chart '{title}': {str(e)}")
        return None

def create_correlation_heatmap(data: pd.DataFrame, x_col: str, y_col: str, 
                              title: str, normalize: str = 'index') -> Optional[go.Figure]:
    """Crea un heatmap de correlación estándar."""
    if not handle_dataframe_error(data, f"create_correlation_heatmap for {title}"):
        return None
    
    try:
        # Validar columnas requeridas
        if not validate_required_columns(data, [x_col, y_col], f"create_correlation_heatmap: {title}"):
            return None
        
        correlation_matrix = pd.crosstab(data[y_col], data[x_col], normalize=normalize) * 100
        
        if correlation_matrix.empty:
            logger.warning(f"Empty correlation matrix for {title}")
            return None
        
        # Reorganizar columnas por orden lógico de nivel si aplicable
        if 'LEVEL' in x_col.upper():
            available_levels = [level for level in config.LEVEL_ORDER if level in correlation_matrix.columns]
            correlation_matrix = correlation_matrix.reindex(columns=available_levels, fill_value=0)
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(x=x_col.replace('_', ' ').title(), y=y_col.replace('_', ' ').title(), color="Porcentaje (%)"),
            title=title,
            color_continuous_scale='RdYlBu_r',
            text_auto='.1f',
            aspect='auto'
        )
        
        fig.update_layout(
            title_font_size=14, 
            height=450,
            font=dict(size=10),
            margin=dict(l=120, r=20, t=80, b=60)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap '{title}': {str(e)}")
        return None

# —————————————————————————————————————————
# FUNCIONES DE SIDEBAR Y SETUP
# —————————————————————————————————————————

def create_sidebar(dataframes: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Crea un sidebar limpio y atractivo con filtros de fecha optimizados."""
    log_function_call("create_sidebar", num_dataframes=len(dataframes))
    
    try:
        with st.sidebar:
            st.markdown("""
                <div style='text-align: center; padding: 0.75rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.5rem;'>
                    <h2 style='color: #374151; margin: 0; font-weight: 600; font-size: 1.25rem;'>Customer Analytics</h2>
                    <p style='color: #6b7280; margin: 0.25rem 0 0 0; font-size: 0.875rem; font-weight: 400;'>Dashboard</p>
                </div>
            """, unsafe_allow_html=True)
            
            date_columns = {
                'df_items': 'ORDER_CREATE_DATE', 'df_sessions': 'CAMPAIGN_DATE', 
                'df_ga_orders': 'CAMPAIGN_DATE', 'df_ads': 'CAMPAIGN_DATE'
            }
            
            date_ranges = [
                info['min_date'] or info['max_date'] 
                for name, df in dataframes.items() 
                if name in date_columns and handle_dataframe_error(df, f"sidebar date range for {name}")
                for info in [get_date_range_info(df, date_columns[name])]
            ]
            date_ranges = [d for d in date_ranges if pd.notna(d)]

            overall_min_date = min(date_ranges) if date_ranges else (datetime.now() - timedelta(days=730))
            overall_max_date = max(date_ranges) if date_ranges else datetime.now()

            # --- Session State Initialization and Correction ---
            # Calculate YTD start date first
            ytd_start = datetime(datetime.now().year, 1, 1).date()
            
            if 'start_date' not in st.session_state:
                # Asegurar que YTD no sea menor que los datos disponibles
                if ytd_start < overall_min_date.date():
                    st.session_state.start_date = overall_min_date.date()
                else:
                    st.session_state.start_date = ytd_start
            if 'end_date' not in st.session_state:
                st.session_state.end_date = overall_max_date.date()
            if 'custom_date_active' not in st.session_state:
                st.session_state.custom_date_active = False

            def set_date_range(start, end, custom=False):
                st.session_state.start_date = start
                st.session_state.end_date = end
                st.session_state.custom_date_active = custom

            st.markdown("<h3 style='color: #374151; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.75rem;'>Filtro de Fechas</h3>", unsafe_allow_html=True)
            
            # --- Quick Buttons ---
            st.markdown("<p style='font-size: 0.9rem; font-weight: 500; color: #4B5563;'>Rangos rápidos</p>", unsafe_allow_html=True)
            button_cols = st.columns(5)
            if button_cols[0].button("3M", use_container_width=True, key="3m_button"):
                set_date_range((overall_max_date - pd.DateOffset(months=3)).date(), overall_max_date.date())
                st.rerun()
            if button_cols[1].button("6M", use_container_width=True, key="6m_button"):
                set_date_range((overall_max_date - pd.DateOffset(months=6)).date(), overall_max_date.date())
                st.rerun()
            if button_cols[2].button("YTD", use_container_width=True, key="ytd_button"):
                set_date_range(datetime(datetime.now().year, 1, 1).date(), overall_max_date.date())
                st.rerun()
            if button_cols[3].button("Todo", use_container_width=True, key="all_button"):
                set_date_range(overall_min_date.date(), overall_max_date.date())
                st.rerun()
            if button_cols[4].button("Custom", use_container_width=True, key="custom_button"):
                st.session_state.custom_date_active = not st.session_state.custom_date_active
                st.rerun()

            # --- Custom Date Range Input (conditionally displayed) ---
            if st.session_state.custom_date_active:
                st.markdown("<hr style='margin: 1.2rem 0;'>", unsafe_allow_html=True)
                
                def on_date_change():
                    if st.session_state.date_selector and len(st.session_state.date_selector) == 2:
                        st.session_state.start_date, st.session_state.end_date = st.session_state.date_selector

                st.date_input(
                    "Rango personalizado",
                    value=(pd.to_datetime(st.session_state.start_date).date(), pd.to_datetime(st.session_state.end_date).date()),
                    min_value=overall_min_date.date(),
                    max_value=overall_max_date.date(),
                    on_change=on_date_change,
                    key="date_selector"
                )

            if state.error_messages:
                st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
                with st.expander("⚠️ Errores Detectados", expanded=False):
                    for error in state.error_messages[-5:]:
                        st.error(error)
            
            # --- Additional Filters ---
            st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #374151; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.75rem;'>Filtros Adicionales</h3>", unsafe_allow_html=True)
            
            # Sport filter
            if 'df_items' in dataframes and not dataframes['df_items'].empty:
                available_sports = ['Todos'] + sorted(dataframes['df_items']['SPORT_UNIVERSE'].dropna().unique().tolist())
                selected_sport = st.selectbox(
                    "Deporte",
                    available_sports,
                    key="sport_filter"
                )
                if selected_sport != 'Todos':
                    st.session_state.selected_sport = selected_sport
                else:
                    st.session_state.selected_sport = None
            
            # Product category filter
            if 'df_items' in dataframes and not dataframes['df_items'].empty:
                available_categories = ['Todas'] + sorted(dataframes['df_items']['PRODUCT_CATEGORY'].dropna().unique().tolist())
                selected_category = st.selectbox(
                    "Categoría de Producto",
                    available_categories,
                    key="category_filter"
                )
                if selected_category != 'Todas':
                    st.session_state.selected_category = selected_category
                else:
                    st.session_state.selected_category = None
            
            # Level filter
            if 'df_items' in dataframes and not dataframes['df_items'].empty:
                available_levels = ['Todos'] + sorted(dataframes['df_items']['PLAYER_LEVEL'].dropna().unique().tolist())
                selected_level = st.selectbox(
                    "Nivel de Jugador",
                    available_levels,
                    key="level_filter"
                )
                if selected_level != 'Todos':
                    st.session_state.selected_level = selected_level
                else:
                    st.session_state.selected_level = None
            
            st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style='background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; 
                           padding: 0.75rem; text-align: center;'>
                    <p style='color: #374151; margin: 0; font-size: 0.8rem;'>
                        <strong>Datos disponibles:</strong><br>
                        <span style='color: #6b7280;'>
                            {overall_min_date.strftime('%d/%m/%Y')} - {overall_max_date.strftime('%d/%m/%Y')}
                        </span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            return pd.to_datetime(st.session_state.start_date), pd.to_datetime(st.session_state.end_date)
            
    except Exception as e:
        error_msg = f"Critical error in sidebar creation: {str(e)}"
        state.add_error_message(error_msg)
        logger.error(error_msg)
        return pd.to_datetime(datetime.now() - timedelta(days=365)), pd.to_datetime(datetime.now())

def setup_dashboard() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                              pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame, 
                              pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Configura el dashboard principal con estilos y datos."""
    log_function_call("setup_dashboard")
    
    # Limpiar errores previos
    state.clear_errors()
    
    # Aplicar estilos
    apply_custom_css()
    
    # Cargar datos
    logger.info("Loading dashboard data...")
    df_items, df_sessions, df_ga_orders, df_ads, df_ft_customers = load_real_data()
    
    # Verificar que al menos los datos principales fueron cargados
    if df_items is None:
        error_msg = "Error crítico: No se pudieron cargar los datos principales de órdenes (SHOPIFY_ORDERS_LINEITEMS.csv)"
        st.error(error_msg)
        state.add_error_message(error_msg)
        st.stop()
    
    # Crear DataFrames vacíos para los datos faltantes
    if df_sessions is None:
        logger.warning("GA Sessions data not loaded, creating empty DataFrame")
        df_sessions = pd.DataFrame()
    
    if df_ga_orders is None:
        logger.warning("GA Orders data not loaded, creating empty DataFrame")
        df_ga_orders = pd.DataFrame()
    
    if df_ads is None:
        logger.warning("GA Ads data not loaded, creating empty DataFrame")
        df_ads = pd.DataFrame()
    
    if df_ft_customers is None:
        logger.warning("FT_CUSTOMERS data not loaded, creating empty DataFrame")
        df_ft_customers = pd.DataFrame()
    
    # Diccionario de DataFrames para facilitar el manejo
    dataframes = {
        'df_items': df_items,
        'df_sessions': df_sessions, 
        'df_ga_orders': df_ga_orders,
        'df_ads': df_ads,
        'df_ft_customers': df_ft_customers
    }
    
    # Crear sidebar y obtener fechas
    start_date, end_date = create_sidebar(dataframes)
    
    # Validar fechas
    if start_date >= end_date:
        logger.warning("Invalid date range, adjusting...")
        end_date = start_date + timedelta(days=1)
    
    # Filtrar DataFrames por fechas - usar nombres correctos de columnas de BigQuery
    date_columns = {
        'df_items': 'ORDER_CREATE_DATE',
        'df_sessions': 'CAMPAIGN_DATE',  # Cambiado de 'EVENT_DATE' a 'CAMPAIGN_DATE'
        'df_ga_orders': 'CAMPAIGN_DATE', # Cambiado de 'Date' a 'CAMPAIGN_DATE'
        'df_ads': 'CAMPAIGN_DATE',       # Cambiado de 'Date' a 'CAMPAIGN_DATE'
        'df_ft_customers': 'CUSTOMER_CREATE_DATE'  # Cambiado de 'CUSTOMER_CREATION_DATE'
    }
    
    try:
        filtered_dataframes = filter_dataframes_by_date(dataframes, date_columns, start_date, end_date)
    except Exception as e:
        error_msg = f"Error filtering dataframes by date: {str(e)}"
        state.add_error_message(error_msg)
        logger.error(error_msg)
        # Return unfiltered data as fallback
        filtered_dataframes = dataframes
    
    logger.info("Dashboard setup completed successfully")
    
    return (filtered_dataframes['df_items'], filtered_dataframes['df_sessions'], 
            filtered_dataframes['df_ga_orders'], filtered_dataframes['df_ads'],
            start_date, end_date, df_items, df_sessions, df_ga_orders, df_ads, df_ft_customers)

# —————————————————————————————————————————
# FUNCIONES ESPECÍFICAS PARA TAB FRECUENCIA Y RECURRENCIA
# —————————————————————————————————————————

def analyze_frequency_metrics(df_items_filtered: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Calcula métricas clave de frecuencia y recurrencia."""
    if df_items_filtered.empty:
        # Devolver diccionario con todas las claves esperadas pero con valores None/0
        empty_metrics = {
            'avg_orders_per_customer': 0.0,
            'recurrent_customer_count': 0,
            'total_customer_count': 0,
            'percentage_recurrent': 0.0,
            'avg_time_between_purchases': None
        }
        return pd.DataFrame(), empty_metrics
    
    # Protección contra datasets muy grandes
    if len(df_items_filtered) > 500000:  # Más de 500k filas
        logger.warning(f"Large dataset detected ({len(df_items_filtered):,} rows), sampling for performance")
        # Tomar una muestra representativa para análisis de frecuencia
        df_sample = df_items_filtered.sample(n=min(250000, len(df_items_filtered)), random_state=42)
        st.info(f"ℹ️ **Dataset grande detectado**: Analizando muestra de {len(df_sample):,} registros de {len(df_items_filtered):,} totales para optimizar rendimiento.")
    else:
        df_sample = df_items_filtered
    
    customer_orders_freq = df_sample.groupby('USER_EMAIL')['ORDER_GID'].nunique().reset_index(name='NUM_ORDERS')
    
    # Métricas básicas
    avg_orders_per_customer = customer_orders_freq['NUM_ORDERS'].mean()
    
    # Clientes con múltiples órdenes
    multi_order_customers = customer_orders_freq[customer_orders_freq['NUM_ORDERS'] > 1]
    recurrent_customer_count = len(multi_order_customers)
    total_customer_count = len(customer_orders_freq)
    percentage_recurrent = (recurrent_customer_count / total_customer_count * 100) if total_customer_count > 0 else 0
    
    # Tiempo promedio entre compras
    avg_time_between_purchases = None
    if not multi_order_customers.empty:
        # Usar la misma muestra para consistencia
        df_multi_orders = df_sample[df_sample['USER_EMAIL'].isin(multi_order_customers['USER_EMAIL'])].copy()
        
        # Protección adicional para el cálculo de tiempo entre compras
        if len(df_multi_orders) > 100000:
            df_multi_orders = df_multi_orders.sample(n=100000, random_state=42)
            
        df_multi_orders.sort_values(['USER_EMAIL', 'ORDER_CREATE_DATE'], inplace=True)
        df_multi_orders['TIME_DIFF'] = df_multi_orders.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].diff().dt.days
        avg_time_between_purchases = df_multi_orders['TIME_DIFF'].mean()
    
    metrics = {
        'avg_orders_per_customer': avg_orders_per_customer,
        'recurrent_customer_count': recurrent_customer_count,
        'total_customer_count': total_customer_count,
        'percentage_recurrent': percentage_recurrent,
        'avg_time_between_purchases': avg_time_between_purchases
    }
    
    return customer_orders_freq, metrics

def create_frequency_metrics_display(metrics: Dict[str, float]) -> None:
    """Crea el display de métricas de frecuencia."""
    # Información de debugging para verificar datos
    recurrent_count = metrics.get('recurrent_customer_count', 0)
    total_count = metrics.get('total_customer_count', 0)
    percentage = metrics.get('percentage_recurrent', 0)
    
    metrics_data = [
        {
            'label': "👥 Total Clientes",
            'value': f"{int(total_count):,}",
            'help': "Número total de clientes únicos (`USER_EMAIL`) que realizaron compras dentro del período global seleccionado. Basado en datos de `SHOPIFY_ORDERS_LINEITEMS.csv`."
        },
        {
            'label': "🔁 Clientes Recurrentes",
            'value': f"{int(recurrent_count):,} ({percentage:.1f}%)",
            'help': f"Clientes únicos que realizaron MÁS DE UN PEDIDO dentro del período. {int(recurrent_count):,} de {int(total_count):,} clientes = {percentage:.1f}% de recurrencia."
        },
        {
            'label': "📦 Pedidos Promedio por Cliente",
            'value': f"{metrics['avg_orders_per_customer']:.2f}" if pd.notna(metrics['avg_orders_per_customer']) else "N/A",
            'help': "Calculado como el número total de pedidos únicos dividido por el número total de clientes únicos (`USER_EMAIL`) que realizaron compras. Todo dentro del período global seleccionado."
        },
        {
            'label': "⏱️ Tiempo Promedio entre Compras",
            'value': f"{metrics['avg_time_between_purchases']:.1f} días" if pd.notna(metrics['avg_time_between_purchases']) else "N/A",
            'help': "Promedio de días transcurridos entre compras consecutivas. Se calcula solo para clientes que han realizado más de un pedido dentro del período global seleccionado."
        }
    ]
    
    create_metric_layout(metrics_data, columns=4)
    
    # Añadir un mensaje informativo si hay pocos clientes recurrentes
    if total_count > 0:
        if percentage == 0:
            st.warning("⚠️ **No se detectaron clientes recurrentes en este período.** Considera expandir el rango de fechas o verificar los datos.")
        elif percentage < 5:
            st.info(f"ℹ️ **Baja recurrencia detectada ({percentage:.1f}%).** Solo {int(recurrent_count)} de {int(total_count)} clientes han realizado compras múltiples en este período.")
        else:
            st.success(f"✅ **Recurrencia detectada correctamente:** {int(recurrent_count)} clientes recurrentes de {int(total_count)} total ({percentage:.1f}%)")

def create_frequency_distribution_charts(customer_orders_freq: pd.DataFrame, df_items_filtered: pd.DataFrame) -> None:
    """Crea gráficos de distribución de frecuencia."""
    if customer_orders_freq.empty:
        st.caption("No hay datos para las distribuciones de frecuencia.")
        return
    
    # Gráfico 1: Distribución del Nº de Pedidos por Cliente
    bins_num_orders = [0, 1, 2, 3, 5, 10, float('inf')]
    labels_num_orders = ['1 Pedido', '2 Pedidos', '3 Pedidos', '4-5 Pedidos', '6-10 Pedidos', '11+ Pedidos']
    
    customer_orders_freq_copy = customer_orders_freq.copy()
    customer_orders_freq_copy['PEDIDOS_AGRUPADOS'] = pd.cut(
        customer_orders_freq_copy['NUM_ORDERS'],
        bins=bins_num_orders,
        labels=labels_num_orders,
        right=True,
        include_lowest=True
    )
    
    order_counts_grouped = customer_orders_freq_copy['PEDIDOS_AGRUPADOS'].value_counts().reset_index()
    order_counts_grouped.columns = ['Grupo de Pedidos', 'Número de Clientes']
    order_counts_grouped['Grupo de Pedidos'] = pd.Categorical(
        order_counts_grouped['Grupo de Pedidos'],
        categories=labels_num_orders,
        ordered=True
    )
    order_counts_grouped = order_counts_grouped.sort_values('Grupo de Pedidos')

    fig_dist_num_orders = create_bar_chart(
        order_counts_grouped,
        x="Grupo de Pedidos",
        y="Número de Clientes",
        title="Distribución del Nº de Pedidos por Cliente (Agrupado)<br><sub>Cuántos clientes (`USER_EMAIL`) caen en rangos de cantidad de pedidos realizados en el período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
        height=350
    )
    safe_plotly_chart(fig_dist_num_orders)
    
    # Gráfico 2: Distribución del Tiempo Entre Compras
    multi_order_customers = customer_orders_freq[customer_orders_freq['NUM_ORDERS'] > 1]['USER_EMAIL']
    if not multi_order_customers.empty:
        df_multi_orders = df_items_filtered[df_items_filtered['USER_EMAIL'].isin(multi_order_customers)].copy()
        df_multi_orders.sort_values(['USER_EMAIL', 'ORDER_CREATE_DATE'], inplace=True)
        df_multi_orders['TIME_DIFF'] = df_multi_orders.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].diff().dt.days
        
        if df_multi_orders['TIME_DIFF'].notna().any():
            fig_dist_time_between = px.histogram(
                df_multi_orders.dropna(subset=['TIME_DIFF']), 
                x="TIME_DIFF",
                title="Distribución del Tiempo Entre Compras (Días)<br><sub>Frecuencia de días entre compras consecutivas para clientes con >1 pedido en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
                labels={"TIME_DIFF": "Días entre Compras Consecutivas"}
            )
            fig_dist_time_between.update_layout(bargap=0.1, title_font_size=15, height=350)
            safe_plotly_chart(fig_dist_time_between)
        else:
            st.caption("No hay datos suficientes para la distribución del tiempo entre compras.")
    else:
        st.caption("No hay clientes con múltiples pedidos para análisis de tiempo entre compras.")

def create_frequency_segmentation_chart(customer_orders_freq: pd.DataFrame) -> None:
    """Crea gráfico de segmentación por frecuencia."""
    if customer_orders_freq.empty:
        st.caption("No hay datos para segmentar por frecuencia.")
        return
    
    customer_orders_copy = customer_orders_freq.copy()
    customer_orders_copy['SEGMENTO_FRECUENCIA'] = pd.cut(
        customer_orders_copy['NUM_ORDERS'], 
        bins=FREQUENCY_BINS, 
        labels=FREQUENCY_LABELS, 
        right=True
    )
    
    segment_counts = customer_orders_copy['SEGMENTO_FRECUENCIA'].value_counts().reset_index()
    segment_counts.columns = ['Segmento por Frecuencia', 'Número de Clientes']
    segment_counts = segment_counts.sort_values(
        by='Segmento por Frecuencia', 
        key=lambda x: x.map({label: i for i, label in enumerate(FREQUENCY_LABELS)})
    )

    fig_segment_freq = create_bar_chart(
        segment_counts,
        x="Segmento por Frecuencia",
        y="Número de Clientes",
        title="Segmentación de Clientes por Frecuencia de Pedidos<br><sub>Clientes en segmentos (Comprador Único, Ocasional, etc.) según nº de pedidos en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
        height=400
    )
    safe_plotly_chart(fig_segment_freq)

def analyze_cohort_retention(df_items: pd.DataFrame, df_items_filtered: pd.DataFrame) -> None:
    """Analiza la retención por cohortes."""
    st.markdown("#### 📅 Análisis de Cohortes (Retención Mensual)")
    st.markdown("""
    <div class='dashboard-subtext' style='font-size:0.88rem; line-height:1.4;'>
    Este análisis muestra qué porcentaje de clientes que hicieron su <i>primera compra global</i> en un mes específico ('cohorte')
    volvieron a comprar en los meses siguientes. La cohorte se define usando todos los datos históricos de órdenes, 
    y luego se observa su actividad de compra (retención) dentro del <b>período actualmente filtrado en el dashboard</b>.
    </div>
    """, unsafe_allow_html=True)
    
    # Preparar datos de cohorte
    df_items_copy_cohort = df_items.copy()
    df_items_copy_cohort['ORDER_MONTH'] = df_items_copy_cohort['ORDER_CREATE_DATE'].dt.to_period('M')
    df_items_copy_cohort['COHORT'] = df_items_copy_cohort.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].transform('min').dt.to_period('M')
    
    # Fusionar con actividad del período filtrado
    df_cohort_data_activity = pd.merge(
        df_items_filtered.copy(),
        df_items_copy_cohort[['USER_EMAIL', 'COHORT']].drop_duplicates(subset=['USER_EMAIL']),
        on='USER_EMAIL',
        how='left'
    )
    df_cohort_data_activity['ORDER_MONTH_ACTIVITY'] = df_cohort_data_activity['ORDER_CREATE_DATE'].dt.to_period('M')

    if df_cohort_data_activity.empty or 'COHORT' not in df_cohort_data_activity.columns:
        st.info("No hay datos suficientes para el análisis de cohortes.")
        return
    
    # Procesar datos de cohorte
    df_cohort_counts = df_cohort_data_activity.groupby(['COHORT', 'ORDER_MONTH_ACTIVITY']).agg(
        n_customers=('USER_EMAIL', 'nunique')
    ).reset_index()
    
    if df_cohort_counts.empty:
        st.info("No hay datos suficientes para el análisis de cohortes.")
        return
    
    df_cohort_counts['PERIOD_NUMBER'] = (df_cohort_counts['ORDER_MONTH_ACTIVITY'] - df_cohort_counts['COHORT']).apply(
        lambda x: x.n if pd.notnull(x) else -1
    )
    df_cohort_counts = df_cohort_counts[df_cohort_counts['PERIOD_NUMBER'] >= 0]

    cohort_pivot = df_cohort_counts.pivot_table(
        index='COHORT',
        columns='PERIOD_NUMBER',
        values='n_customers'
    )
    
    if cohort_pivot.empty:
        st.info("No hay suficientes datos para generar el análisis de cohortes.")
        return
    
    # Calcular matriz de retención
    cohort_size_df = df_items_copy_cohort.groupby('COHORT')['USER_EMAIL'].nunique().reset_index(name='TOTAL_CUSTOMERS_IN_COHORT')
    cohort_pivot_with_size = cohort_pivot.reset_index().merge(cohort_size_df, on='COHORT', how='left').set_index('COHORT')
    cohort_matrix = cohort_pivot_with_size.iloc[:, :-1].divide(cohort_pivot_with_size['TOTAL_CUSTOMERS_IN_COHORT'], axis=0)
    
    if cohort_matrix.empty:
        st.info("No se pudo generar la matriz de retención de cohortes.")
        return
    
    # Crear gráfico de retención promedio
    avg_retention_curve = cohort_matrix.mean(axis=0)
    avg_retention_curve.index = avg_retention_curve.index.astype(int)
    avg_retention_curve = avg_retention_curve.sort_index()
    avg_retention_df = avg_retention_curve.reset_index()
    avg_retention_df.columns = ['Meses Desde Primera Compra', 'Tasa de Retención Promedio']
    avg_retention_df['Tasa de Retención Promedio'] = avg_retention_df['Tasa de Retención Promedio'] * 100

    # Descripción del gráfico
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

    fig_avg_retention = create_line_chart(
        avg_retention_df,
        x='Meses Desde Primera Compra',
        y='Tasa de Retención Promedio',
        title="Curva de Retención Promedio Global",
        labels={'Tasa de Retención Promedio': 'Tasa de Retención Promedio (%)'},
        height=400
    )
    fig_avg_retention.update_layout(yaxis_ticksuffix="%", xaxis_dtick=1)
    safe_plotly_chart(fig_avg_retention)

def analyze_recurrence_by_sport_and_level(df_items_filtered: pd.DataFrame) -> None:
    """Analiza la recurrencia por deporte y nivel de jugador."""
    st.markdown("#### 🎯 Análisis de Recurrencia por Deporte y Nivel")
    
    # Ejecutar análisis de recompra
    customer_orders, repurchase_analysis, first_purchases = analyze_repurchase_behavior(df_items_filtered)
    repurchase_times = calculate_time_to_repurchase(df_items_filtered)
    
    if customer_orders.empty:
        st.info("No hay suficientes datos para el análisis avanzado de recurrencia.")
        return
    
    col_sport_rec1, col_sport_rec2 = st.columns(2)
    
    with col_sport_rec1:
        st.markdown("##### 🏓 Recurrencia por Deporte")
        if 'FIRST_SPORT' in customer_orders.columns:
            sport_recurrence = customer_orders.groupby('FIRST_SPORT').agg({
                'USER_EMAIL': 'count',
                'TOTAL_ORDERS': lambda x: (x > 1).sum()
            }).reset_index()
            sport_recurrence.columns = ['DEPORTE', 'TOTAL_CLIENTES', 'CLIENTES_RECURRENTES']
            sport_recurrence['TASA_RECURRENCIA'] = (sport_recurrence['CLIENTES_RECURRENTES'] / sport_recurrence['TOTAL_CLIENTES'] * 100).round(1)
            
            if not sport_recurrence.empty:
                fig_sport_rec = create_bar_chart(
                    sport_recurrence,
                    x='DEPORTE',
                    y='TASA_RECURRENCIA',
                    title="Tasa de Recurrencia por Deporte<br><sub>Porcentaje de clientes que realizaron más de una compra, por deporte de primera compra.</sub>",
                    labels={'DEPORTE': 'Deporte', 'TASA_RECURRENCIA': 'Tasa de Recurrencia (%)'},
                    color='TASA_RECURRENCIA',
                    color_scale='Blues',
                    height=350
                )
                safe_plotly_chart(fig_sport_rec)
                st.dataframe(sport_recurrence, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de deporte disponibles.")
    
    with col_sport_rec2:
        st.markdown("##### 🏆 Recurrencia por Nivel de Jugador")
        if 'FIRST_LEVEL' in customer_orders.columns:
            level_recurrence = customer_orders.groupby('FIRST_LEVEL').agg({
                'USER_EMAIL': 'count',
                'TOTAL_ORDERS': lambda x: (x > 1).sum()
            }).reset_index()
            level_recurrence.columns = ['NIVEL', 'TOTAL_CLIENTES', 'CLIENTES_RECURRENTES']
            level_recurrence['TASA_RECURRENCIA'] = (level_recurrence['CLIENTES_RECURRENTES'] / level_recurrence['TOTAL_CLIENTES'] * 100).round(1)
            
            if not level_recurrence.empty:
                fig_level_rec = create_bar_chart(
                    level_recurrence,
                    x='NIVEL',
                    y='TASA_RECURRENCIA',
                    title="Tasa de Recurrencia por Nivel de Jugador<br><sub>Porcentaje de clientes que realizaron más de una compra, por nivel inferido de primera compra.</sub>",
                    labels={'NIVEL': 'Nivel de Jugador', 'TASA_RECURRENCIA': 'Tasa de Recurrencia (%)'},
                    color='TASA_RECURRENCIA',
                    color_scale='Blues',
                    height=350
                )
                safe_plotly_chart(fig_level_rec)
                st.dataframe(level_recurrence, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de nivel de jugador disponibles.")

    # Tiempo hasta recompra por deporte
    if not repurchase_times.empty:
        st.markdown("##### ⏰ Tiempo Promedio hasta Recompra")
        
        col_time_sport, col_time_category = st.columns(2)
        
        with col_time_sport:
            time_by_sport = repurchase_times.groupby('FIRST_SPORT')['DAYS_TO_REPURCHASE'].agg(['mean', 'median', 'count']).reset_index()
            time_by_sport.columns = ['DEPORTE', 'PROMEDIO_DIAS', 'MEDIANA_DIAS', 'NUM_CLIENTES']
            
            if not time_by_sport.empty:
                fig_time_sport = create_bar_chart(
                    time_by_sport,
                    x='DEPORTE',
                    y='PROMEDIO_DIAS',
                    title="Días Promedio hasta Recompra por Deporte<br><sub>Tiempo promedio entre primera y segunda compra.</sub>",
                    labels={'DEPORTE': 'Deporte', 'PROMEDIO_DIAS': 'Días Promedio'},
                    text='PROMEDIO_DIAS',
                    height=350
                )
                fig_time_sport.update_traces(texttemplate='%{text:.0f}', textposition='inside', textfont=dict(color='white', size=11))
                safe_plotly_chart(fig_time_sport)
        
        with col_time_category:
            time_by_category = repurchase_times.groupby('FIRST_CATEGORY')['DAYS_TO_REPURCHASE'].agg(['mean', 'count']).reset_index()
            time_by_category.columns = ['CATEGORIA', 'PROMEDIO_DIAS', 'NUM_CLIENTES']
            time_by_category = time_by_category.sort_values('PROMEDIO_DIAS').head(8)
            
            if not time_by_category.empty:
                fig_time_cat = create_bar_chart(
                    time_by_category,
                    x='CATEGORIA',
                    y='PROMEDIO_DIAS',
                    title="Días Promedio hasta Recompra por Categoría<br><sub>Top 8 categorías por tiempo de recompra.</sub>",
                    labels={'CATEGORIA': 'Categoría', 'PROMEDIO_DIAS': 'Días Promedio'},
                    text='PROMEDIO_DIAS',
                    height=350
                )
                fig_time_cat.update_traces(texttemplate='%{text:.0f}', textposition='inside', textfont=dict(color='white', size=11))
                fig_time_cat.update_xaxes(tickangle=45)
                fig_time_cat.update_layout(margin=dict(b=100))
                safe_plotly_chart(fig_time_cat)

# —————————————————————————————————————————
# FUNCIONES ESPECÍFICAS PARA TAB VALOR DEL CLIENTE
# —————————————————————————————————————————

def calculate_ltv_data(df_items_filtered: pd.DataFrame) -> pd.DataFrame:
    """Calcula datos de LTV (Lifetime Value) básicos."""
    if df_items_filtered.empty:
        return pd.DataFrame()
    
    return df_items_filtered.groupby('USER_EMAIL')['ORDER_TOTAL_PRICE'].sum().reset_index(name='LTV')

def display_basic_ltv_metrics(ltv_data: pd.DataFrame) -> None:
    """Muestra métricas básicas de LTV."""
    if ltv_data.empty:
        st.info("No hay datos de LTV disponibles.")
        return
    
    avg_ltv = ltv_data['LTV'].mean()
    st.metric(
        "LTV Promedio (en período)", 
        f"${avg_ltv:,.2f}" if pd.notna(avg_ltv) else "N/A",
        help="Valor de Vida del Cliente promedio. Se calcula sumando el `ORDER_TOTAL_PRICE` de todos los pedidos para cada cliente (`USER_EMAIL`) dentro del período global seleccionado, y luego promediando estas sumas entre todos los clientes. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`."
    )

def analyze_ltv_by_acquisition_channel(ltv_data: pd.DataFrame, df_items: pd.DataFrame, df_ga_orders: pd.DataFrame) -> None:
    """Analiza LTV por canal de adquisición."""
    st.markdown("#### 📊 LTV Promedio por Canal de Adquisición (Primer Pedido)")
    
    if ltv_data.empty or df_items.empty or df_ga_orders.empty:
        st.info("Datos de LTV, pedidos o GA Orders no disponibles para este análisis.")
        return
    
    # Determinar la columna de email correcta - puede ser USER_EMAIL o CUSTOMER_EMAIL
    email_col = None
    if 'USER_EMAIL' in ltv_data.columns and 'USER_EMAIL' in df_items.columns:
        email_col = 'USER_EMAIL'
    elif 'CUSTOMER_EMAIL' in ltv_data.columns and 'CUSTOMER_EMAIL' in df_items.columns:
        email_col = 'CUSTOMER_EMAIL'
    
    if email_col is None:
        st.warning("Columnas de email necesarias no disponibles para el análisis de LTV por canal.")
        return
    
    # Obtener información de primer pedido
    try:
        first_order_info = df_items.loc[df_items.groupby(email_col)['ORDER_CREATE_DATE'].idxmin()]
        if first_order_info.empty:
            st.info("No hay datos de primer pedido disponibles.")
            return
    except Exception as e:
        st.warning(f"Error al obtener información del primer pedido: {str(e)}")
        return
    
    # Verificar que las columnas necesarias para el merge existan
    merge_col_left = 'ORDER_ID' if 'ORDER_ID' in first_order_info.columns else 'ORDER_GID'
    merge_col_right = 'TRANSACTION_ID' if 'TRANSACTION_ID' in df_ga_orders.columns else 'Transaction_ID'
    
    if merge_col_left not in first_order_info.columns or merge_col_right not in df_ga_orders.columns:
        st.info("Columnas necesarias para unir órdenes con datos de GA no están disponibles.")
        return
    
    merged_first_orders = pd.merge(
        first_order_info, 
        df_ga_orders,
        left_on=merge_col_left, 
        right_on=merge_col_right,
        how='left'
    )
    
    # Verificar que la columna de canal exista
    channel_col = None
    for col in ['CAMPAIGN_PRIMARY_GROUP', 'Session_primary_channel_group', 'campaign_primary_group']:
        if col in merged_first_orders.columns:
            channel_col = col
            break
    
    if merged_first_orders.empty or channel_col is None:
        st.info("No se pudo determinar el canal de adquisición para LTV. Datos de canal no disponibles.")
        return
    
    # Fusionar con datos de LTV
    ltv_with_channel = pd.merge(
        ltv_data,
        merged_first_orders[[email_col, channel_col]].drop_duplicates(subset=[email_col]),
        on=email_col,
        how='left'
    )
    
    if ltv_with_channel.empty or channel_col not in ltv_with_channel.columns:
        st.info("No se pudo calcular LTV por canal de adquisición (datos de canal o LTV insuficientes).")
        return
    
    # Calcular LTV promedio por canal
    ltv_by_channel = ltv_with_channel.groupby(channel_col)['LTV'].mean().reset_index()
    ltv_by_channel = ltv_by_channel.sort_values('LTV', ascending=False)
    
    if ltv_by_channel.empty:
        st.info("No hay datos suficientes para mostrar LTV por canal de adquisición.")
        return
    
    fig_ltv_channel = create_bar_chart(
        ltv_by_channel,
        x=channel_col,
        y='LTV',
        title="LTV Promedio por Canal de Adquisición del Cliente (1ª Compra Global)<br><sub>LTV (gasto total en período global) promediado por canal de 1ª compra global. Canales de GA, LTV de SHOPIFY.</sub>",
        labels={channel_col: 'Canal de Adquisición', 'LTV': 'LTV Promedio ($)'}
    )
    safe_plotly_chart(fig_ltv_channel)

def create_ltv_segmentation(ltv_data: pd.DataFrame) -> None:
    """Crea segmentación de clientes por LTV."""
    st.markdown("#### 🏆 Segmentación de Clientes por Ranking de Facturación (en período)")
    
    if ltv_data.empty:
        st.info("No hay datos de LTV para segmentar clientes.")
        return
    
    ltv_data_sorted = ltv_data.sort_values('LTV', ascending=False)
    
    # Crear segmentos por cuartiles
    try:
        ltv_data_sorted['RANK'] = pd.qcut(
            ltv_data_sorted['LTV'], 
            q=4, 
            labels=["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"], 
            duplicates='drop'
        )
    except ValueError:
        ltv_data_sorted['RANK'] = pd.cut(
            ltv_data_sorted['LTV'], 
            bins=4, 
            labels=["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"], 
            duplicates='drop', 
            include_lowest=True
        )

    segment_summary = ltv_data_sorted.groupby('RANK', observed=False).agg(
        num_customers=('USER_EMAIL', 'count'),
        total_revenue=('LTV', 'sum'),
        avg_revenue_per_customer=('LTV', 'mean')
    ).reset_index()
    
    st.write("Segmentación de Clientes por Facturación (LTV en período global):")
    st.dataframe(segment_summary)
    st.markdown(
        "<div class='dashboard-subtext' style='font-size:0.8rem; text-align:center; margin-top:-5px;'>"
        "Los clientes se agrupan en cuatro categorías (Alto, Medio-Alto, Medio-Bajo, Bajo) según su gasto total (`LTV`) durante el período global seleccionado. "
        "LTV se calcula como la suma de `ORDER_TOTAL_PRICE` por cliente desde `SHOPIFY_ORDERS_LINEITEMS.csv`. "
        "La tabla muestra el número de clientes, los ingresos totales y el ingreso promedio por cliente para cada segmento."
        "</div>", 
        unsafe_allow_html=True
    )

def analyze_ltv_by_frequency_segment(ltv_data: pd.DataFrame, df_items_filtered: pd.DataFrame) -> None:
    """Analiza LTV por segmento de frecuencia de compra."""
    st.markdown("#### 💎 LTV Promedio por Segmento de Frecuencia de Compra")
    
    if ltv_data.empty or df_items_filtered.empty:
        st.info("Datos de frecuencia de clientes o LTV no disponibles para este análisis.")
        return
    
    # Calcular frecuencia de pedidos
    customer_orders_freq = df_items_filtered.groupby('USER_EMAIL')['ORDER_GID'].nunique().reset_index(name='NUM_ORDERS')
    customer_orders_freq['SEGMENTO_FRECUENCIA'] = pd.cut(
        customer_orders_freq['NUM_ORDERS'],
        bins=FREQUENCY_BINS,
        labels=FREQUENCY_LABELS,
        right=True
    )
    
    # Fusionar con datos de LTV
    ltv_with_frequency = pd.merge(
        ltv_data, 
        customer_orders_freq[['USER_EMAIL', 'SEGMENTO_FRECUENCIA']], 
        on='USER_EMAIL', 
        how='left'
    )
    
    if ltv_with_frequency.empty or 'SEGMENTO_FRECUENCIA' not in ltv_with_frequency.columns or not ltv_with_frequency['SEGMENTO_FRECUENCIA'].notna().any():
        st.info("No se pudo calcular el LTV por segmento de frecuencia (datos insuficientes o segmentos no aplicables).")
        return
    
    # Calcular LTV promedio por segmento
    avg_ltv_by_freq_segment = ltv_with_frequency.groupby('SEGMENTO_FRECUENCIA', observed=False)['LTV'].mean().reset_index()
    avg_ltv_by_freq_segment = avg_ltv_by_freq_segment.sort_values(
        by='SEGMENTO_FRECUENCIA', 
        key=lambda x: x.map({label: i for i, label in enumerate(FREQUENCY_LABELS)})
    )

    fig_ltv_by_freq = create_bar_chart(
        avg_ltv_by_freq_segment,
        x='SEGMENTO_FRECUENCIA',
        y='LTV',
        title="LTV Promedio por Segmento de Frecuencia de Compra<br><sub>LTV promedio (gasto total en período global) por segmento de frecuencia de compra (Nº pedidos en mismo período). Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
        labels={'SEGMENTO_FRECUENCIA': 'Segmento por Frecuencia', 'LTV': 'LTV Promedio ($)'},
        height=400
    )
    safe_plotly_chart(fig_ltv_by_freq)

def analyze_aov_trends(df_items_filtered: pd.DataFrame) -> None:
    """Analiza tendencias del AOV (Average Order Value)."""
    st.markdown("#### 📈 Tendencias del AOV (Valor Promedio del Pedido)")
    
    if df_items_filtered.empty or df_items_filtered['ORDER_GID'].nunique() == 0:
        st.info("No hay órdenes en el período seleccionado para calcular AOV.")
        return
    
    # Calcular AOV mensual
    aov_trend = df_items_filtered.groupby(pd.Grouper(key='ORDER_CREATE_DATE', freq='M')).agg(
        total_revenue=('ORDER_TOTAL_PRICE', 'sum'),
        total_orders=('ORDER_GID', 'nunique')
    ).reset_index()
    
    # Evitar división por cero y no inventar datos
    aov_trend['AOV'] = safe_division(
        aov_trend['total_revenue'], 
        aov_trend['total_orders'], 
        default=None  # No inventar AOV cuando no hay órdenes
    )
    aov_trend = aov_trend.dropna(subset=['AOV'])
    
    if aov_trend.empty:
        st.info("No hay datos suficientes para mostrar la tendencia del AOV.")
        return
    
    fig_aov = create_line_chart(
        aov_trend,
        x='ORDER_CREATE_DATE',
        y='AOV',
        title="Tendencia Mensual del AOV (Valor Promedio del Pedido)<br><sub>AOV mensual = Ingresos totales del mes / Nº de pedidos únicos del mes. Período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
        labels={'ORDER_CREATE_DATE': 'Mes', 'AOV': 'AOV ($)'}
    )
    safe_plotly_chart(fig_aov)

def analyze_ltv_by_sport(ltv_data: pd.DataFrame, df_items_filtered: pd.DataFrame) -> None:
    """Analiza LTV segmentado por deporte."""
    st.markdown("#### 🏓 LTV Promedio por Deporte")
    
    if ltv_data.empty or df_items_filtered.empty:
        st.info("Datos de LTV o de deportes no disponibles para este análisis.")
        return
    
    try:
        # Obtener primera compra de cada cliente con categorización
        df_enriched = enrich_dataframe_with_categories(df_items_filtered)
        first_purchases = df_enriched.loc[df_enriched.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].idxmin()]
        
        # Fusionar LTV con deporte de primera compra
        ltv_with_sport = pd.merge(
            ltv_data, 
            first_purchases[['USER_EMAIL', 'SPORT_GROUPED']].drop_duplicates(subset=['USER_EMAIL']), 
            on='USER_EMAIL', 
            how='left'
        )
        
        if ltv_with_sport.empty or 'SPORT_GROUPED' not in ltv_with_sport.columns:
            st.info("No se pudo calcular el LTV por deporte (datos insuficientes).")
            return
        
        # Calcular LTV promedio por deporte
        ltv_by_sport = ltv_with_sport.groupby('SPORT_GROUPED').agg({
            'LTV': ['mean', 'count', 'sum']
        }).round(2)
        
        ltv_by_sport.columns = ['LTV_Promedio', 'Num_Clientes', 'LTV_Total']
        ltv_by_sport = ltv_by_sport.reset_index()
        ltv_by_sport = ltv_by_sport.sort_values('LTV_Promedio', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not ltv_by_sport.empty:
                fig_ltv_sport = create_bar_chart(
                    ltv_by_sport,
                    x='SPORT_GROUPED',
                    y='LTV_Promedio',
                    title="LTV Promedio por Deporte<br><sub>LTV promedio (gasto total en período global) por deporte de primera compra. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
                    labels={'SPORT_GROUPED': 'Deporte', 'LTV_Promedio': 'LTV Promedio ($)'},
                    color='LTV_Promedio',
                    color_scale='Blues',
                    text='LTV_Promedio',
                    height=350
                )
                fig_ltv_sport.update_traces(texttemplate='$%{text:.0f}', textposition='inside', textfont=dict(color='white', size=11))
                safe_plotly_chart(fig_ltv_sport)
        
        with col2:
            st.markdown("**Resumen por Deporte:**")
            ltv_by_sport['LTV_Promedio_Formatted'] = ltv_by_sport['LTV_Promedio'].apply(format_currency)
            ltv_by_sport['LTV_Total_Formatted'] = ltv_by_sport['LTV_Total'].apply(format_currency)
            
            st.dataframe(
                ltv_by_sport[['SPORT_GROUPED', 'LTV_Promedio_Formatted', 'Num_Clientes', 'LTV_Total_Formatted']],
                column_config={
                    "SPORT_GROUPED": "Deporte",
                    "LTV_Promedio_Formatted": "LTV Promedio",
                    "Num_Clientes": "# Clientes",
                    "LTV_Total_Formatted": "LTV Total"
                },
                hide_index=True,
                use_container_width=True
            )
            
    except Exception as e:
        logger.error(f"Error analyzing LTV by sport: {str(e)}")
        st.error("Error al analizar LTV por deporte.")

def analyze_ltv_cac_ratio_by_channel(ltv_data: pd.DataFrame, df_items: pd.DataFrame, 
                                   df_ga_orders: pd.DataFrame, df_ads_filtered: pd.DataFrame) -> None:
    """Analiza la relación LTV/CAC por canal para evaluar inversiones."""
    st.markdown("#### 💰 Análisis LTV vs CAC por Canal de Adquisición")
    
    if ltv_data.empty or df_ga_orders.empty or df_ads_filtered.empty:
        st.info("Datos insuficientes para análisis LTV/CAC (requiere datos de LTV, GA Orders y Ads).")
        return
    
    try:
        # 1. Get LTV with acquisition channel from ga_orders
        first_order_info = df_items.loc[df_items.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].idxmin()]
        merged_first_orders = pd.merge(
            first_order_info, 
            df_ga_orders,
            left_on='ORDER_GID_STR', 
            right_on='Transaction_ID',
            how='left'
        )
        
        if merged_first_orders.empty or 'Session_primary_channel_group' not in merged_first_orders.columns:
            st.warning("No se pudo determinar el canal de adquisición para análisis LTV/CAC.")
            return
        
        ltv_with_channel = pd.merge(
            ltv_data,
            merged_first_orders[['USER_EMAIL', 'Session_primary_channel_group']].drop_duplicates(subset=['USER_EMAIL']),
            on='USER_EMAIL',
            how='left'
        )
        
        ltv_by_channel_detailed = ltv_with_channel.groupby('Session_primary_channel_group').agg({
            'LTV': ['mean', 'count']
        }).round(2)
        
        ltv_by_channel_detailed.columns = ['LTV_Promedio', 'Num_Clientes']
        ltv_by_channel_detailed = ltv_by_channel_detailed.reset_index()
        
        # 2. Get CAC by joining Ads and Orders data to get channel group
        if 'Ads_cost' in df_ads_filtered.columns and 'Campaign_ID' in df_ads_filtered.columns:
            
            # Aggregate ad costs by Campaign_ID
            ads_cost_agg = df_ads_filtered.groupby('Campaign_ID')['Ads_cost'].sum().reset_index()

            # Get channel group and users from ga_orders (transactions)
            channel_agg = df_ga_orders.groupby(['Campaign_ID', 'Session_primary_channel_group']).agg(
                Total_users=('Transaction_ID', 'nunique') # Using transactions as a proxy for users from a channel
            ).reset_index()

            # Merge to get costs and channels together
            cac_by_channel = pd.merge(
                ads_cost_agg,
                channel_agg,
                on='Campaign_ID',
                how='inner'
            )

            # Sum metrics by the final channel group
            final_cac_agg = cac_by_channel.groupby('Session_primary_channel_group').agg({
                'Ads_cost': 'sum',
                'Total_users': 'sum'
            }).reset_index()
            
            final_cac_agg['CAC'] = safe_division(final_cac_agg['Ads_cost'], final_cac_agg['Total_users'], default=None)
            final_cac_agg = final_cac_agg.dropna(subset=['CAC'])
            
            # 3. Combine LTV and CAC
            ltv_cac_analysis = pd.merge(
                ltv_by_channel_detailed,
                final_cac_agg[['Session_primary_channel_group', 'CAC']],
                on='Session_primary_channel_group',
                how='inner'
            )
            
            if not ltv_cac_analysis.empty:
                # Calcular ratio LTV/CAC
                ltv_cac_analysis['LTV_CAC_Ratio'] = safe_division(
                    ltv_cac_analysis['LTV_Promedio'], 
                    ltv_cac_analysis['CAC'],
                    default=None  # No inventar ratios cuando faltan datos
                )
                
                # Filtrar filas con datos válidos para análisis
                ltv_cac_analysis = ltv_cac_analysis.dropna(subset=['LTV_CAC_Ratio'])
                
                # Clasificar inversiones
                ltv_cac_analysis['Investment_Quality'] = ltv_cac_analysis['LTV_CAC_Ratio'].apply(
                    lambda x: 'Excelente (>3)' if x > 3 
                    else 'Buena (1.5-3)' if x > 1.5 
                    else 'Marginal (1-1.5)' if x > 1 
                    else 'Pérdida (<1)'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de LTV vs CAC
                    fig_ltv_cac = px.scatter(
                        ltv_cac_analysis,
                        x='CAC',
                        y='LTV_Promedio',
                        size='Num_Clientes',
                        color='LTV_CAC_Ratio',
                        hover_name='Session_primary_channel_group',
                        title="LTV vs CAC por Canal<br><sub>Tamaño = Número de clientes, Color = Ratio LTV/CAC</sub>",
                        labels={'CAC': 'CAC ($)', 'LTV_Promedio': 'LTV Promedio ($)'},
                        color_continuous_scale='RdYlGn'
                    )
                    
                    # Agregar línea diagonal (LTV = CAC)
                    max_val = max(ltv_cac_analysis['CAC'].max(), ltv_cac_analysis['LTV_Promedio'].max())
                    fig_ltv_cac.add_shape(
                        type="line",
                        x0=0, y0=0, x1=max_val, y1=max_val,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    fig_ltv_cac.add_annotation(
                        x=max_val*0.7, y=max_val*0.8,
                        text="Línea de equilibrio<br>(LTV = CAC)",
                        showarrow=False,
                        font=dict(color="red", size=10)
                    )
                    
                    safe_plotly_chart(fig_ltv_cac)
                
                with col2:
                    # Tabla resumen
                    st.markdown("**Análisis de Inversión por Canal:**")
                    
                    summary_table = ltv_cac_analysis.copy()
                    summary_table['LTV_Promedio_Formatted'] = summary_table['LTV_Promedio'].apply(format_currency)
                    summary_table['CAC_Formatted'] = summary_table['CAC'].apply(format_currency)
                    summary_table['LTV_CAC_Ratio_Formatted'] = summary_table['LTV_CAC_Ratio'].apply(lambda x: f"{x:.1f}x")
                    
                    st.dataframe(
                        summary_table[['Session_primary_channel_group', 'LTV_Promedio_Formatted', 'CAC_Formatted', 'LTV_CAC_Ratio_Formatted', 'Investment_Quality']],
                        column_config={
                            "Session_primary_channel_group": "Canal",
                            "LTV_Promedio_Formatted": "LTV Promedio",
                            "CAC_Formatted": "CAC",
                            "LTV_CAC_Ratio_Formatted": "Ratio LTV/CAC",
                            "Investment_Quality": "Calidad Inversión"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Métricas clave
                    best_channel = summary_table.loc[summary_table['LTV_CAC_Ratio'].idxmax()]
                    st.metric(
                        "🏆 Mejor Canal (Ratio LTV/CAC)",
                        best_channel['Session_primary_channel_group'],
                        delta=f"{best_channel['LTV_CAC_Ratio']:.1f}x",
                        help="Canal con mayor retorno de inversión basado en la relación LTV/CAC"
                    )
            else:
                st.info("No se pudieron combinar datos de LTV y CAC por canal.")
                
        else:
            st.warning("Faltan columnas necesarias en datos de anuncios para calcular CAC (Ads_cost, Campaign_ID).")
            
    except Exception as e:
        logger.error(f"Error analyzing LTV/CAC ratio by channel: {str(e)}")
        st.error("Error al analizar la relación LTV/CAC por canal.")

def display_data_loading_status(data: Dict[str, pd.DataFrame]) -> None:
    """Muestra el estado de carga de los DataFrames en una tabla."""
    st.dataframe(
        pd.DataFrame([
            {
                "Dataset": name,
                "Filas Cargadas": len(df) if df is not None else 0,
                "Estado": "✅ Cargado" if df is not None and not df.empty else "⚠️ Vacío o no cargado"
            }
            for name, df in data.items()
        ]),
        hide_index=True,
        use_container_width=True
    )

def analyze_missing_columns(df_items, df_ft_customers, df_sessions, df_ga_orders, df_ads):
    """Analiza automáticamente las columnas faltantes en los DataFrames para el diagnóstico."""
    with st.expander("Verificar columnas faltantes para análisis avanzado"):
        # Columnas esperadas basadas en BigQuery - nombres reales de las tablas RAW_
        expected = {
            'SHOPIFY_ORDERS_LINEITEMS': ['ORDER_GID', 'CUSTOMER_ID', 'ORDER_CREATE_DATE', 'PRODUCT_NAME', 'LINEITEM_QTY', 'PRICE'],
            'SHOPIFY_CUSTOMERS': ['CUSTOMER_ID', 'CUSTOMER_EMAIL'],
            'GA_METRICS': ['CAMPAIGN_DATE', 'CAMPAIGN_USERS', 'CAMPAIGN_SESSIONS', 'CAMPAIGN_BOUNCE_RATE', 'CAMPAIGN_AVG_SESSION_DURATION'],
            'GA_TRANSACTIONS': ['TRANSACTION_ID', 'CAMPAIGN_DATE', 'CAMPAIGN_PRIMARY_GROUP'],
            'GA_ADS': ['CAMPAIGN_DATE', 'CAMPAIGN_ID', 'CAMPAIGN_COST', 'CAMPAIGN_CLICKS', 'CAMPAIGN_IMPRESSIONS']
        }
        
        # DataFrames actuales
        dfs = {
            'SHOPIFY_ORDERS_LINEITEMS': df_items,
            'SHOPIFY_CUSTOMERS': df_ft_customers,
            'GA_METRICS': df_sessions,
            'GA_TRANSACTIONS': df_ga_orders,
            'GA_ADS': df_ads
        }

        # Comprobar y mostrar estado
        for name, req_cols in expected.items():
            df = dfs.get(name)
            if df is None or df.empty:
                st.warning(f"Dataset '{name}' no disponible o vacío.")
                continue

            missing = [col for col in req_cols if col not in df.columns]
            available_cols = list(df.columns) if not df.empty else []
            
            if not missing:
                st.success(f"✅ {name}: Todas las columnas clave están presentes.")
            else:
                st.error(f"❌ {name}: Faltan columnas esperadas: {', '.join(missing)}")
                
            # Mostrar columnas disponibles para diagnóstico
            if available_cols:
                st.info(f"📋 {name} - Columnas disponibles: {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}")

# —————————————————————————————————————————
# FUNCIONES ESPECÍFICAS PARA TAB COMPORTAMIENTO DE COMPRA
# —————————————————————————————————————————

def analyze_top_products_recurrent_customers(df_items_filtered: pd.DataFrame) -> None:
    """Analiza los productos más comprados por clientes recurrentes."""
    st.markdown("#### 🛍️ Productos Más Comprados por Clientes Recurrentes (en período)")
    
    if df_items_filtered.empty:
        st.info("No hay datos para analizar productos de clientes recurrentes.")
        return
    
    customer_order_counts = df_items_filtered.groupby('USER_EMAIL')['ORDER_GID'].nunique()
    recurrent_customers = customer_order_counts[customer_order_counts > 1].index
    
    if recurrent_customers.empty:
        st.info("No hay clientes recurrentes en el período seleccionado para este análisis.")
        return
    
    df_recurrent_purchases = df_items_filtered[df_items_filtered['USER_EMAIL'].isin(recurrent_customers)]
    
    if df_recurrent_purchases.empty:
        st.info("No hay compras de clientes recurrentes en el período seleccionado.")
        return
    
    top_products_recurrent = df_recurrent_purchases.groupby('PRODUCT_NAME')['LINEITEM_QTY'].sum().nlargest(10).reset_index()
    
    fig_top_recurrent = create_bar_chart(
        top_products_recurrent,
        x='PRODUCT_NAME',
        y='LINEITEM_QTY',
        title="Top 10 Productos por Clientes Recurrentes (Cantidad Total de Unidades)<br><sub>Productos más comprados (suma de `LINEITEM_QTY`) por clientes con >1 pedido en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
    )
    safe_plotly_chart(fig_top_recurrent)

def analyze_cross_sell_patterns(df_items_filtered: pd.DataFrame) -> None:
    """Analiza patrones de cross-sell y bundles."""
    st.markdown("#### 🔗 Cross-sell y Bundles (Pares de Productos Más Comunes en el Mismo Pedido)")
    
    if df_items_filtered.empty or 'ORDER_GID' not in df_items_filtered.columns or 'PRODUCT_NAME' not in df_items_filtered.columns:
        st.warning("Datos insuficientes para análisis de cross-sell.")
        return
    
    order_item_counts = df_items_filtered.groupby('ORDER_GID')['PRODUCT_NAME'].nunique()
    multi_item_orders = order_item_counts[order_item_counts > 1].index
    
    if multi_item_orders.empty:
        st.info("No hay pedidos con múltiples productos diferentes en el período seleccionado para análisis de cross-sell.")
        return
    
    df_multi_item_orders = df_items_filtered[df_items_filtered['ORDER_GID'].isin(multi_item_orders)]
    
    frequent_pairs = {}
    for order_gid, group in df_multi_item_orders.groupby('ORDER_GID'):
        products_in_order = sorted(list(set(group['PRODUCT_NAME'])))
        if len(products_in_order) >= 2:
            for pair in combinations(products_in_order, 2):
                frequent_pairs[pair] = frequent_pairs.get(pair, 0) + 1
    
    if not frequent_pairs:
        st.info("No se encontraron pares de productos comprados juntos con frecuencia en el período.")
        return
    
    top_pairs_df = pd.DataFrame(list(frequent_pairs.items()), columns=['Product_Pair', 'Frequency'])
    top_pairs_df = top_pairs_df.sort_values('Frequency', ascending=False).head(10)
    top_pairs_df['Product_Pair_Display'] = top_pairs_df['Product_Pair'].apply(lambda x: f"{x[0]} & {x[1]}")
    
    fig_pairs = create_bar_chart(
        top_pairs_df,
        x='Product_Pair_Display',
        y='Frequency',
        title="Top 10 Pares de Productos Comprados Juntos (Frecuencia en Pedidos)<br><sub>Pares de productos distintos que más frecuentemente aparecen juntos en el mismo pedido (`ORDER_GID`) en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
        labels={'Product_Pair_Display': 'Par de Productos', 'Frequency': 'Frecuencia'}
    )
    safe_plotly_chart(fig_pairs)

def analyze_seasonality(df_items_filtered: pd.DataFrame) -> None:
    """Analiza la estacionalidad de las compras."""
    st.markdown("#### ☀️ Estacionalidad de Compras")
    
    if df_items_filtered.empty:
        st.info("No hay datos para analizar estacionalidad.")
        return
    
    df_seasonal = df_items_filtered.copy()
    df_seasonal['MONTH'] = df_seasonal['ORDER_CREATE_DATE'].dt.strftime('%B')
    df_seasonal['DAY_OF_WEEK'] = df_seasonal['ORDER_CREATE_DATE'].dt.day_name()
    
    sales_by_month = df_seasonal.groupby('MONTH')['ORDER_TOTAL_PRICE'].sum().reindex(config.MONTHS_ORDER).reset_index()
    sales_by_day = df_seasonal.groupby('DAY_OF_WEEK')['ORDER_TOTAL_PRICE'].sum().reindex(config.DAYS_ORDER).reset_index()

    col1, col2 = st.columns(2)
    
    with col1:
        if not sales_by_month.empty:
            fig_month = create_bar_chart(
                sales_by_month,
                x='MONTH',
                y='ORDER_TOTAL_PRICE',
                title="Estacionalidad: Ingresos Totales por Mes<br><sub>Suma de `ORDER_TOTAL_PRICE` por mes, en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
                labels={'MONTH': 'Mes', 'ORDER_TOTAL_PRICE': 'Ingresos Totales ($)'}
            )
            safe_plotly_chart(fig_month)
        else:
            st.info("No hay datos de ventas mensuales en el período.")
    
    with col2:
        if not sales_by_day.empty:
            fig_day = create_bar_chart(
                sales_by_day,
                x='DAY_OF_WEEK',
                y='ORDER_TOTAL_PRICE',
                title="Estacionalidad: Ingresos Totales por Día de la Semana<br><sub>Suma de `ORDER_TOTAL_PRICE` por día de la semana, en período global. Fuente: `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>",
                labels={'DAY_OF_WEEK': 'Día de la Semana', 'ORDER_TOTAL_PRICE': 'Ingresos Totales ($)'}
            )
            safe_plotly_chart(fig_day)
        else:
            st.info("No hay datos de ventas por día de la semana en el período.")

def create_category_level_correlation_heatmap(analysis_data: pd.DataFrame) -> None:
    """Crea el heatmap de correlación entre categoría y nivel."""
    correlation_matrix = pd.crosstab(
        analysis_data['FIRST_CATEGORY'], 
        analysis_data['FIRST_LEVEL'],
        normalize='index'
    ) * 100
    
    if correlation_matrix.empty:
        st.info("No hay datos suficientes para crear el heatmap de correlación.")
        return
    
    # Reorganizar columnas por orden lógico de nivel
    available_levels = [level for level in config.LEVEL_ORDER if level in correlation_matrix.columns]
    correlation_matrix = correlation_matrix.reindex(columns=available_levels, fill_value=0)
    
    col_viz1, col_viz2 = st.columns([3, 2])
    
    with col_viz1:
        fig_correlation = create_correlation_heatmap(
            analysis_data,
            x_col='FIRST_LEVEL',
            y_col='FIRST_CATEGORY',
            title="Distribución de Nivel de Jugador por Categoría<br><sub>Porcentaje de cada nivel dentro de cada categoría de primera compra. Colores más intensos = mayor concentración.</sub>"
        )
        safe_plotly_chart(fig_correlation)
    
    with col_viz2:
        st.markdown("###### 🎯 Insights por Categoría:")
        insights_df = []
        for category in correlation_matrix.index:
            if correlation_matrix.loc[category].sum() > 0:
                max_level = correlation_matrix.loc[category].idxmax()
                max_percentage = correlation_matrix.loc[category].max()
                total_clients = analysis_data[analysis_data['FIRST_CATEGORY'] == category]['USER_EMAIL'].nunique()
                insights_df.append({
                    'Categoría': category,
                    'Nivel Dominante': max_level,
                    'Porcentaje': f"{max_percentage:.1f}%",
                    'Total Clientes': total_clients
                })
        
        if insights_df:
            insights_table = pd.DataFrame(insights_df)
            st.dataframe(
                insights_table,
                column_config={
                    "Categoría": st.column_config.TextColumn("Categoría", width="medium"),
                    "Nivel Dominante": st.column_config.TextColumn("Nivel Dominante", width="small"),
                    "Porcentaje": st.column_config.TextColumn("%", width="small"),
                    "Total Clientes": st.column_config.NumberColumn("Clientes", width="small")
                },
                hide_index=True,
                use_container_width=True
            )

def create_sport_analysis_tabs(analysis_data: pd.DataFrame, available_levels: List[str]) -> None:
    """Crea tabs de análisis detallado por deporte."""
    st.markdown("###### 🏓 Análisis Detallado por Deporte")
    
    sport_tabs = st.tabs(["🏓 Pádel", "🏸 Pickleball", "🎾 Otros/Genérico", "📊 Comparativo"])
    
    sports_data = {
        'Pádel': analysis_data[analysis_data['DEPORTE_AGRUPADO'] == 'Pádel'],
        'Pickleball': analysis_data[analysis_data['DEPORTE_AGRUPADO'] == 'Pickleball'],
        'Otros/Genérico': analysis_data[analysis_data['DEPORTE_AGRUPADO'] == 'Otros/Genérico']
    }
    
    color_schemes = {
        'Pádel': ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78'],
        'Pickleball': ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'],
        'Otros/Genérico': ['#1e40af', '#3b82f6', '#60a5fa', '#93c5fd']
    }
    
    # Tabs individuales por deporte
    for i, (sport_name, sport_data) in enumerate(sports_data.items()):
        with sport_tabs[i]:
            if not sport_data.empty:
                sport_matrix = pd.crosstab(sport_data['FIRST_CATEGORY'], sport_data['FIRST_LEVEL'], normalize='index') * 100
                sport_matrix = sport_matrix.reindex(columns=available_levels, fill_value=0)
                
                sport_long = sport_matrix.reset_index().melt(id_vars='FIRST_CATEGORY', var_name='Nivel', value_name='Porcentaje')
                sport_long = sport_long[sport_long['Porcentaje'] > 0]
                
                if not sport_long.empty:
                    fig_sport = px.bar(
                        sport_long,
                        x='FIRST_CATEGORY',
                        y='Porcentaje',
                        color='Nivel',
                        title=f"Distribución de Niveles en {sport_name} por Categoría",
                        labels={'FIRST_CATEGORY': 'Categoría de Producto', 'Porcentaje': 'Porcentaje (%)'},
                        color_discrete_sequence=color_schemes[sport_name]
                    )
                    fig_sport.update_layout(height=400, xaxis_tickangle=45)
                    safe_plotly_chart(fig_sport)
                    
                    st.markdown(f"**Total clientes {sport_name}:** {sport_data['USER_EMAIL'].nunique():,}")
                else:
                    st.info(f"No hay suficientes datos de {sport_name} para mostrar distribución.")
            else:
                st.info(f"No hay datos de clientes de {sport_name} en el período seleccionado.")
    
    # Tab comparativo
    with sport_tabs[3]:
        comparative_data = []
        
        for sport in ['Pádel', 'Pickleball', 'Otros/Genérico']:
            sport_subset = analysis_data[analysis_data['DEPORTE_AGRUPADO'] == sport]
            if not sport_subset.empty:
                level_dist = sport_subset['FIRST_LEVEL'].value_counts(normalize=True) * 100
                for level, percentage in level_dist.items():
                    comparative_data.append({
                        'Deporte': sport,
                        'Nivel': level,
                        'Porcentaje': percentage,
                        'Clientes': sport_subset[sport_subset['FIRST_LEVEL'] == level]['USER_EMAIL'].nunique()
                    })
        
        if comparative_data:
            comp_df = pd.DataFrame(comparative_data)
            
            fig_comp = px.bar(
                comp_df,
                x='Deporte',
                y='Porcentaje',
                color='Nivel',
                title="Comparación de Distribución de Niveles entre Deportes",
                labels={'Porcentaje': 'Porcentaje (%)', 'Deporte': 'Tipo de Deporte'},
                text='Clientes',
                color_discrete_sequence=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
            )
            fig_comp.update_traces(texttemplate='%{text}', textposition='inside', textfont=dict(color='white', size=10))
            fig_comp.update_layout(height=450, font=dict(size=11))
            safe_plotly_chart(fig_comp)
            
            # Tabla resumen comparativa
            summary_pivot = comp_df.pivot_table(
                index='Deporte', 
                columns='Nivel', 
                values='Clientes', 
                fill_value=0
            )
            summary_pivot = summary_pivot.reindex(columns=available_levels, fill_value=0)
            summary_pivot['Total'] = summary_pivot.sum(axis=1)
            
            st.markdown("**Resumen de Clientes por Deporte y Nivel:**")
            st.dataframe(summary_pivot, use_container_width=True)
        else:
            st.info("No hay datos suficientes para comparación entre deportes.")

def analyze_category_migration(repurchase_analysis: pd.DataFrame) -> None:
    """Analiza la migración de categorías en recompras."""
    st.markdown("##### 🔄 Análisis de Migración de Categorías en Recompras")
    
    if repurchase_analysis.empty:
        st.info("No hay datos suficientes para analizar migración de categorías.")
        return
    
    customer_purchase_sequence = []
    
    for email in repurchase_analysis['USER_EMAIL'].unique():
        customer_data = repurchase_analysis[repurchase_analysis['USER_EMAIL'] == email].sort_values('ORDER_CREATE_DATE')
        
        if len(customer_data) >= 2:
            first_category = customer_data.iloc[0]['PRODUCT_CATEGORY']
            second_category = customer_data.iloc[1]['PRODUCT_CATEGORY']
            first_sport = customer_data.iloc[0]['SPORT_UNIVERSE']
            
            customer_purchase_sequence.append({
                'USER_EMAIL': email,
                'PRIMERA_CATEGORIA': first_category,
                'SEGUNDA_CATEGORIA': second_category,
                'DEPORTE': first_sport,
                'MISMA_CATEGORIA': first_category == second_category
            })
    
    if not customer_purchase_sequence:
        st.info("No hay suficientes datos de migración de categorías.")
        return
    
    migration_df = pd.DataFrame(customer_purchase_sequence)
    
    col_mig1, col_mig2 = st.columns(2)
    
    with col_mig1:
        # Métricas de migración
        loyalty_rate = migration_df['MISMA_CATEGORIA'].mean() * 100
        
        st.metric(
            label="🎯 Lealtad a Categoría",
            value=f"{loyalty_rate:.1f}%",
            help="Porcentaje de clientes que compran la misma categoría de producto en su primera recompra."
        )
        
        # Análisis por deporte
        sport_loyalty = migration_df.groupby('DEPORTE')['MISMA_CATEGORIA'].mean() * 100
        if not sport_loyalty.empty:
            for sport, loyalty in sport_loyalty.items():
                st.markdown(f"**{sport}**: {loyalty:.1f}% lealtad")
    
    with col_mig2:
        # Migración más común
        migration_patterns = migration_df.groupby(['PRIMERA_CATEGORIA', 'SEGUNDA_CATEGORIA']).size().reset_index(name='FRECUENCIA')
        migration_patterns = migration_patterns.sort_values('FRECUENCIA', ascending=False)
        
        if not migration_patterns.empty:
            top_migration = migration_patterns.iloc[0]
            st.metric(
                label="🔄 Migración Más Común",
                value=f"{top_migration['PRIMERA_CATEGORIA']} → {top_migration['SEGUNDA_CATEGORIA']}",
                delta=f"{top_migration['FRECUENCIA']} clientes",
                help="Patrón de migración de categoría más frecuente entre primera compra y primera recompra."
            )
            
            # Top 5 migraciones
            st.markdown("###### Top 5 Migraciones:")
            for i, row in migration_patterns.head(5).iterrows():
                st.markdown(f"• {row['PRIMERA_CATEGORIA']} → {row['SEGUNDA_CATEGORIA']}: {row['FRECUENCIA']} clientes")

    # Gráfico de flujo de categorías
    if len(migration_patterns) > 0:
        top_migrations = migration_patterns.head(8)
        top_migrations['PATRON'] = top_migrations['PRIMERA_CATEGORIA'] + ' → ' + top_migrations['SEGUNDA_CATEGORIA']
        
        fig_migration = create_bar_chart(
            top_migrations,
            x='FRECUENCIA',
            y='PATRON',
            title="Patrones de Migración de Categorías (Top 8)<br><sub>Flujo más común de categorías entre primera compra y primera recompra.</sub>",
            labels={'PATRON': 'Patrón de Migración', 'FRECUENCIA': 'Número de Clientes'},
            height=400
        )
        fig_migration.update_layout(margin=dict(l=150))
        fig_migration.update_traces(orientation='h')
        safe_plotly_chart(fig_migration)

def analyze_category_behavior(customer_orders: pd.DataFrame) -> None:
    """Analiza el comportamiento por categoría de producto."""
    st.markdown("##### 📊 Comportamiento por Categoría de Producto")
    
    if customer_orders.empty:
        st.info("No hay datos para analizar comportamiento por categoría.")
        return
    
    category_behavior = customer_orders.groupby('FIRST_CATEGORY').agg({
        'USER_EMAIL': 'count',
        'TOTAL_ORDERS': ['mean', lambda x: (x > 1).sum()],
        'FIRST_SPORT': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'
    }).round(2)
    
    category_behavior.columns = ['Total_Clientes', 'Avg_Orders', 'Recurrent_Customers', 'Deporte_Dominante']
    category_behavior['Tasa_Recurrencia'] = (category_behavior['Recurrent_Customers'] / category_behavior['Total_Clientes'] * 100).round(1)
    category_behavior = category_behavior.reset_index()
    category_behavior = category_behavior.sort_values('Tasa_Recurrencia', ascending=False)
    
    if not category_behavior.empty:
        st.dataframe(
            category_behavior[['FIRST_CATEGORY', 'Total_Clientes', 'Avg_Orders', 'Tasa_Recurrencia', 'Deporte_Dominante']],
            column_config={
                "FIRST_CATEGORY": "Categoría",
                "Total_Clientes": "Total Clientes",
                "Avg_Orders": "Pedidos Promedio",
                "Tasa_Recurrencia": st.column_config.NumberColumn("Tasa Recurrencia (%)", format="%.1f%%"),
                "Deporte_Dominante": "Deporte Principal"
            },
            hide_index=True,
            use_container_width=True
        )

def analyze_advanced_category_sports_analysis(df_items_filtered: pd.DataFrame) -> None:
    """Ejecuta el análisis avanzado completo por categorías y deportes."""
    st.markdown("#### 🎯 Análisis Avanzado por Categorías y Deportes")
    
    # Ejecutar análisis de recompra para obtener datos categorizados
    customer_orders, repurchase_analysis, first_purchases = analyze_repurchase_behavior(df_items_filtered)
    
    if customer_orders.empty:
        st.info("No hay suficientes datos para el análisis avanzado de categorías.")
        return
    
    st.markdown("##### 🏆 Correlación: Categoría de Primera Compra vs Nivel de Jugador")
    
    required_columns = ['FIRST_CATEGORY', 'FIRST_LEVEL', 'FIRST_SPORT']
    if not all(col in customer_orders.columns for col in required_columns):
        st.warning("Faltan columnas necesarias para el análisis de correlación (FIRST_CATEGORY, FIRST_LEVEL, o FIRST_SPORT).")
        return
    
    # Preparar datos para análisis multi-dimensional
    analysis_data = customer_orders[required_columns + ['USER_EMAIL']].copy()
    analysis_data['DEPORTE_AGRUPADO'] = analysis_data['FIRST_SPORT'].apply(group_sports)
    
    # Visualización 1: Heatmap General
    create_category_level_correlation_heatmap(analysis_data)
    
    # Visualización 2: Análisis por Deporte
    available_levels = [level for level in config.LEVEL_ORDER if level in analysis_data['FIRST_LEVEL'].unique()]
    create_sport_analysis_tabs(analysis_data, available_levels)
    
    # Análisis de migración de categorías
    analyze_category_migration(repurchase_analysis)
    
    # Comportamiento por categoría
    analyze_category_behavior(customer_orders)

# —————————————————————————————————————————
# CONFIGURACIÓN PRINCIPAL DEL DASHBOARD
# —————————————————————————————————————————

# —————————————————————————————————————————
# FUNCIONES ESPECÍFICAS DE ANÁLISIS POR TAB
# —————————————————————————————————————————

def analyze_customer_overview(df_items_for_tab_vision: pd.DataFrame, df_items: pd.DataFrame, 
                            period_info: Dict[str, str]) -> None:
    """Analiza la visión general del cliente, incluyendo nuevos vs recurrentes."""
    log_function_call("analyze_customer_overview", period=period_info['period_str'])
    
    # NEW VS RECURRENT
    client = get_bigquery_client()
    first_purchase_dates_all_time = None

    if not client:
        st.warning("Conexión a BigQuery no disponible. Las métricas de nuevos vs recurrentes pueden ser imprecisas.", icon="⚠️")
        # Fallback to using the loaded df_items which is likely date-filtered
        first_purchase_dates_all_time = df_items.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].min()
    else:
        # Fetch true first purchase dates from the entire history
        first_purchase_dates_all_time = get_all_time_first_purchase_dates(client)
        if first_purchase_dates_all_time is None:
            st.warning("No se pudieron obtener las fechas de primera compra históricas. Usando datos limitados.", icon="⚠️")
            first_purchase_dates_all_time = df_items.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].min()

    # TOTALS - ALL TIME
    total_customers_all_time = len(first_purchase_dates_all_time) if first_purchase_dates_all_time is not None else df_items['USER_EMAIL'].nunique()
    st.metric(
        label="👥 Total Clientes Históricos (Global)",
        value=f"{total_customers_all_time:,}",
        help="Número total de clientes únicos en toda la historia de los datos."
    )

    st.markdown("##### Resumen para el período filtrado")
    st.markdown(f"<p class='dashboard-subtext' style='font-size:0.88rem;'>Análisis basado en el rango de fechas seleccionado en el sidebar: {period_info['start_date']} - {period_info['end_date']}</p>", unsafe_allow_html=True)
    
    if df_items_for_tab_vision.empty:
        st.warning("No hay datos de clientes para el período seleccionado.")
        return
    
    # METRICS FOR THE FILTERED PERIOD
    total_customers_period = df_items_for_tab_vision['USER_EMAIL'].nunique()
    
    # Identify new customers WITHIN THE FILTERED PERIOD
    # A customer is new if their first-ever purchase falls within the selected date range
    customers_in_period = df_items_for_tab_vision['USER_EMAIL'].unique()
    
    # Filter the all-time list to only the customers present in the current period
    first_purchases_of_customers_in_period = first_purchase_dates_all_time[
        first_purchase_dates_all_time.index.isin(customers_in_period)
    ]
    
    new_customers_mask = first_purchases_of_customers_in_period.between(
        period_info['start_date_dt'], 
        period_info['end_date_dt']
    )
    
    actual_new_customers_in_period = new_customers_mask.sum()
    recurrent_customers_period = total_customers_period - actual_new_customers_in_period
    
    # Display metrics
    metrics_data = [
        {"label": "Total Clientes Únicos", "value": f"{total_customers_period:,}"},
        {"label": "Clientes Nuevos", "value": f"{actual_new_customers_in_period:,}"},
        {"label": "Clientes Recurrentes", "value": f"{recurrent_customers_period:,}"},
        {
            "label": "% Clientes Nuevos", 
            "value": format_percentage(safe_division(actual_new_customers_in_period, total_customers_period) * 100)
        },
        {
            "label": "% Clientes Recurrentes", 
            "value": format_percentage(safe_division(recurrent_customers_period, total_customers_period) * 100)
        }
    ]
    create_metric_layout(metrics_data, columns=5)

    st.markdown("---")
    
    # Gráficos de evolución
    create_evolution_charts(df_items_for_tab_vision, period_info, first_purchase_dates_all_time)

def create_evolution_charts(df_items_for_tab_vision: pd.DataFrame, period_info: Dict[str, str],
                          first_purchase_dates_all_time: pd.Series) -> None:
    """Crea gráficos de evolución mensual."""
    st.markdown("##### Evolución Mensual")
    
    # Gráfico de evolución de clientes únicos mensuales
    monthly_unique_tab = df_items_for_tab_vision.groupby(
        pd.Grouper(key='ORDER_CREATE_DATE', freq='M')
    )['USER_EMAIL'].nunique().reset_index()
    monthly_unique_tab.rename(columns={'USER_EMAIL': 'Clientes Únicos Mensuales', 'ORDER_CREATE_DATE': 'Mes'}, inplace=True)
    
    if not monthly_unique_tab.empty:
        title = f"Evolución de Clientes Únicos Mensuales<br><sub style='font-size:0.75em;'>Período: {period_info['subtitle']}. Conteo de clientes (`USER_EMAIL`) únicos con compras por mes desde `SHOPIFY_ORDERS_LINEITEMS.csv`.</sub>"
        fig_evo_unique = create_line_chart(
            monthly_unique_tab, 'Mes', 'Clientes Únicos Mensuales', title, height=350
        )
        safe_plotly_chart(fig_evo_unique)
    else:
        st.caption("No hay datos de evolución de clientes únicos.")
    
    # Gráfico de evolución nuevos vs recurrentes
    create_new_vs_recurrent_evolution_chart(df_items_for_tab_vision, period_info, first_purchase_dates_all_time)

def create_new_vs_recurrent_evolution_chart(df_items_for_tab_vision: pd.DataFrame, 
                                          period_info: Dict[str, str],
                                          first_purchase_dates_all_time: pd.Series) -> None:
    """Crea gráfico de evolución de nuevos vs recurrentes."""
    df_evolution_source_tab = df_items_for_tab_vision.merge(
        first_purchase_dates_all_time.reset_index(), on='USER_EMAIL', how='left'
    )
    
    if df_evolution_source_tab.empty or 'FIRST_PURCHASE_DATE_GLOBAL' not in df_evolution_source_tab.columns:
        st.caption("Datos insuficientes para evolución nuevos vs recurrentes.")
        return
    
    df_evolution_source_tab['ORDER_MONTH_PERIOD'] = df_evolution_source_tab['ORDER_CREATE_DATE'].dt.to_period('M')
    df_evolution_source_tab['FIRST_PURCHASE_MONTH_PERIOD_GLOBAL'] = pd.to_datetime(df_evolution_source_tab['FIRST_PURCHASE_DATE_GLOBAL']).dt.to_period('M')
    df_evolution_source_tab['CUSTOMER_TYPE_FOR_MONTH'] = np.select(
        [df_evolution_source_tab['ORDER_MONTH_PERIOD'] == df_evolution_source_tab['FIRST_PURCHASE_MONTH_PERIOD_GLOBAL'],
         df_evolution_source_tab['ORDER_MONTH_PERIOD'] > df_evolution_source_tab['FIRST_PURCHASE_MONTH_PERIOD_GLOBAL']],
        ['Nuevo en Mes', 'Recurrente en Mes'], default='Indeterminado'
    )
    
    df_evo_classified_tab = df_evolution_source_tab[df_evolution_source_tab['CUSTOMER_TYPE_FOR_MONTH'] != 'Indeterminado']
    
    if df_evo_classified_tab.empty:
        st.caption("No se pudieron clasificar clientes para evolución nuevos vs recurrentes.")
        return
    
    monthly_counts_tab = df_evo_classified_tab.groupby(
        ['ORDER_MONTH_PERIOD', 'CUSTOMER_TYPE_FOR_MONTH']
    )['USER_EMAIL'].nunique().reset_index()
    monthly_counts_tab.rename(columns={'ORDER_MONTH_PERIOD': 'Mes', 'CUSTOMER_TYPE_FOR_MONTH': 'Tipo de Cliente', 'USER_EMAIL': 'Número de Clientes'}, inplace=True)
    monthly_counts_tab['Mes'] = monthly_counts_tab['Mes'].dt.to_timestamp()
    
    if not monthly_counts_tab.empty:
        title = f"Evolución Mensual: Nuevos vs. Recurrentes<br><sub style='font-size:0.75em;'>Período: {period_info['subtitle']}. 'Nuevo en Mes': 1ª compra global en ese mes. 'Recurrente': 1ª compra global anterior.</sub>"
        
        fig_stacked_bar_tab = px.bar(
            monthly_counts_tab, x='Mes', y='Número de Clientes', color='Tipo de Cliente', 
            title=title, barmode='stack'
        )
        fig_stacked_bar_tab.update_layout(title_font_size=15, margin=dict(t=60, b=20, l=20, r=20), height=350, legend_title_text='Tipo Cliente')
        safe_plotly_chart(fig_stacked_bar_tab)

def analyze_geographic_distribution(df_items_for_tab_vision: pd.DataFrame, period_info: Dict[str, str]) -> None:
    """Analiza la distribución geográfica de los pedidos."""
    # Función removida - no hay datos de SHIPPING_COUNTRY disponibles en BigQuery
    pass

def create_us_analysis(df_items_for_tab_vision: pd.DataFrame, period_info: Dict[str, str]) -> None:
    """Crea análisis específico de Estados Unidos con mapa interactivo de estados."""
    # Función removida - no hay datos de SHIPPING_COUNTRY disponibles en BigQuery
    pass

def create_customer_segments(rfm_data: pd.DataFrame) -> pd.DataFrame:
    """Crea segmentos de clientes basados en métricas RFM."""
    if rfm_data.empty or 'RFM_SEGMENT' not in rfm_data.columns:
        return pd.DataFrame()
    
    def categorize_customer(row):
        if pd.isna(row['R_SCORE']) or pd.isna(row['F_SCORE']) or pd.isna(row['M_SCORE']):
            return 'Sin Clasificar'
        
        r, f, m = int(row['R_SCORE']), int(row['F_SCORE']), int(row['M_SCORE'])
        
        if r >= 3 and f >= 3 and m >= 3:
            return 'Champions'
        elif r >= 3 and f >= 2:
            return 'Loyal Customers'
        elif r >= 3 and m >= 3:
            return 'Potential Loyalists'
        elif r >= 3:
            return 'New Customers'
        elif f >= 3 and m >= 3:
            return 'Cannot Lose Them'
        elif f >= 2:
            return 'At Risk'
        elif m >= 3:
            return 'Need Attention'
        else:
            return 'Lost Customers'
    
    rfm_data['CUSTOMER_SEGMENT'] = rfm_data.apply(categorize_customer, axis=1)
    return rfm_data

def analyze_customer_lifetime_value(df_items: pd.DataFrame) -> pd.DataFrame:
    """Analiza el valor de vida del cliente (CLV)."""
    if df_items.empty:
        return pd.DataFrame()
    
    clv_data = df_items.groupby('USER_EMAIL').agg({
        'ORDER_CREATE_DATE': ['min', 'max', 'count'],
        'PRICE': 'sum',
        'ORDER_GID': 'nunique'
    }).reset_index()
    
    # Aplanar columnas multi-nivel
    clv_data.columns = ['USER_EMAIL', 'FIRST_PURCHASE', 'LAST_PURCHASE', 'PURCHASE_COUNT', 'TOTAL_SPENT', 'ORDER_COUNT']
    
    # Calcular días como cliente
    clv_data['DAYS_AS_CUSTOMER'] = (clv_data['LAST_PURCHASE'] - clv_data['FIRST_PURCHASE']).dt.days + 1
    clv_data['DAYS_AS_CUSTOMER'] = clv_data['DAYS_AS_CUSTOMER'].clip(lower=1)
    
    # Métricas de CLV
    clv_data['AVG_ORDER_VALUE'] = clv_data['TOTAL_SPENT'] / clv_data['ORDER_COUNT']
    clv_data['PURCHASE_FREQUENCY'] = clv_data['PURCHASE_COUNT'] / clv_data['DAYS_AS_CUSTOMER'] * 365  # Anualizada
    clv_data['PREDICTED_CLV'] = clv_data['AVG_ORDER_VALUE'] * clv_data['PURCHASE_FREQUENCY'] * 2  # 2 años
    
    return clv_data

def analyze_customer_acquisition_rate_by_channel(df_ads_filtered: pd.DataFrame) -> None:
    """Analiza la tasa de adquisición de clientes por canal en formato tabla."""
    st.markdown("#### 📊 Customer Acquisition Rate por Canal")
    
    if df_ads_filtered.empty:
        st.info("No hay datos de anuncios para analizar tasa de adquisición.")
        return
    
    try:
        # Verificar columnas necesarias (usando Total_users que sí existe en BigQuery)
        required_columns = ['Primary_channel_group', 'Total_users', 'Ads_cost']
        if not all(col in df_ads_filtered.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df_ads_filtered.columns]
            st.warning(f"Faltan columnas necesarias para Customer Acquisition Rate: {missing_cols}")
            
            # Mostrar tabla de diagnóstico de columnas disponibles
            st.markdown("**🔍 Columnas disponibles en los datos:**")
            available_cols = list(df_ads_filtered.columns)
            cols_info = pd.DataFrame({
                'Columna': available_cols,
                'Tipo': [str(df_ads_filtered[col].dtype) for col in available_cols],
                'Valores No Nulos': [df_ads_filtered[col].notna().sum() for col in available_cols]
            })
            st.dataframe(cols_info, hide_index=True, use_container_width=True)
            return
        
        # Limpiar datos antes del análisis
        df_clean = df_ads_filtered.copy()
        
        # Eliminar filas con valores nulos en columnas críticas
        df_clean = df_clean.dropna(subset=['Primary_channel_group'])
        
        # Convertir columnas numéricas y manejar errores
        for col in ['Total_users', 'Ads_cost']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Para columnas opcionales, llenar con 0 si no existen o tienen valores nulos
        optional_cols = ['Ads_impressions', 'Ads_clicks']
        for col in optional_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0
            else:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        if df_clean.empty:
            st.warning("No hay datos válidos después de la limpieza.")
            return
        
        # Tabla 1: Resumen completo de todos los canales
        st.markdown("##### 📊 Resumen Completo de Todos los Canales")
        
        all_channels_summary = df_clean.groupby('Primary_channel_group').agg({
            'Total_users': 'sum',
            'Ads_cost': 'sum',
            'Ads_impressions': 'sum',
            'Ads_clicks': 'sum'
        }).reset_index()
        
        # Calcular métricas básicas para todos los canales
        all_channels_summary['Total_users'] = all_channels_summary['Total_users'].astype(int)
        all_channels_summary['Ads_cost'] = all_channels_summary['Ads_cost'].round(2)
        all_channels_summary['Ads_impressions'] = all_channels_summary['Ads_impressions'].astype(int)
        all_channels_summary['Ads_clicks'] = all_channels_summary['Ads_clicks'].astype(int)
        
        # Formatear para mostrar
        all_channels_display = all_channels_summary.copy()
        all_channels_display['Ads_cost_formatted'] = all_channels_display['Ads_cost'].apply(format_currency)
        
        # Configuración de columnas para la tabla completa
        all_channels_config = {
            "Primary_channel_group": st.column_config.TextColumn("Canal", width="medium"),
            "Total_users": st.column_config.NumberColumn("Total Usuarios", width="small"),
            "Ads_cost_formatted": st.column_config.TextColumn("Costo Total", width="small"),
            "Ads_impressions": st.column_config.NumberColumn("Impresiones", width="small"),
            "Ads_clicks": st.column_config.NumberColumn("Clics", width="small")
        }
        
        st.dataframe(
            all_channels_display[['Primary_channel_group', 'Total_users', 'Ads_cost_formatted', 'Ads_impressions', 'Ads_clicks']],
            column_config=all_channels_config,
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Tabla 2: Análisis avanzado de Customer Acquisition Rate
        st.markdown("##### 🎯 Análisis Avanzado de Customer Acquisition Rate")
        
        # Calcular métricas por canal solo para canales con usuarios > 0
        acquisition_metrics = df_clean[df_clean['Total_users'] > 0].groupby('Primary_channel_group').agg({
            'Total_users': 'sum',
            'Ads_cost': 'sum',
            'Ads_impressions': 'sum',
            'Ads_clicks': 'sum'
        }).reset_index()
        
        if acquisition_metrics.empty:
            st.info("No hay canales con usuarios adquiridos para análisis avanzado.")
            return
    
        # Calcular métricas derivadas con validación
        acquisition_metrics['CAC'] = acquisition_metrics.apply(
            lambda row: safe_division(row['Ads_cost'], row['Total_users'], default=0), axis=1
        )
        
        acquisition_metrics['CTR'] = acquisition_metrics.apply(
            lambda row: safe_division(row['Ads_clicks'], row['Ads_impressions'], default=0) * 100, axis=1
        )
        
        acquisition_metrics['Conversion_Rate'] = acquisition_metrics.apply(
            lambda row: safe_division(row['Total_users'], row['Ads_clicks'], default=0) * 100, axis=1
        )
        
        # Calcular acquisition rate (usuarios por cada $1,000 invertidos)
        acquisition_metrics['Acquisition_Rate'] = acquisition_metrics.apply(
            lambda row: safe_division(row['Total_users'], row['Ads_cost'] / 1000, default=0), axis=1
        )
        
        # Calcular eficiencia del canal
        total_users = acquisition_metrics['Total_users'].sum()
        total_cost = acquisition_metrics['Ads_cost'].sum()
        
        if total_users > 0 and total_cost > 0:
            acquisition_metrics['User_Share'] = acquisition_metrics['Total_users'] / total_users * 100
            acquisition_metrics['Cost_Share'] = acquisition_metrics['Ads_cost'] / total_cost * 100
            acquisition_metrics['Efficiency_Index'] = acquisition_metrics.apply(
                lambda row: safe_division(row['User_Share'], row['Cost_Share'], default=0), axis=1
            )
        else:
            acquisition_metrics['User_Share'] = 0
            acquisition_metrics['Cost_Share'] = 0
            acquisition_metrics['Efficiency_Index'] = 0
        
        # Ordenar por eficiencia
        acquisition_metrics = acquisition_metrics.sort_values('Efficiency_Index', ascending=False)
        
        # Crear tabla formateada para mostrar
        display_table = acquisition_metrics.copy()
        display_table['Total_users'] = display_table['Total_users'].astype(int)
        display_table['Ads_cost_formatted'] = display_table['Ads_cost'].apply(format_currency)
        display_table['CAC_formatted'] = display_table['CAC'].apply(format_currency)
        display_table['CTR_formatted'] = display_table['CTR'].apply(lambda x: f"{x:.2f}%")
        display_table['Conversion_Rate_formatted'] = display_table['Conversion_Rate'].apply(lambda x: f"{x:.2f}%")
        display_table['Acquisition_Rate_formatted'] = display_table['Acquisition_Rate'].apply(lambda x: f"{x:.1f}")
        display_table['User_Share_formatted'] = display_table['User_Share'].apply(lambda x: f"{x:.1f}%")
        display_table['Cost_Share_formatted'] = display_table['Cost_Share'].apply(lambda x: f"{x:.1f}%")
        display_table['Efficiency_Index_formatted'] = display_table['Efficiency_Index'].apply(lambda x: f"{x:.2f}")
        
        # Mostrar tabla principal
        columns_to_show = ['Primary_channel_group', 'Total_users', 'Ads_cost_formatted', 'CAC_formatted', 
                          'CTR_formatted', 'Conversion_Rate_formatted', 'Acquisition_Rate_formatted', 
                          'User_Share_formatted', 'Cost_Share_formatted', 'Efficiency_Index_formatted']
        
        column_config = {
            "Primary_channel_group": st.column_config.TextColumn("Canal", width="medium"),
            "Total_users": st.column_config.NumberColumn("Total Usuarios", width="small"),
            "Ads_cost_formatted": st.column_config.TextColumn("Costo Total", width="small"),
            "CAC_formatted": st.column_config.TextColumn("CAC", width="small"),
            "CTR_formatted": st.column_config.TextColumn("CTR", width="small", help="Click-Through Rate: Clics / Impresiones"),
            "Conversion_Rate_formatted": st.column_config.TextColumn("Conv. Rate", width="small", help="Total usuarios / Clics"),
            "Acquisition_Rate_formatted": st.column_config.TextColumn("Usuarios/K$", width="small", help="Total usuarios por cada $1,000 invertidos"),
            "User_Share_formatted": st.column_config.TextColumn("% Usuarios", width="small"),
            "Cost_Share_formatted": st.column_config.TextColumn("% Costo", width="small"),
            "Efficiency_Index_formatted": st.column_config.TextColumn("Índice Eficiencia", width="small", help="Ratio de % usuarios / % costo. >1 = más eficiente")
        }
        
        st.dataframe(
                display_table[columns_to_show],
                column_config=column_config,
            hide_index=True,
            use_container_width=True
        )

            # Métricas clave
        col1, col2, col3 = st.columns(3)
            
        if not display_table.empty:
                with col1:
                    best_efficiency = display_table.iloc[0]
                    st.metric(
                        "🎯 Canal Más Eficiente",
                        best_efficiency['Primary_channel_group'],
                        delta=f"Índice: {best_efficiency['Efficiency_Index']:.2f}",
                        help="Canal con mayor ratio de usuarios adquiridos vs costo invertido"
                    )
                
                with col2:
                    if display_table['CAC'].max() > 0:
                        lowest_cac = display_table.loc[display_table['CAC'].idxmin()]
                        st.metric(
                            "💰 Menor CAC",
                            lowest_cac['Primary_channel_group'],
                            delta=f"CAC: {format_currency(lowest_cac['CAC'])}",
                            help="Canal con menor costo de adquisición por cliente"
                        )
                    else:
                        st.metric("💰 Menor CAC", "N/A", help="No hay datos válidos de CAC")
                
                with col3:
                    highest_volume = display_table.loc[display_table['Total_users'].idxmax()]
                    st.metric(
                        "📈 Mayor Volumen",
                        highest_volume['Primary_channel_group'],
                        delta=f"{highest_volume['Total_users']:,} usuarios",
                        help="Canal que ha adquirido más usuarios totales"
                    )
            
            # Resumen global
        st.markdown("---")
        st.markdown("##### 📋 Resumen Global")
        
        total_summary_col1, total_summary_col2, total_summary_col3, total_summary_col4 = st.columns(4)
        
        with total_summary_col1:
            total_all_users = all_channels_summary['Total_users'].sum()
            st.metric("👥 Total Usuarios", f"{total_all_users:,}")
        
        with total_summary_col2:
            total_all_cost = all_channels_summary['Ads_cost'].sum()
            st.metric("💰 Costo Total", format_currency(total_all_cost))
        
        with total_summary_col3:
            avg_cac = safe_division(total_all_cost, total_all_users, default=0)
            st.metric("📊 CAC Promedio", format_currency(avg_cac))
        
        with total_summary_col4:
            total_channels = len(all_channels_summary)
            st.metric("🎯 Total Canales", f"{total_channels}")
            
    except Exception as e:
        logger.error(f"Error analyzing customer acquisition rate by channel: {str(e)}")
        st.error("Error al analizar tasa de adquisición por canal.")
        
        # Información de diagnóstico en caso de error
        with st.expander("🔍 Información de Diagnóstico"):
            st.text(f"Error: {str(e)}")
            if not df_ads_filtered.empty:
                st.text(f"Filas en datos: {len(df_ads_filtered)}")
                st.text(f"Columnas disponibles: {list(df_ads_filtered.columns)}")
                st.text("Primeras 5 filas:")
                st.dataframe(df_ads_filtered.head())

@st.cache_data(ttl=config.CACHE_TTL, show_spinner="🔍 Obteniendo historial de clientes...")
def get_all_time_first_purchase_dates(_client: bigquery.Client) -> Optional[pd.Series]:
    """
    Obtiene la fecha de la primera compra de TODOS los clientes desde el inicio de los tiempos.
    Ejecuta una query optimizada para minimizar la carga de datos.
    """
    log_function_call("get_all_time_first_purchase_dates")
    try:
        items_table = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['items']}"
        customers_table = f"{config.BIGQUERY_PROJECT_ID}.{config.BIGQUERY_DATASET}.{config.BIGQUERY_TABLES['ft_customers']}"

        # Esta query une items con clientes para obtener el email, luego encuentra la fecha mínima de pedido por email.
        query = f"""
            SELECT
                c.CUSTOMER_EMAIL AS USER_EMAIL,
                MIN(i.ORDER_CREATE_DATE) as FIRST_PURCHASE_DATE
            FROM `{items_table}` AS i
            JOIN `{customers_table}` AS c ON i.CUSTOMER_ID = c.CUSTOMER_ID
            WHERE c.CUSTOMER_EMAIL IS NOT NULL AND i.ORDER_CREATE_DATE IS NOT NULL
            GROUP BY USER_EMAIL
        """
        
        df_first_purchase = execute_bigquery_query(_client, query)
        
        if df_first_purchase is None or df_first_purchase.empty:
            logger.error("Could not retrieve all-time first purchase dates.")
            return None

        df_first_purchase['FIRST_PURCHASE_DATE'] = pd.to_datetime(df_first_purchase['FIRST_PURCHASE_DATE']).dt.tz_localize(None)
        
        # Convertir a una Serie para búsquedas rápidas
        first_purchase_series = df_first_purchase.set_index('USER_EMAIL')['FIRST_PURCHASE_DATE']
        
        logger.info(f"Successfully retrieved first purchase dates for {len(first_purchase_series)} unique customers.")
        return first_purchase_series
                
    except Exception as e:
        logger.error(f"Error getting all-time first purchase dates: {e}")
        return None

def create_missing_data_section():
    """Crea una sección dinámica con todos los datos faltantes detectados basada solo en datos reales de BigQuery."""
    
    # Combinar todos los datos faltantes
    all_missing = state.missing_data_points
    
    if not all_missing and not state.error_messages:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ✅ Estado de Datos BigQuery")
        st.sidebar.success("Conexión exitosa - Datos principales disponibles")
        st.sidebar.markdown("📊 **Fuente:** BigQuery en tiempo real")
        return
    
    st.sidebar.markdown("---")
    
    # Mostrar resumen compacto
    total_missing = len(all_missing)
    total_errors = len(state.error_messages)
    
    if total_missing > 0 or total_errors > 0:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if total_missing > 0:
                st.metric("📊 Datos Faltantes", total_missing)
        with col2:
            if total_errors > 0:
                st.metric("⚠️ Errores", total_errors)
    
    # Sección expandible con todos los detalles
    with st.sidebar.expander("📋 Ver Datos Faltantes Detectados"):
        # Categorizar los datos faltantes
        data_categories = {
            "🚨 Errores Críticos": [],
            "📊 Analytics Avanzado": [],
            "💰 Marketing & Ads": [], 
            "😊 Satisfacción Cliente": [],
            "🏗️ Estructura Datos": []
        }
        
        # Clasificar automáticamente los puntos faltantes
        for point in all_missing:
            point_lower = point.lower()
            if any(word in point_lower for word in ['bounce', 'rebote', 'session', 'sesión', 'duration', 'duración', 'add_to_cart', 'checkout']):
                data_categories["📊 Analytics Avanzado"].append(point)
            elif any(word in point_lower for word in ['leads_generated', 'cpl', 'lead', 'new_users', 'conversion_rate']):
                data_categories["💰 Marketing & Ads"].append(point)
            elif any(word in point_lower for word in ['nps', 'satisfacción', 'satisfaction']):
                data_categories["😊 Satisfacción Cliente"].append(point)
            elif any(word in point_lower for word in ['columna', 'column', 'archivo', 'file', 'faltante', 'error', 'null']):
                data_categories["🏗️ Estructura Datos"].append(point)
            else:
                data_categories["📊 Analytics Avanzado"].append(point)
        
        # Agregar errores críticos si existen
        for error in state.error_messages[-3:]:  # Solo los últimos 3 errores
            data_categories["🚨 Errores Críticos"].append(error)
        
        # Mostrar categorías con datos
        for category, items in data_categories.items():
            if items:
                st.markdown(f"**{category}**")
                for item in items:
                    st.markdown(f"• {item}")
                st.markdown("")
        
        # Recomendaciones específicas para BigQuery
        st.markdown("---")
        st.markdown("**💡 Recomendaciones BigQuery:**")
        st.markdown("• Verificar permisos de tabla en BigQuery")
        st.markdown("• Validar estructura de columnas")
        st.markdown("• Revisar logs de conexión en dashboard.log")
        st.markdown("• Ejecutar test_connection.py para diagnóstico")
        
        st.markdown("**📈 Nota:** Métricas mostradas usan solo datos reales disponibles - no se inventan valores.")

def main():
    """Función principal para ejecutar el dashboard."""
    # Setup principal
    (df_items_filtered, df_sessions_filtered, df_ga_orders_filtered, df_ads_filtered,
     start_date, end_date,
     df_items, df_sessions, df_ga_orders, df_ads, df_ft_customers) = setup_dashboard()

    # Título principal del Dashboard
    st.title("👥 Customer Analytics Dashboard")
    st.markdown("""
        <div class='dashboard-subtext'>
        Analiza el comportamiento de tus clientes para descubrir insights sobre recurrencia, valor de vida (LTV) y patrones de compra. 
        Usa el filtro de fechas en el sidebar para acotar el análisis.
        </div>
        <hr>
    """, unsafe_allow_html=True)

    # Definir información del período para pasar a las pestañas
    period_str = f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
    period_info = {
        'start_date': start_date.strftime('%d/%m/%Y'),
        'end_date': end_date.strftime('%d/%m/%Y'),
        'start_date_dt': start_date,
        'end_date_dt': end_date,
        'period_str': period_str
    }

    # Crear pestañas principales
    tab_vision, tab_frecuencia, tab_ltv, tab_productos, tab_canales, tab_diagnostico = st.tabs([
        "🌐 Visión General", 
        "🔄 Frecuencia", 
        "💰 LTV", 
        "🛒 Productos", 
        "📈 Canales", 
        "⚙️ Diagnóstico"
    ])

    # Tab 1: Visión General del Cliente  
    with tab_vision:
        st.markdown("### 🌐 Visión General del Cliente")

        st.markdown(f"#### Resumen para el período filtrado")
        st.markdown(f"<div class='dashboard-subtext' style='margin-bottom:1rem;'>Análisis basado en el rango de fechas: {period_info['period_str']}</div>", unsafe_allow_html=True)
        
        if not df_items_filtered.empty:
            # Crear período info simplificado para compatibilidad con funciones existentes
            period_info_for_overview = {
                'label': f"Filtro del Sidebar ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')})",
                'subtitle': f"Datos filtrados desde {start_date.strftime('%d/%m/%Y')} hasta {end_date.strftime('%d/%m/%Y')}",
                'start_date': start_date.strftime('%d/%m/%Y'),
                'end_date': end_date.strftime('%d/%m/%Y'),
                'start_date_dt': start_date,
                'end_date_dt': end_date,
                'period_str': period_str
            }
            
            # Análisis de métricas clave
            analyze_customer_overview(df_items_filtered, df_items, period_info_for_overview)
            
            st.markdown("---")
            
            # Análisis de evolución
            first_purchase_dates_all_time = df_items.groupby('USER_EMAIL')['ORDER_CREATE_DATE'].min().rename('FIRST_PURCHASE_DATE_GLOBAL')
            create_evolution_charts(df_items_filtered, period_info_for_overview, first_purchase_dates_all_time)
            
            st.markdown("---")
            
            # Análisis geográfico - REMOVIDO: no hay datos de SHIPPING_COUNTRY disponibles
            # analyze_geographic_distribution(df_items_filtered, period_info_for_overview)
        else:
            st.info(f"No hay datos de órdenes para el período seleccionado en el sidebar. Por favor, ajusta el rango de fechas.")

        if state.error_messages:
            st.error("Se detectaron los siguientes errores críticos durante la ejecución:")
            for msg in state.error_messages:
                st.code(msg, language='text')

    # Tab 2: Frecuencia y Recurrencia
    with tab_frecuencia:
        st.markdown("### 🔄 Frecuencia y Recurrencia")
        st.markdown("""
            <div class='dashboard-subtext'>
            Analiza con qué frecuencia compran los clientes, cuántos de ellos regresan y cuánto tardan en hacerlo. 
            Este análisis se basa en los datos filtrados por el rango de fechas global seleccionado.
            </div>
        """, unsafe_allow_html=True)
        
        # Realizar análisis y mostrar métricas
        customer_orders_freq, freq_metrics = analyze_frequency_metrics(df_items_filtered)
        create_frequency_metrics_display(freq_metrics)
        st.markdown("---")
                    
        # Crear gráficos de distribución
        col_dist1, col_dist2 = st.columns([1, 1])
        with col_dist1:
            create_frequency_segmentation_chart(customer_orders_freq)
        with col_dist2:
            create_frequency_distribution_charts(customer_orders_freq, df_items_filtered)
        st.markdown("---")
        
        # Análisis de cohortes
        analyze_cohort_retention(df_items, df_items_filtered)
                    
    # Tab 3: Valor del Cliente (LTV)
    with tab_ltv:
        st.markdown("### 💰 Valor del Cliente (LTV)")
        st.markdown("<div class='dashboard-subtext'>Entendiendo el valor que los clientes aportan a lo largo del tiempo, basado en el período seleccionado en el filtro global de fechas.</div>", unsafe_allow_html=True)

        # Calcular LTV
        ltv_data = calculate_ltv_data(df_items_filtered)
            
        # Métricas básicas LTV
        display_basic_ltv_metrics(ltv_data)
        st.markdown("---")
        
        # Análisis detallado
        col_ltv1, col_ltv2 = st.columns(2)
        with col_ltv1:
            analyze_aov_trends(df_items_filtered)
            st.markdown("---")
            create_ltv_segmentation(ltv_data)
        with col_ltv2:
            analyze_ltv_by_frequency_segment(ltv_data, df_items_filtered)
        st.markdown("---")

        analyze_ltv_by_sport(ltv_data, df_items_filtered)
        st.markdown("---")
        
        analyze_ltv_by_acquisition_channel(ltv_data, df_items, df_ga_orders_filtered)

    # Tab 4: Comportamiento de Compra
    with tab_productos:
        st.markdown("### 🛒 Comportamiento de Compra")
        st.markdown("<div class='dashboard-subtext'>Qué compran los clientes y cómo se relacionan los productos.</div>", unsafe_allow_html=True)

        # Análisis de productos
        analyze_top_products_recurrent_customers(df_items_filtered)
        st.markdown("---")

        analyze_cross_sell_patterns(df_items_filtered)
        st.markdown("---")

        # Análisis de temporada
        analyze_seasonality(df_items_filtered)
        st.markdown("---")

        # Análisis avanzado de categorías
        analyze_advanced_category_sports_analysis(df_items_filtered)

    # Tab 5: Canales y Adquisición
    with tab_canales:
        st.markdown("### 📈 Canales y Adquisición")
        st.markdown("<div class='dashboard-subtext'>Análisis del rendimiento de los canales de adquisición y su impacto en el LTV.</div>", unsafe_allow_html=True)
        
        # LTV/CAC ratio
        ltv_data_for_cac = calculate_ltv_data(df_items_filtered)
        analyze_ltv_cac_ratio_by_channel(ltv_data_for_cac, df_items, df_ga_orders_filtered, df_ads_filtered)
        st.markdown("---")
        
        # Tasa de adquisición
        analyze_customer_acquisition_rate_by_channel(df_ads_filtered)
            
    # Tab 6: Diagnóstico
    with tab_diagnostico:
        st.markdown("### ⚙️ Diagnóstico del Dashboard")
        st.markdown("""
        <div class='dashboard-subtext'>
        Esta sección proporciona información sobre la integridad de los datos y la configuración del dashboard,
        ayudando a identificar posibles problemas con las fuentes de datos o el procesamiento.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ✅ Estado de los Datos Cargados")
        data_status_expander = st.expander("Verificar detalles de los datos cargados")
        with data_status_expander:
            display_data_loading_status({
                "Items (Líneas de Pedido)": df_items_filtered,
                "Sesiones GA": df_sessions_filtered,
                "Órdenes GA": df_ga_orders_filtered,
                "Anuncios GA": df_ads_filtered,
                "Clientes": df_ft_customers
            })

        st.markdown("#### ⚠️ Datos Faltantes o Ignorados")
        analyze_missing_columns(df_items, df_ft_customers, df_sessions, df_ga_orders, df_ads)
        
        st.markdown("#### 📝 Logs y Errores")
        st.info("Revisa el archivo `dashboard.log` para un registro detallado de todas las operaciones y posibles errores no críticos.")
        
        # Mostrar errores críticos si existen
        if state.error_messages:
            st.error("Se detectaron los siguientes errores críticos durante la ejecución:")
            for msg in state.error_messages:
                st.code(msg, language='text')

    # Llamar a la sección de datos faltantes en el sidebar
create_missing_data_section()

if __name__ == '__main__':
    main()