"""
Configuración para BigQuery Dashboard
=====================================

Este archivo contiene la configuración para conectar el dashboard a BigQuery.
"""

# --- GCP Project Configuration ---
BIGQUERY_PROJECT_ID = "racket-central-gcp"
SERVICE_ACCOUNT_FILE = "service_account.json"

# --- BigQuery Dataset Configuration ---
# Usar 'racket_central_dev' para desarrollo o 'racket_central_prd' para producción.
BIGQUERY_DATASET = "racket_central_dev" 

# --- BigQuery Table Names ---
# Diccionario centralizado de tablas para cambiar fácilmente entre entornos.
BIGQUERY_TABLES = {
    "items": "RAW_SHOPIFY_ORDERS_LINEITEMS",
    "sessions": "RAW_GA_CAMPAIGN_METRICS",
    "ga_orders": "RAW_GA_CAMPAIGN_TRANSACTIONS",
    "ga_ads": "RAW_GA_CAMPAIGN_METRICS",
    "ft_customers": "RAW_SHOPIFY_CUSTOMERS",
    "shopify_orders": "RAW_SHOPIFY_ORDERS"  # Added for shipping data
}

# --- Cache Configuration ---
# Tiempo de vida del cache en segundos (1 hora = 3600 segundos)
CACHE_TTL = 3600

# --- Query Optimization ---
# Límite máximo de filas por consulta (optimizado para dashboard)
MAX_QUERY_ROWS = 100000 

# --- BigQuery Scopes ---
# Permisos necesarios para la cuenta de servicio.
BIGQUERY_SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive.readonly",
]

# --- App Behavior ---
USE_BIGQUERY = True  # Mantener en True, el dashboard usa solo BigQuery
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos 