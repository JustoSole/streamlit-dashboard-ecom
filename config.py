# config.py

# --- BigQuery Configuration ---
BIGQUERY_PROJECT_ID = "racket-central-gcp"
BIGQUERY_DATASET = "racket_central_dev"

# --- Service Account ---
# Make sure this file is in .gitignore if your repo is public
SERVICE_ACCOUNT_FILE = "service_account.json"

# Add the necessary scopes for BigQuery and Google Drive (if using linked sheets)
BIGQUERY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive",
]

# --- Cache Settings ---
# Time-to-live for data cache in seconds (e.g., 3600 = 1 hour)
CACHE_TTL = 3600

# --- BigQuery Table Names ---
# Mapping of short names to ONLY the table names, not the full path.
# The full path will be constructed dynamically in the data loading functions.
BIGQUERY_TABLES = {
    "orders": "RAW_SHOPIFY_ORDERS",
    "customers": "RAW_SHOPIFY_CUSTOMERS",
    "products": "RAW_SHOPIFY_PRODUCTS",
    "ga_metrics": "RAW_GA_CAMPAIGN_METRICS",
    "ga_transactions": "RAW_GA_CAMPAIGN_TRANSACTIONS",
    "line_items": "RAW_SHOPIFY_ORDERS_LINEITEMS"
}

# --- Query Limits ---
# Maximum number of rows to fetch in a query to prevent overloads
# Set to None for no limit
MAX_QUERY_ROWS = 250000

# --- Expected Schemas ---
# Defines the expected columns for our final, processed dataframes.
# These are derived from `database_schema.md` and should be kept in sync.
# Column names are LOWERCASE as they are standardized during processing.
EXPECTED_ORDERS_SCHEMA = [
    # --- From RAW_SHOPIFY_ORDERS ---
    "order_id", "order_number", "order_create_date",
    "order_fulfilled_date", "order_cancel_date",
    "order_cancel_reason", "order_cancel_note",
    "order_return_status",
    "order_original_lineitem_qty", "order_final_lineitem_qty",
    "order_original_subtotal_amount", "order_final_subtotal_amount",
    "order_original_shipping_amount", "order_final_shipping_amount",
    "order_original_tax_amount", "order_final_tax_amount",
    "order_original_discount_amount", "order_final_discount_amount",
    "order_original_total_amount", "order_final_total_amount",
    "order_final_refound_amount", "order_final_net_amount",
    "order_discount_code", "order_financial_status",
    "order_fulfillment_status", "shipping_address",
    "shipping_city", "shipping_state", "shipping_zipcode",
    "shipping_latitude", "shipping_longitude",
    "shipping_service_level", "shipping_source",
    "order_first_visit_source", "order_first_visit_source_type",
    "order_first_visit_date", "order_moments_qty",
    "order_days_to_conversion", "order_delivery_date",
    "order_estimated_delivery_date", "order_delivered_on_time",
    "order_tags",
    # --- From RAW_SHOPIFY_CUSTOMERS (joined) ---
    "customer_id", "customer_email",
    # --- From RAW_SHOPIFY_ORDERS_LINEITEMS (joined) ---
    "lineitem_id",
    "lineitem_original_qty", "lineitem_final_qty",
    "lineitem_unit_original_amount", "lineitem_unit_final_amount",
    # --- From RAW_SHOPIFY_PRODUCTS (joined) ---
    "variant_id", "product_id", "product_status",
    "product_category", "product_type", "product_vendor",
    "product_name", "product_price", "product_sku",
    "variant_create_date", "variant_last_update_date", 
    "product_description", "variant_stock_qty", 
    "product_compare_at_price", "product_tags", 
    "variant_ean", "product_is_collective", 
    "product_has_price_markdown",
    # --- From enrich_dataframe_with_categories (added in utils.py) ---
    "sport_universe", "player_level"
]

EXPECTED_CUSTOMERS_SCHEMA = [
    "customer_id", "customer_create_date",
    "customer_display_name", "customer_email",
    "customer_phone", "customer_city",
    "customer_state", "customer_country",
    "customer_total_spent_amount", "customer_total_orders",
    "customer_last_order_id", "customer_last_order_ga_transaction_id",
    "customer_last_order_order_number", "order_create_date"
]

EXPECTED_GA_METRICS_SCHEMA = [
    "sk_datecampaign", "campaign_date", "campaign_id",
    "campaign_name", "campaign_sessions", "campaign_bounce_rate",
    "campaign_ad_revenue", "campaign_total_revenue",
    "campaign_roas", "campaign_new_users", "campaign_first_time_purcharsers",
    "campaign_avg_session_duration", "campaign_add_to_cart_events",
    "campaign_begin_checkout_events", "campaign_purchase_events",
    # --- Added during processing in data.py ---
    "primary_channel_group", "total_users", "ads_cost",
    "ads_impressions", "ads_clicks"
]

EXPECTED_PRODUCTS_SCHEMA = [
    "variant_id", "product_id", "variant_ean",
    "variant_sku", "variant_create_date", "variant_last_update_date",
    "product_status", "product_category", "product_type",
    "product_vendor", "product_name", "product_description",
    "product_price", "product_compare_at_price", "variant_stock_qty",
    "product_tags", "product_is_collective", "product_has_price_markdown"
]

EXPECTED_LINE_ITEMS_SCHEMA = [
    "lineitem_id", "order_create_date", "order_id",
    "order_number", "order_fulfillment_status", "order_financial_status",
    "customer_id", "product_id", "product_sku",
    "product_name", "product_vendor", "product_type",
    "lineitem_original_qty", "lineitem_final_qty",
    "lineitem_unit_original_amount", "lineitem_unit_final_amount"
]

EXPECTED_GA_TRANSACTIONS_SCHEMA = [
    "sk_rownumber", "campaign_date", "transaction_id",
    "campaign_source", "campaign_primary_group", "campaign_source_platform",
    "campaign_id", "campaign_sessions", "campaign_users",
    "campaign_avg_session_duration", "campaign_landing_page"
]

# --- Visual Settings ---
DEFAULT_CHART_HEIGHT = 450
COLOR_PALETTE = "Blues"
PRIMARY_COLOR = "#008080" # Teal color for branding

# --- Business Logic ---
# Define keywords or categories for analysis
SPORT_KEYWORDS = {
    'Pickleball': ['pickleball', 'pickle ball', 'pickball'],
    'Pádel': ['padel', 'pádel', 'paddle tennis', 'pop tennis']
}

# --- Product Categorization Keywords ---
PRODUCT_CATEGORIES = {
    'Paddles/Rackets': ['paddle', 'paleta', 'racket', 'raqueta'],
    'Balls': ['ball', 'pelota', 'bola'],
    'Apparel': ['shirt', 'shorts', 'polo', 'camiseta', 'pantalón', 'short', 'apparel', 'clothing'],
    'Footwear': ['shoe', 'sneaker', 'zapato', 'calzado'],
    'Bags/Cases': ['bag', 'case', 'cover', 'bolsa', 'funda', 'mochila'],
    'Accessories': ['grip', 'overgrip', 'string', 'cuerda', 'cordaje', 'wristband', 'headband'],
    'Court Equipment': ['net', 'red', 'court', 'cancha']
}

LEVEL_KEYWORDS = {
    'Beginner': ['beginner', 'starter', 'principiante', 'inicial', 'básico', 'basic'],
    'Intermediate': ['intermediate', 'intermedio', 'medio', 'recreational', 'recreativo', 'club', 'sport', 'game'],
    'Advanced': ['advanced', 'avanzado', 'pro', 'professional', 'profesional', 'elite', 'competition', 'competición', 'tour', 'championship', 'master', 'premium']
}

MONTHS_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

DAYS_ORDER = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
] 