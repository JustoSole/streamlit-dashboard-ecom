from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core import exceptions
import json
import pandas as pd
import os
from datetime import datetime
import sys
import subprocess

def check_dependencies():
    """Check and install required dependencies"""
    try:
        import db_dtypes
        return True
    except ImportError:
        print("\n⚠️ Instalando dependencia requerida: db-dtypes...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "db-dtypes"])
            print("✅ db-dtypes instalado correctamente")
            return True
        except Exception as e:
            print(f"❌ Error instalando db-dtypes: {str(e)}")
            return False

def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def create_data_dir():
    """Create directory for storing data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"bigquery_data_{timestamp}"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def download_table_sample(bq_client, table_ref: str, data_dir: str, sample_percent: float = 50):
    """Download a sample of data from a BigQuery table"""
    table_name = table_ref.split('.')[-1]
    print(f"\nDescargando muestra de {table_name}...")
    
    try:
        # Obtener información de la tabla
        table = bq_client.get_table(table_ref)
        total_rows = table.num_rows
        
        # Calcular número de filas a descargar
        sample_rows = int(total_rows * (sample_percent / 100))
        print(f"   Total de filas: {total_rows:,}")
        print(f"   Muestra ({sample_percent}%): {sample_rows:,} filas")
        
        # Query para obtener muestra aleatoria
        query = f"""
        SELECT *
        FROM `{table_ref}`
        WHERE RAND() <= {sample_percent/100}
        """
        
        print("   Ejecutando query...")
        df = bq_client.query(query).to_dataframe()
        
        # Guardar datos
        csv_path = os.path.join(data_dir, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Datos guardados en: {csv_path}")
        
        # Guardar esquema
        schema_path = os.path.join(data_dir, f"{table_name}_schema.txt")
        with open(schema_path, 'w') as f:
            f.write(f"Tabla: {table_ref}\n")
            f.write(f"Total de filas: {total_rows:,}\n")
            f.write(f"Muestra: {sample_percent}% ({sample_rows:,} filas)\n")
            f.write(f"Fecha de descarga: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Esquema:\n")
            for field in table.schema:
                f.write(f"• {field.name}: {field.field_type}\n")
        
        print(f"✅ Esquema guardado en: {schema_path}")
        
        # Guardar estadísticas básicas
        stats_path = os.path.join(data_dir, f"{table_name}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Estadísticas básicas para {table_name}\n")
            f.write("="*50 + "\n\n")
            f.write("Conteo de valores no nulos por columna:\n")
            f.write(df.count().to_string())
            f.write("\n\nTipos de datos:\n")
            f.write(df.dtypes.to_string())
            f.write("\n\nEstadísticas numéricas:\n")
            f.write(df.describe().to_string())
        
        print(f"✅ Estadísticas guardadas en: {stats_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando {table_name}: {str(e)}")
        return False

def download_bigquery_samples():
    """Download sample data from all tables"""
    print_section("📥 DESCARGANDO MUESTRAS DE BIGQUERY")
    
    # Verificar dependencias
    if not check_dependencies():
        print("❌ No se pueden descargar los datos sin las dependencias necesarias")
        return
    
    try:
        # 1. Cargar credenciales
        print("1. Verificando credenciales...")
        with open('service_account.json') as source:
            info = json.load(source)
        project_id = info["project_id"]
        sa_email = info.get("client_email", "<unknown>")
        print("✅ Credenciales cargadas correctamente")
        print(f"   Project ID: {project_id}")
        print(f"   Service Account: {sa_email}")
        
        # 2. Inicializar cliente
        print("\n2. Inicializando cliente BigQuery...")
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=[
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/cloud-platform"
            ]
        )
        bq_client = bigquery.Client(
            credentials=credentials,
            project=project_id,
            client_options={"scopes": credentials.scopes}
        )
        print("✅ Cliente BigQuery inicializado")
        
        # 3. Crear directorio para datos
        data_dir = create_data_dir()
        print(f"\n3. Directorio de datos creado: {data_dir}")
        
        # 4. Lista de tablas a descargar
        tables_to_download = [
            "racket-central-gcp.racket_central_prd.SHOPIFY_ORDERS_LINEITEMS",
            "racket-central-gcp.racket_central_prd.GA_ORDERS",
            "racket-central-gcp.racket_central_prd.GA_SESSIONS",
            "racket-central-gcp.racket_central_prd.GA_ADS_CAMPAIGNS"
        ]
        
        # 5. Descargar muestras
        print_section("4. DESCARGANDO MUESTRAS DE TABLAS")
        
        all_ok = True
        for table_ref in tables_to_download:
            if not download_table_sample(bq_client, table_ref, data_dir):
                all_ok = False
        
        if all_ok:
            print(f"""
✅ Todas las muestras descargadas exitosamente

Los datos están en el directorio: {data_dir}
Para cada tabla encontrarás:
- [nombre_tabla].csv: Datos de la muestra
- [nombre_tabla]_schema.txt: Esquema y metadatos
- [nombre_tabla]_stats.txt: Estadísticas básicas

Puedes usar estos archivos para:
1. Análisis exploratorio de datos (EDA)
2. Desarrollo y pruebas locales
3. Visualizaciones y reportes
            """)
        else:
            print(f"""
⚠️ Algunas tablas no pudieron ser descargadas

Verifica:
1. Permisos del service account ({sa_email})
2. Conexión a internet
3. Espacio disponible en disco

Los datos descargados están en: {data_dir}
            """)
        
    except FileNotFoundError:
        print("❌ Error: service_account.json no encontrado")
    except json.JSONDecodeError:
        print("❌ Error: service_account.json está mal formateado")
    except exceptions.Forbidden as e:
        print(f"❌ Error de permisos: {e.message}")
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")

if __name__ == "__main__":
    download_bigquery_samples() 