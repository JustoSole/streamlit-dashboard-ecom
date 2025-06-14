from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core import exceptions
import json
import pandas as pd
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

def test_table_access(bq_client, table_ref: str):
    """Test basic access to a table with a simple query"""
    print(f"\nProbando acceso a {table_ref}...")
    try:
        # Verificar que la tabla existe y su metadata
        table = bq_client.get_table(table_ref)
        print(f"✅ Tabla encontrada")
        print(f"   Filas (metadata): {table.num_rows:,}")
        print(f"   Tamaño: {table.num_bytes / (1024*1024):.2f} MB")
        print("\nEsquema de la tabla:")
        for field in table.schema:
            print(f"   • {field.name}: {field.field_type}")
        
        # Intentar una query simple y contar filas reales
        print("\nProbando query de conteo real...")
        count_query = f"SELECT COUNT(*) AS total FROM `{table_ref}`"
        count_result = bq_client.query(count_query).result()
        # Usar next() en lugar de to_dataframe() para evitar el error de db-dtypes
        real_count = next(count_result).total
        print(f"✅ Filas (COUNT(*)): {int(real_count):,}")

        print("\nProbando SELECT * LIMIT 10...")
        query = f"SELECT * FROM `{table_ref}` LIMIT 10"
        query_job = bq_client.query(query)
        results = query_job.result()
        
        # Convertir resultados a lista de diccionarios
        rows = []
        for row in results:
            row_dict = {}
            for field in table.schema:
                row_dict[field.name] = row[field.name]
            rows.append(row_dict)
        
        # Crear DataFrame manualmente
        df = pd.DataFrame(rows)
        print(f"✅ Query ejecutada exitosamente")
        print(f"   Filas retornadas: {len(df):,}")
        if not df.empty:
            print("\nPrimeras 3 filas:")
            # Convertir valores a string para evitar problemas de formato
            print(df.head(3).astype(str).to_string(index=False))
        return True

    except exceptions.Forbidden as e:
        print(f"❌ PERMISOS DENEGADOS: {e.message}")
        return False
    except Exception as e:
        print(f"❌ Error accediendo a la tabla: {str(e)}")
        return False

def test_bigquery_connection():
    """Test BigQuery connection and verify access to main tables"""
    print_section("🔍 TEST DE CONEXIÓN BIGQUERY")
    
    # Verificar dependencias primero
    if not check_dependencies():
        print("❌ No se pueden ejecutar las queries sin las dependencias necesarias")
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
        
        # 2. Inicializar cliente con scopes de Drive
        print("\n2. Inicializando cliente BigQuery con Drive scopes...")
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
        print("✅ Cliente BigQuery inicializado con scopes:", credentials.scopes)
        
        # 3. Verificar permisos básicos
        print("\n3. Verificando permisos básicos...")
        test_query = "SELECT 1"
        bq_client.query(test_query).result()
        print("✅ Permisos básicos verificados correctamente")
        
        # 4. Probar acceso a las tablas principales
        print_section("4. VERIFICANDO ACCESO A TABLAS PRINCIPALES")
        
        tables_to_check = [
            "racket-central-gcp.racket_central_dev.RAW_SHOPIFY_ORDERS",
            "racket-central-gcp.racket_central_dev.RAW_SHOPIFY_CUSTOMERS",
            "racket-central-gcp.racket_central_dev.RAW_GA_CAMPAIGN_METRICS",
            "racket-central-gcp.racket_central_dev.RAW_GA_CAMPAIGN_TRANSACTIONS",
            "racket-central-gcp.racket_central_dev.RAW_SHOPIFY_PRODUCTS"
        ]
        
        all_ok = True
        for table_ref in tables_to_check:
            if not test_table_access(bq_client, table_ref):
                all_ok = False
        
        if all_ok:
            print("\n✅ Todas las tablas son accesibles y devuelven datos")
        else:
            print(f"""
⚠️ Algunas tablas no devolvieron datos correctamente.

Asegúrate de que:
- El Service Account ({sa_email}) tiene estos roles:
  • BigQuery Data Viewer (roles/bigquery.dataViewer)
  • BigQuery Job User (roles/bigquery.jobUser)
  • BigQuery User (roles/bigquery.user)
  • BigQuery Connection User (roles/bigquery.connectionUser)

- La API de Google Drive está habilitada en tu proyecto.
- El archivo de Google Sheets está compartido con tu Service Account.
- Estás apuntando al proyecto y dataset correctos.
            """)
        
    except FileNotFoundError:
        print("❌ Error: service_account.json no encontrado")
    except json.JSONDecodeError:
        print("❌ Error: service_account.json está mal formateado")
    except exceptions.Forbidden as e:
        print(f"❌ Error de permisos generales: {e.message}")
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")

if __name__ == "__main__":
    test_bigquery_connection()
