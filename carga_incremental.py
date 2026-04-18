import psycopg2
import os
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ─────────────────────────────────────────
# Conexión a Go_BD
# ─────────────────────────────────────────
load_dotenv()

def conectar():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432"),
        sslmode="require"
    )

# ─────────────────────────────────────────
# Datos base para generación sintética
# ─────────────────────────────────────────
TIPO_TRANSAC   = ["venta", "reposicion", "devolucion", "traslado"]
TEMPORADA_ALTA = [1, 2, 6, 7, 10, 11, 12]

# ─────────────────────────────────────────
# 1. Obtener última fecha cargada
# ─────────────────────────────────────────
def obtener_ultima_fecha(cursor):
    cursor.execute("""
        SELECT MAX(transaction_date) FROM transacciones
    """)
    resultado = cursor.fetchone()[0]

    if resultado is None:
        # Si no hay datos, empieza desde hace 3 días
        ultima = datetime.now() - timedelta(days=3)
    else:
        ultima = resultado

    print(f"   Última fecha en Go_BD: {ultima.strftime('%Y-%m-%d %H:%M')}")
    return ultima

# ─────────────────────────────────────────
# 2. Obtener tiendas y SKUs existentes
# ─────────────────────────────────────────
def obtener_tiendas_skus(cursor):
    cursor.execute("SELECT id FROM tiendas")
    tiendas = [str(row[0]) for row in cursor.fetchall()]

    cursor.execute("SELECT id FROM catalogos")
    skus = [str(row[0]) for row in cursor.fetchall()]

    print(f"   Tiendas activas: {len(tiendas)}")
    print(f"   SKUs en catálogo: {len(skus)}")
    return tiendas, skus

# ─────────────────────────────────────────
# 3. Generar transacciones nuevas
# ─────────────────────────────────────────
def generar_transacciones_nuevas(cursor, tiendas, skus, desde, hasta):
    print(f"\n   Generando transacciones desde {desde.strftime('%Y-%m-%d')} hasta {hasta.strftime('%Y-%m-%d')}...")

    total    = 0
    fecha    = desde + timedelta(days=1)

    while fecha <= hasta:
        mes_actual  = fecha.month
        ventas_dia  = random.randint(15, 30) if mes_actual in TEMPORADA_ALTA else random.randint(5, 15)

        for _ in range(ventas_dia):
            tienda_id = random.choice(tiendas)
            sku_id    = random.choice(skus)
            tipo      = random.choices(
                TIPO_TRANSAC,
                weights=[70, 15, 10, 5]
            )[0]

            precio_venta = round(random.uniform(30000, 350000), 2)
            cantidad     = random.randint(1, 5) if tipo == "venta" else random.randint(5, 30)

            cursor.execute("""
                INSERT INTO transacciones (
                    receipt_id, sku_id, source_location_id, target_location_id,
                    quantity, sale_price, currency, type,
                    transaction_date, transaction_date_process
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                f"INC_{total:08d}",
                sku_id,
                "BODEGA_CENTRAL",
                tienda_id,
                cantidad,
                precio_venta if tipo == "venta" else 0,
                "COP",
                tipo,
                fecha,
                datetime.now()
            ))
            total += 1

        fecha += timedelta(days=1)

    return total

# ─────────────────────────────────────────
# 4. Actualizar inventario
# ─────────────────────────────────────────
def actualizar_inventario(cursor, tiendas, skus):
    print("   Actualizando niveles de inventario...")
    actualizados = 0

    for tienda_id in random.sample(tiendas, k=min(10, len(tiendas))):
        for sku_id in random.sample(skus, k=min(20, len(skus))):
            nuevo_stock = random.randint(0, 50)

            cursor.execute("""
                UPDATE inventarios
                SET site_qty = %s,
                    status_date = %s,
                    transaction_date_process = %s
                WHERE location_id = %s AND sku_id = %s
            """, (
                nuevo_stock,
                datetime.now().date(),
                datetime.now(),
                tienda_id,
                sku_id
            ))
            if cursor.rowcount > 0:
                actualizados += 1

    print(f"   Inventarios actualizados: {actualizados} registros")

# ─────────────────────────────────────────
# 5. Registrar log de carga
# ─────────────────────────────────────────
def registrar_log(cursor, desde, hasta, total_transac):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS log_cargas (
            id              SERIAL PRIMARY KEY,
            fecha_ejecucion TIMESTAMP DEFAULT NOW(),
            fecha_desde     TIMESTAMP,
            fecha_hasta     TIMESTAMP,
            transacciones   INTEGER,
            estado          VARCHAR(50)
        )
    """)

    cursor.execute("""
        INSERT INTO log_cargas (fecha_desde, fecha_hasta, transacciones, estado)
        VALUES (%s, %s, %s, %s)
    """, (desde, hasta, total_transac, "exitoso"))

    print(f"\n   Log de carga registrado en Go_BD.")

# ─────────────────────────────────────────
# Ejecutar
# ─────────────────────────────────────────
def main():
    print("\n🔄 Iniciando carga incremental — Go_Retail")
    print(f"   Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    conn = None
    try:
        conn   = conectar()
        cursor = conn.cursor()
        print("✅ Conexión exitosa a Go_BD\n")

        # Obtener rango de fechas a cargar
        ultima_fecha = obtener_ultima_fecha(cursor)
        fecha_actual = datetime.now()

        # Verificar si hay días nuevos por cargar
        dias_pendientes = (fecha_actual - ultima_fecha).days
        if dias_pendientes <= 0:
            print("\n⚠️  No hay días nuevos por cargar. La data está al día.")
            return

        print(f"   Días pendientes por cargar: {dias_pendientes}\n")

        # Obtener tiendas y SKUs
        tiendas, skus = obtener_tiendas_skus(cursor)

        # Generar transacciones nuevas
        total = generar_transacciones_nuevas(
            cursor, tiendas, skus,
            desde=ultima_fecha,
            hasta=fecha_actual
        )
        conn.commit()
        print(f"   ✅ {total:,} transacciones nuevas cargadas")

        # Actualizar inventario
        actualizar_inventario(cursor, tiendas, skus)
        conn.commit()

        # Registrar log
        registrar_log(cursor, ultima_fecha, fecha_actual, total)
        conn.commit()

        print("\n✅ Carga incremental completada exitosamente.")
        print(f"   Período cargado: {ultima_fecha.strftime('%Y-%m-%d')} → {fecha_actual.strftime('%Y-%m-%d')}")
        print(f"   Total transacciones nuevas: {total:,}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("🔒 Conexión cerrada.")

if __name__ == "__main__":
    main()
