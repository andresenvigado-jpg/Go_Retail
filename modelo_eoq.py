import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Conexión a Go_BD
# ─────────────────────────────────────────
load_dotenv()

def conectar_engine():
    host     = os.getenv("DB_HOST")
    dbname   = os.getenv("DB_NAME")
    user     = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port     = os.getenv("DB_PORT", "5432")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require")

# ─────────────────────────────────────────
# 1. Leer datos necesarios
# ─────────────────────────────────────────
def leer_datos(engine):
    print("📥 Leyendo datos desde Go_BD...\n")

    # Demanda real por SKU y tienda (últimos 12 meses)
    df_demanda = pd.read_sql("""
        SELECT
            sku_id,
            target_location_id      AS tienda_id,
            SUM(quantity)           AS demanda_anual,
            AVG(quantity)           AS demanda_diaria_prom,
            STDDEV(quantity)        AS desviacion_demanda,
            COUNT(DISTINCT DATE(transaction_date)) AS dias_con_venta
        FROM transacciones
        WHERE type = 'venta'
        GROUP BY sku_id, target_location_id
    """, engine)
    print(f"   ✅ Demanda: {len(df_demanda):,} combinaciones SKU-Tienda")

    # Inventario con lead time
    df_inv = pd.read_sql("""
        SELECT
            sku_id,
            location_id             AS tienda_id,
            replenishment_lead_time AS lead_time,
            min_stock,
            max_stock,
            site_qty
        FROM inventarios
    """, engine)
    print(f"   ✅ Inventarios: {len(df_inv):,} registros")

    # Costo del producto desde catálogo
    df_costo = pd.read_sql("""
        SELECT
            id::VARCHAR AS sku_id,
            cost,
            price,
            categories  AS categoria,
            brands      AS marca
        FROM catalogos
    """, engine)
    print(f"   ✅ Catálogo: {len(df_costo):,} productos\n")

    return df_demanda, df_inv, df_costo

# ─────────────────────────────────────────
# 2. Calcular EOQ
# ─────────────────────────────────────────
def calcular_eoq(df_demanda, df_inv, df_costo):
    print("🧮 Calculando EOQ por SKU y tienda...\n")

    # Unir datos
    df = df_demanda.merge(df_inv, on=["sku_id", "tienda_id"], how="left")
    df = df.merge(df_costo, on="sku_id", how="left")

    # Parámetros EOQ
    # Costo por pedido: estimado como el 2% del costo del producto
    # Tasa de almacenamiento: 20% anual del costo (estándar retail)
    df["costo_pedido"]        = (df["cost"] * 0.02).clip(lower=500)
    df["tasa_almacenamiento"] = 0.20
    df["costo_almacenamiento"] = df["cost"] * df["tasa_almacenamiento"]

    # Desviación mínima para evitar división por cero
    df["desviacion_demanda"] = df["desviacion_demanda"].fillna(1).clip(lower=0.1)
    df["lead_time"]          = df["lead_time"].fillna(3).clip(lower=1)
    df["demanda_anual"]      = df["demanda_anual"].fillna(1).clip(lower=1)
    df["demanda_diaria_prom"] = df["demanda_diaria_prom"].fillna(0.1).clip(lower=0.01)

    # ─── Fórmula EOQ ───
    # EOQ = √(2 × D × S / H)
    # D = Demanda anual
    # S = Costo por pedido
    # H = Costo de almacenamiento anual por unidad
    df["eoq"] = np.sqrt(
        (2 * df["demanda_anual"] * df["costo_pedido"]) /
        df["costo_almacenamiento"].clip(lower=1)
    ).round(0)

    # ─── Stock de seguridad ───
    # SS = Z × σ × √(Lead time)
    # Z = 1.65 para 95% de nivel de servicio
    Z = 1.65
    df["stock_seguridad"] = (
        Z * df["desviacion_demanda"] * np.sqrt(df["lead_time"])
    ).round(0)

    # ─── Punto de reorden ───
    # ROP = Demanda diaria × Lead time + Stock de seguridad
    df["punto_reorden"] = (
        df["demanda_diaria_prom"] * df["lead_time"] + df["stock_seguridad"]
    ).round(0)

    # ─── Costo total optimizado ───
    # CT = (D/EOQ) × S + (EOQ/2) × H
    df["costo_total_optimizado"] = (
        (df["demanda_anual"] / df["eoq"]) * df["costo_pedido"] +
        (df["eoq"] / 2) * df["costo_almacenamiento"]
    ).round(2)

    # ─── Frecuencia de pedido ───
    # Número de pedidos al año = D / EOQ
    df["pedidos_por_año"] = (df["demanda_anual"] / df["eoq"]).round(1)
    df["dias_entre_pedidos"] = (365 / df["pedidos_por_año"]).round(0)

    # ─── Estado de reposición ───
    def estado_reposicion(row):
        if row["site_qty"] <= row["punto_reorden"]:
            return "🔴 Pedir ahora"
        elif row["site_qty"] <= row["punto_reorden"] * 1.5:
            return "🟡 Pedir pronto"
        else:
            return "🟢 Stock OK"

    df["estado_reposicion"] = df.apply(estado_reposicion, axis=1)

    print(f"   ✅ EOQ calculado para {len(df):,} combinaciones SKU-Tienda")

    # Resumen por estado
    resumen = df.groupby("estado_reposicion").size().reset_index(name="cantidad")
    print("\n📊 Resumen de estados de reposición:")
    print("─" * 45)
    for _, row in resumen.iterrows():
        print(f"  {row['estado_reposicion']:30} | {row['cantidad']:>4} SKUs")
    print("─" * 45)

    return df

# ─────────────────────────────────────────
# 3. Mostrar top urgentes
# ─────────────────────────────────────────
def mostrar_urgentes(df):
    urgentes = df[df["estado_reposicion"] == "🔴 Pedir ahora"].nlargest(10, "demanda_anual")

    if not urgentes.empty:
        print("\n🚨 TOP 10 SKUs que necesitan pedido inmediato:")
        print("─" * 75)
        for _, row in urgentes.iterrows():
            print(f"  SKU {row['sku_id']:>6} | Tienda {row['tienda_id']:>4} | "
                  f"EOQ: {row['eoq']:>6.0f} und | "
                  f"ROP: {row['punto_reorden']:>5.0f} | "
                  f"Stock: {row['site_qty']:>5.0f} | "
                  f"SS: {row['stock_seguridad']:>4.0f}")
        print("─" * 75)

# ─────────────────────────────────────────
# 4. Guardar en Go_BD
# ─────────────────────────────────────────
def guardar_eoq(engine, df):
    print("\n💾 Guardando resultados EOQ en Go_BD...")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS eoq_resultados (
                id                      SERIAL PRIMARY KEY,
                sku_id                  VARCHAR(100),
                tienda_id               VARCHAR(100),
                categoria               VARCHAR(255),
                marca                   VARCHAR(255),
                demanda_anual           NUMERIC(12,2),
                demanda_diaria_prom     NUMERIC(12,4),
                eoq                     NUMERIC(12,2),
                stock_seguridad         NUMERIC(12,2),
                punto_reorden           NUMERIC(12,2),
                costo_total_optimizado  NUMERIC(12,2),
                pedidos_por_año         NUMERIC(12,2),
                dias_entre_pedidos      NUMERIC(12,2),
                estado_reposicion       VARCHAR(50),
                site_qty                NUMERIC(12,2),
                lead_time               INTEGER,
                fecha_calculo           TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()

    columnas = [
        "sku_id", "tienda_id", "categoria", "marca",
        "demanda_anual", "demanda_diaria_prom",
        "eoq", "stock_seguridad", "punto_reorden",
        "costo_total_optimizado", "pedidos_por_año",
        "dias_entre_pedidos", "estado_reposicion",
        "site_qty", "lead_time"
    ]

    df[columnas].to_sql("eoq_resultados", engine, if_exists="replace", index=False)
    print(f"   ✅ {len(df):,} registros guardados en tabla 'eoq_resultados'")

# ─────────────────────────────────────────
# Ejecutar
# ─────────────────────────────────────────
def main():
    print("\n🚀 Iniciando cálculo EOQ — Go_Retail\n")

    engine                    = conectar_engine()
    df_demanda, df_inv, df_costo = leer_datos(engine)
    df_eoq                    = calcular_eoq(df_demanda, df_inv, df_costo)

    mostrar_urgentes(df_eoq)
    guardar_eoq(engine, df_eoq)

    print("\n✅ Proceso EOQ completado.")
    print("   Tabla guardada en Go_BD: eoq_resultados")

if __name__ == "__main__":
    main()
