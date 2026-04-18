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
# 1. Leer datos
# ─────────────────────────────────────────
def leer_datos(engine):
    print("📥 Leyendo datos desde Go_BD...\n")

    df_ventas = pd.read_sql("""
        SELECT
            sku_id,
            target_location_id       AS tienda_id,
            DATE(transaction_date)   AS fecha,
            SUM(quantity)            AS cantidad
        FROM transacciones
        WHERE type = 'venta'
        GROUP BY sku_id, target_location_id, DATE(transaction_date)
        ORDER BY sku_id, tienda_id, fecha
    """, engine)
    print(f"   ✅ Ventas diarias: {len(df_ventas):,} registros")

    df_inv = pd.read_sql("""
        SELECT
            sku_id,
            location_id              AS tienda_id,
            site_qty,
            min_stock,
            replenishment_lead_time  AS lead_time
        FROM inventarios
    """, engine)
    print(f"   ✅ Inventarios: {len(df_inv):,} registros\n")

    return df_ventas, df_inv

# ─────────────────────────────────────────
# 2. Simulación Monte Carlo por SKU-Tienda
# ─────────────────────────────────────────
def simular_monte_carlo(df_ventas, df_inv, n_simulaciones=1000, dias_simulacion=30):
    print(f"🎲 Ejecutando {n_simulaciones:,} simulaciones por SKU-Tienda ({dias_simulacion} días)...\n")

    # Top SKUs-Tiendas con más historial
    grupos = (
        df_ventas.groupby(["sku_id", "tienda_id"])
        .agg(
            media    =("cantidad", "mean"),
            std      =("cantidad", "std"),
            registros=("cantidad", "count")
        )
        .reset_index()
    )

    # Filtrar grupos con suficiente historial
    grupos = grupos[grupos["registros"] >= 5].copy()
    grupos["std"] = grupos["std"].fillna(1).clip(lower=0.1)

    # Tomar top 50 combinaciones más activas
    grupos = grupos.nlargest(50, "registros").reset_index(drop=True)
    print(f"   ✅ Simulando {len(grupos):,} combinaciones SKU-Tienda\n")

    resultados = []
    np.random.seed(42)

    for _, row in grupos.iterrows():
        sku_id    = row["sku_id"]
        tienda_id = row["tienda_id"]
        media     = row["media"]
        std       = row["std"]

        # Obtener inventario actual
        inv = df_inv[
            (df_inv["sku_id"] == sku_id) &
            (df_inv["tienda_id"] == tienda_id)
        ]
        stock_actual = inv["site_qty"].values[0]  if len(inv) > 0 else media * 5
        lead_time    = inv["lead_time"].values[0] if len(inv) > 0 else 3

        # Simular n_simulaciones escenarios de demanda
        demanda_simulada = np.random.normal(
            loc=media,
            scale=std,
            size=(n_simulaciones, dias_simulacion)
        ).clip(min=0)

        demanda_total = demanda_simulada.sum(axis=1)

        # Calcular métricas
        p50  = np.percentile(demanda_total, 50)
        p90  = np.percentile(demanda_total, 90)
        p95  = np.percentile(demanda_total, 95)
        p99  = np.percentile(demanda_total, 99)

        prob_quiebre = (demanda_total > stock_actual).mean() * 100

        # Stock recomendado para cubrir P95
        stock_recomendado = p95

        # Días hasta quiebre esperado
        stock_diario = stock_actual / dias_simulacion if dias_simulacion > 0 else 0
        dias_cobertura = stock_actual / media if media > 0 else 999

        # Nivel de riesgo
        if prob_quiebre >= 70:
            nivel_riesgo = "🔴 Riesgo alto"
        elif prob_quiebre >= 40:
            nivel_riesgo = "🟡 Riesgo medio"
        elif prob_quiebre >= 15:
            nivel_riesgo = "🟠 Riesgo bajo-medio"
        else:
            nivel_riesgo = "🟢 Riesgo bajo"

        resultados.append({
            "sku_id":             sku_id,
            "tienda_id":          tienda_id,
            "stock_actual":       round(stock_actual, 2),
            "lead_time":          int(lead_time),
            "demanda_media_diaria": round(media, 2),
            "demanda_p50":        round(p50,  2),
            "demanda_p90":        round(p90,  2),
            "demanda_p95":        round(p95,  2),
            "demanda_p99":        round(p99,  2),
            "prob_quiebre":       round(prob_quiebre, 2),
            "stock_recomendado":  round(stock_recomendado, 2),
            "dias_cobertura":     round(dias_cobertura, 1),
            "nivel_riesgo":       nivel_riesgo,
            "n_simulaciones":     n_simulaciones,
            "dias_simulacion":    dias_simulacion
        })

    df_result = pd.DataFrame(resultados)
    print(f"   ✅ Simulaciones completadas: {len(df_result):,} resultados\n")
    return df_result

# ─────────────────────────────────────────
# 3. Mostrar resumen
# ─────────────────────────────────────────
def mostrar_resumen(df):
    print("📊 Resumen de riesgos por nivel:")
    print("─" * 50)
    resumen = df.groupby("nivel_riesgo").size().reset_index(name="cantidad")
    for _, row in resumen.iterrows():
        print(f"  {row['nivel_riesgo']:30} | {row['cantidad']:>4} combinaciones")
    print("─" * 50)

    criticos = df[df["nivel_riesgo"] == "🔴 Riesgo alto"].nlargest(10, "prob_quiebre")
    if not criticos.empty:
        print("\n🚨 TOP 10 SKUs con mayor probabilidad de quiebre:")
        print("─" * 75)
        for _, row in criticos.iterrows():
            print(f"  SKU {row['sku_id']:>6} | Tienda {row['tienda_id']:>4} | "
                  f"Prob. quiebre: {row['prob_quiebre']:>5.1f}% | "
                  f"Stock: {row['stock_actual']:>6.0f} | "
                  f"P95: {row['demanda_p95']:>6.0f} | "
                  f"Cobertura: {row['dias_cobertura']:>4.1f} días")
        print("─" * 75)

# ─────────────────────────────────────────
# 4. Guardar en Go_BD
# ─────────────────────────────────────────
def guardar_resultados(engine, df):
    print("\n💾 Guardando resultados Monte Carlo en Go_BD...")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS monte_carlo (
                id                      SERIAL PRIMARY KEY,
                sku_id                  VARCHAR(100),
                tienda_id               VARCHAR(100),
                stock_actual            NUMERIC(12,2),
                lead_time               INTEGER,
                demanda_media_diaria    NUMERIC(12,4),
                demanda_p50             NUMERIC(12,2),
                demanda_p90             NUMERIC(12,2),
                demanda_p95             NUMERIC(12,2),
                demanda_p99             NUMERIC(12,2),
                prob_quiebre            NUMERIC(10,2),
                stock_recomendado       NUMERIC(12,2),
                dias_cobertura          NUMERIC(10,2),
                nivel_riesgo            VARCHAR(50),
                n_simulaciones          INTEGER,
                dias_simulacion         INTEGER,
                fecha_calculo           TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()

    df.to_sql("monte_carlo", engine, if_exists="replace", index=False)
    print(f"   ✅ {len(df):,} registros guardados en tabla 'monte_carlo'")

# ─────────────────────────────────────────
# Ejecutar
# ─────────────────────────────────────────
def main():
    print("\n🚀 Iniciando Simulación Monte Carlo — Go_Retail\n")

    engine              = conectar_engine()
    df_ventas, df_inv   = leer_datos(engine)
    df_result           = simular_monte_carlo(df_ventas, df_inv, n_simulaciones=1000, dias_simulacion=30)

    mostrar_resumen(df_result)
    guardar_resultados(engine, df_result)

    print("\n✅ Simulación Monte Carlo completada.")
    print("   Tabla guardada en Go_BD: monte_carlo")

if __name__ == "__main__":
    main()
