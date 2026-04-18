import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
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
# 1. Leer transacciones
# ─────────────────────────────────────────
def leer_transacciones(engine):
    print("📥 Leyendo transacciones desde Go_BD...\n")

    df = pd.read_sql("""
        SELECT
            receipt_id,
            sku_id,
            target_location_id AS tienda_id
        FROM transacciones
        WHERE type = 'venta'
        AND receipt_id IS NOT NULL
    """, engine)

    print(f"   ✅ {len(df):,} transacciones de venta leídas")
    print(f"   ✅ {df['receipt_id'].nunique():,} recibos únicos")
    print(f"   ✅ {df['sku_id'].nunique():,} SKUs distintos\n")

    return df

# ─────────────────────────────────────────
# 2. Preparar matriz de transacciones
# ─────────────────────────────────────────
def preparar_matriz(df):
    print("🔧 Preparando matriz de transacciones...")

    # Agrupar SKUs por recibo (canasta de compra)
    canastas = df.groupby("receipt_id")["sku_id"].apply(list).reset_index()
    canastas.columns = ["receipt_id", "skus"]

    # Filtrar canastas con más de 1 producto
    canastas = canastas[canastas["skus"].apply(len) > 1]
    print(f"   ✅ Canastas con 2 o más productos: {len(canastas):,}")

    if len(canastas) < 10:
        print("   ⚠️  Pocas canastas multi-producto. Ajustando agrupación por tienda...")
        # Agrupar por tienda y día para simular canastas
        df["fecha"] = df["tienda_id"].astype(str) + "_" + df["receipt_id"].str[:6]
        canastas = df.groupby("fecha")["sku_id"].apply(list).reset_index()
        canastas.columns = ["grupo", "skus"]
        canastas = canastas[canastas["skus"].apply(len) > 1]
        print(f"   ✅ Grupos ajustados: {len(canastas):,}")
        lista_transacciones = canastas["skus"].tolist()
    else:
        lista_transacciones = canastas["skus"].tolist()

    # Limitar a top 50 SKUs para evitar MemoryError
    from collections import Counter
    todos_skus = [sku for canasta in lista_transacciones for sku in canasta]
    top_skus   = set(sku for sku, _ in Counter(todos_skus).most_common(50))
    lista_transacciones = [
        [sku for sku in canasta if sku in top_skus]
        for canasta in lista_transacciones
    ]
    lista_transacciones = [c for c in lista_transacciones if len(c) > 1]
    print(f"   ✅ Análisis limitado a top 50 SKUs más frecuentes")
    print(f"   ✅ Canastas válidas tras filtro: {len(lista_transacciones):,}")

    # Codificar con TransactionEncoder
    te        = TransactionEncoder()
    te_arr    = te.fit(lista_transacciones).transform(lista_transacciones)
    df_matrix = pd.DataFrame(te_arr, columns=te.columns_)

    print(f"   ✅ Matriz generada: {df_matrix.shape[0]} transacciones × {df_matrix.shape[1]} SKUs\n")
    return df_matrix

# ─────────────────────────────────────────
# 3. Aplicar Apriori
# ─────────────────────────────────────────
def aplicar_apriori(df_matrix):
    print("🤖 Aplicando algoritmo Apriori...")

    # Probar diferentes valores de soporte hasta encontrar reglas
    for min_sup in [0.05, 0.03, 0.02, 0.01]:
        itemsets = apriori(df_matrix, min_support=min_sup, use_colnames=True)
        if len(itemsets) > 0:
            print(f"   ✅ Itemsets frecuentes encontrados: {len(itemsets):,} (soporte mínimo: {min_sup})")
            break
    else:
        print("   ⚠️  No se encontraron itemsets frecuentes con los parámetros actuales.")
        return pd.DataFrame()

    # Generar reglas de asociación
    reglas = association_rules(itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(itemsets))

    if reglas.empty:
        print("   ⚠️  No se generaron reglas de asociación.")
        return pd.DataFrame()

    # Limpiar y ordenar
    reglas["antecedents"] = reglas["antecedents"].apply(lambda x: ", ".join(list(x)))
    reglas["consequents"] = reglas["consequents"].apply(lambda x: ", ".join(list(x)))
    reglas = reglas.sort_values("lift", ascending=False).reset_index(drop=True)

    print(f"   ✅ Reglas de asociación generadas: {len(reglas):,}\n")
    return reglas

# ─────────────────────────────────────────
# 4. Mostrar resultados
# ─────────────────────────────────────────
def mostrar_resultados(reglas):
    if reglas.empty:
        return

    print("📊 TOP 15 Reglas de asociación más fuertes:")
    print("─" * 75)
    print(f"  {'Si compra SKU':>15} → {'También lleva SKU':>18} | {'Confianza':>10} | {'Lift':>6} | {'Soporte':>8}")
    print("─" * 75)

    for _, row in reglas.head(15).iterrows():
        print(f"  SKU {row['antecedents']:>10} → SKU {row['consequents']:>12} | "
              f"{row['confidence']*100:>9.1f}% | "
              f"{row['lift']:>6.2f} | "
              f"{row['support']*100:>7.1f}%")

    print("─" * 75)
    print("\n📖 Cómo interpretar:")
    print("   Confianza: % de veces que se compran juntos")
    print("   Lift > 1 : relación real (no casual)")
    print("   Lift > 2 : relación fuerte")
    print("   Lift > 3 : relación muy fuerte\n")

# ─────────────────────────────────────────
# 5. Guardar en Go_BD
# ─────────────────────────────────────────
def guardar_reglas(engine, reglas):
    if reglas.empty:
        print("⚠️  Sin reglas para guardar.")
        return

    print("💾 Guardando reglas de asociación en Go_BD...")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_basket (
                id              SERIAL PRIMARY KEY,
                sku_origen      VARCHAR(255),
                sku_destino     VARCHAR(255),
                soporte         NUMERIC(10,4),
                confianza       NUMERIC(10,4),
                lift            NUMERIC(10,4),
                conviction      NUMERIC(10,4),
                fecha_calculo   TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()

    df_save = reglas[[
        "antecedents", "consequents",
        "support", "confidence", "lift", "conviction"
    ]].copy()

    df_save.columns = [
        "sku_origen", "sku_destino",
        "soporte", "confianza", "lift", "conviction"
    ]

    df_save["conviction"] = df_save["conviction"].replace([np.inf, -np.inf], 9999).round(4)
    df_save.to_sql("market_basket", engine, if_exists="replace", index=False)
    print(f"   ✅ {len(df_save):,} reglas guardadas en tabla 'market_basket'")

# ─────────────────────────────────────────
# Ejecutar
# ─────────────────────────────────────────
def main():
    print("\n🚀 Iniciando Market Basket Analysis — Go_Retail\n")

    engine   = conectar_engine()
    df       = leer_transacciones(engine)
    df_matrix = preparar_matriz(df)

    if df_matrix.empty:
        print("❌ No hay datos suficientes para el análisis.")
        return

    reglas   = aplicar_apriori(df_matrix)
    mostrar_resultados(reglas)
    guardar_reglas(engine, reglas)

    print("\n✅ Market Basket Analysis completado.")
    print("   Tabla guardada en Go_BD: market_basket")

if __name__ == "__main__":
    main()
