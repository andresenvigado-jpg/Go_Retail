import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import random
import psycopg2
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ─────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Go Retail — Tablero de Abastecimiento",
    page_icon="📦",
    layout="wide"
)

# ─────────────────────────────────────────
# Conexión a Go_BD
# ─────────────────────────────────────────
load_dotenv()

@st.cache_resource
def conectar_engine():
    host     = os.getenv("DB_HOST")
    dbname   = os.getenv("DB_NAME")
    user     = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port     = os.getenv("DB_PORT", "5432")
    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
    )

# ─────────────────────────────────────────
# Carga incremental automática
# ─────────────────────────────────────────
TIPO_TRANSAC   = ["venta", "reposicion", "devolucion", "traslado"]
TEMPORADA_ALTA = [1, 2, 6, 7, 10, 11, 12]

def verificar_y_cargar():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432"),
            sslmode="require"
        )
        cursor = conn.cursor()

        # Verificar si ya hay data del día actual
        hoy = datetime.now().date()
        cursor.execute("""
            SELECT COUNT(*) FROM transacciones
            WHERE DATE(transaction_date) = %s
        """, (hoy,))
        registros_hoy = cursor.fetchone()[0]

        if registros_hoy > 0:
            cursor.close()
            conn.close()
            return False, registros_hoy, hoy

        # No hay data de hoy → ejecutar carga incremental
        cursor.execute("SELECT MAX(transaction_date) FROM transacciones")
        ultima = cursor.fetchone()[0]
        if ultima is None:
            ultima = datetime.now() - timedelta(days=3)

        cursor.execute("SELECT id FROM tiendas")
        tiendas = [str(r[0]) for r in cursor.fetchall()]

        cursor.execute("SELECT id FROM catalogos")
        skus = [str(r[0]) for r in cursor.fetchall()]

        fecha    = ultima + timedelta(days=1)
        hasta    = datetime.now()
        total    = 0

        while fecha <= hasta:
            mes        = fecha.month
            ventas_dia = random.randint(15, 30) if mes in TEMPORADA_ALTA else random.randint(5, 15)
            for _ in range(ventas_dia):
                tipo     = random.choices(TIPO_TRANSAC, weights=[70, 15, 10, 5])[0]
                cantidad = random.randint(1, 5) if tipo == "venta" else random.randint(5, 30)
                cursor.execute("""
                    INSERT INTO transacciones (
                        receipt_id, sku_id, source_location_id, target_location_id,
                        quantity, sale_price, currency, type,
                        transaction_date, transaction_date_process
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    f"AUTO_{total:08d}",
                    random.choice(skus),
                    "BODEGA_CENTRAL",
                    random.choice(tiendas),
                    cantidad,
                    round(random.uniform(30000, 350000), 2) if tipo == "venta" else 0,
                    "COP", tipo, fecha, datetime.now()
                ))
                total += 1
            fecha += timedelta(days=1)

        # Registrar log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS log_cargas (
                id SERIAL PRIMARY KEY,
                fecha_ejecucion TIMESTAMP DEFAULT NOW(),
                fecha_desde TIMESTAMP,
                fecha_hasta TIMESTAMP,
                transacciones INTEGER,
                estado VARCHAR(50)
            )
        """)
        cursor.execute("""
            INSERT INTO log_cargas (fecha_desde, fecha_hasta, transacciones, estado)
            VALUES (%s, %s, %s, %s)
        """, (ultima, hasta, total, "exitoso"))

        conn.commit()
        cursor.close()
        conn.close()
        return True, total, hoy

    except Exception as e:
        st.warning(f"⚠️ Carga incremental: {e}")
        return False, 0, datetime.now().date()

@st.cache_data(ttl=3600)
def cargar_datos():
    engine = conectar_engine()

    anomalias = pd.read_sql("""
        SELECT * FROM anomalias_inventario
    """, engine)

    pronosticos = pd.read_sql("""
        SELECT * FROM pronosticos
        ORDER BY fecha
    """, engine)

    segmentacion_skus = pd.read_sql("""
        SELECT sku_id, segmento_abc, cantidad_total,
               cantidad_promedio, precio_promedio, num_tiendas
        FROM segmentacion_skus
    """, engine)

    segmentacion_tiendas = pd.read_sql("""
        SELECT tienda_id, segmento_tienda, ventas_totales,
               venta_promedio, num_skus, ciudad, clima, zona
        FROM segmentacion_tiendas
    """, engine)

    transacciones = pd.read_sql("""
        SELECT DATE(transaction_date) AS fecha,
               SUM(quantity)          AS cantidad,
               type
        FROM transacciones
        WHERE type = 'venta'
        GROUP BY DATE(transaction_date), type
        ORDER BY fecha
    """, engine)

    return anomalias, pronosticos, segmentacion_skus, segmentacion_tiendas, transacciones

# ─────────────────────────────────────────
# Cargar datos
# ─────────────────────────────────────────
try:
    anomalias, pronosticos, seg_skus, seg_tiendas, transacciones = cargar_datos()
    conexion_ok = True
except Exception as e:
    st.error(f"Error conectando a Go_BD: {e}")
    conexion_ok = False
    st.stop()

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("📦 Go Retail — Tablero de Abastecimiento")

# ─────────────────────────────────────────
# Validación y carga incremental automática
# ─────────────────────────────────────────
with st.spinner("Verificando data del día..."):
    cargado, total_nuevos, fecha_hoy = verificar_y_cargar()

if cargado:
    st.success(f"✅ Carga incremental ejecutada · {total_nuevos:,} transacciones nuevas agregadas · {fecha_hoy.strftime('%Y-%m-%d')}")
else:
    st.info(f"ℹ️ Data del día ya está cargada · {fecha_hoy.strftime('%Y-%m-%d')} · {total_nuevos:,} registros existentes")

st.caption("Actualizado 2 veces por semana · Go_BD en Neon PostgreSQL")
st.divider()

# ─────────────────────────────────────────
# Métricas principales
# ─────────────────────────────────────────
quiebres     = anomalias[anomalias["tipo_anomalia"] == "🔴 Quiebre de stock"]
riesgos      = anomalias[anomalias["tipo_anomalia"] == "🟠 Riesgo de quiebre"]
sin_mov      = anomalias[anomalias["tipo_anomalia"] == "🔵 Sin movimiento"]
total_anomal = anomalias[anomalias["es_anomalia"] == True]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total SKUs",          len(seg_skus))
col2.metric("Tiendas activas",     len(seg_tiendas))
col3.metric("Quiebres de stock",   len(quiebres),  delta=f"-{len(quiebres)} críticos",  delta_color="inverse")
col4.metric("Riesgo de quiebre",   len(riesgos),   delta=f"-{len(riesgos)} urgentes",   delta_color="inverse")
col5.metric("Sin movimiento",      len(sin_mov),   delta=f"{len(sin_mov)} SKUs",        delta_color="off")

st.divider()

# ─────────────────────────────────────────
# Fila 1: Ventas históricas + Pronóstico
# ─────────────────────────────────────────
col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("📈 Ventas históricas")
    if not transacciones.empty:
        transacciones["fecha"] = pd.to_datetime(transacciones["fecha"])
        fig_ventas = px.area(
            transacciones,
            x="fecha",
            y="cantidad",
            color_discrete_sequence=["#378ADD"],
            labels={"fecha": "Fecha", "cantidad": "Unidades vendidas"}
        )
        fig_ventas.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_ventas, use_container_width=True)

with col_der:
    st.subheader("🔮 Pronóstico 30 días (Prophet)")
    if not pronosticos.empty:
        pronosticos["fecha"] = pd.to_datetime(pronosticos["fecha"])
        top_skus = pronosticos.groupby("sku_id")["demanda_estimada"].sum().nlargest(5).index
        df_top   = pronosticos[pronosticos["sku_id"].isin(top_skus)]
        fig_pron = px.line(
            df_top,
            x="fecha",
            y="demanda_estimada",
            color="sku_id",
            labels={"fecha": "Fecha", "demanda_estimada": "Unidades estimadas", "sku_id": "SKU"}
        )
        fig_pron.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=280
        )
        st.plotly_chart(fig_pron, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# Fila 2: Anomalías + Segmentación ABC
# ─────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🚨 Alertas de inventario")

    resumen_anomalias = (
        anomalias[anomalias["es_anomalia"]]
        .groupby("tipo_anomalia")
        .size()
        .reset_index(name="cantidad")
        .sort_values("cantidad", ascending=False)
    )

    fig_anom = px.bar(
        resumen_anomalias,
        x="tipo_anomalia",
        y="cantidad",
        color="tipo_anomalia",
        color_discrete_sequence=["#E24B4A", "#EF9F27", "#378ADD", "#888780"],
        labels={"tipo_anomalia": "Tipo", "cantidad": "Registros"}
    )
    fig_anom.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
        showlegend=False
    )
    st.plotly_chart(fig_anom, use_container_width=True)

with col_b:
    st.subheader("🗂️ Segmentación ABC de SKUs")

    abc_resumen = (
        seg_skus.groupby("segmento_abc")
        .agg(skus=("sku_id", "count"), ventas=("cantidad_total", "sum"))
        .reset_index()
    )

    fig_abc = px.pie(
        abc_resumen,
        names="segmento_abc",
        values="skus",
        color_discrete_sequence=["#639922", "#BA7517", "#E24B4A"],
        hole=0.45
    )
    fig_abc.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=260
    )
    st.plotly_chart(fig_abc, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# Fila 3: Tiendas + Quiebres detalle
# ─────────────────────────────────────────
col_t, col_q = st.columns(2)

with col_t:
    st.subheader("🏪 Ventas por segmento de tienda")
    if not seg_tiendas.empty:
        fig_tiendas = px.bar(
            seg_tiendas.sort_values("ventas_totales", ascending=True),
            x="ventas_totales",
            y="tienda_id",
            color="segmento_tienda",
            orientation="h",
            labels={"ventas_totales": "Ventas totales", "tienda_id": "Tienda", "segmento_tienda": "Segmento"},
            color_discrete_sequence=["#E24B4A", "#EF9F27", "#5DCAA5", "#888780"]
        )
        fig_tiendas.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
            showlegend=True
        )
        st.plotly_chart(fig_tiendas, use_container_width=True)

with col_q:
    st.subheader("🔴 Detalle quiebres de stock")
    if not quiebres.empty:
        df_q = quiebres[["tienda_id", "sku_id", "site_qty", "min_stock", "cobertura_dias"]].copy()
        df_q.columns = ["Tienda", "SKU", "Stock actual", "Stock mínimo", "Cobertura (días)"]
        df_q["Cobertura (días)"] = df_q["Cobertura (días)"].round(1)
        st.dataframe(
            df_q.sort_values("Stock actual").head(15),
            use_container_width=True,
            height=320,
            hide_index=True
        )

st.divider()

# ─────────────────────────────────────────
# Fila 4: Top SKUs por demanda estimada
# ─────────────────────────────────────────
st.subheader("📊 Top SKUs — Demanda estimada próximos 30 días")

if not pronosticos.empty:
    resumen_pron = (
        pronosticos.groupby("sku_id")
        .agg(
            demanda_total=("demanda_estimada", "sum"),
            demanda_diaria=("demanda_estimada", "mean"),
            demanda_max=("demanda_maxima", "max")
        )
        .round(1)
        .reset_index()
        .sort_values("demanda_total", ascending=False)
        .head(10)
    )
    resumen_pron.columns = ["SKU", "Demanda total", "Promedio diario", "Máximo estimado"]

    fig_top = px.bar(
        resumen_pron,
        x="SKU",
        y="Demanda total",
        color="Demanda total",
        color_continuous_scale="Blues",
        labels={"Demanda total": "Unidades estimadas"}
    )
    fig_top.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
        showlegend=False,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.caption("Go Retail · Modelos: Prophet · LightGBM · K-Means · Isolation Forest · Base de datos: Neon PostgreSQL")
