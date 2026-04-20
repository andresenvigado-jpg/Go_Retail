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
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from styles import STYLE, COLORS, PLOTLY_TEMPLATE, kpi, section, header

st.set_page_config(
    page_title="Go Retail — Inicio",
    page_icon="📦",
    layout="wide"
)
st.markdown(STYLE, unsafe_allow_html=True)
load_dotenv()

TIPO_TRANSAC   = ["venta", "reposicion", "devolucion", "traslado"]
TEMPORADA_ALTA = [1, 2, 6, 7, 10, 11, 12]

@st.cache_resource
def conectar_engine():
    host     = os.getenv("DB_HOST")
    dbname   = os.getenv("DB_NAME")
    user     = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port     = os.getenv("DB_PORT", "5432")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require")

def verificar_y_cargar():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"), dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432"), sslmode="require"
        )
        cursor = conn.cursor()
        hoy = datetime.now().date()
        cursor.execute("SELECT COUNT(*) FROM transacciones WHERE DATE(transaction_date) = %s", (hoy,))
        registros_hoy = cursor.fetchone()[0]
        if registros_hoy > 0:
            cursor.close(); conn.close()
            return False, registros_hoy, hoy

        cursor.execute("SELECT MAX(transaction_date) FROM transacciones")
        ultima = cursor.fetchone()[0]
        if ultima is None:
            ultima = datetime.now() - timedelta(days=3)

        cursor.execute("SELECT id FROM tiendas")
        tiendas = [str(r[0]) for r in cursor.fetchall()]
        cursor.execute("SELECT id FROM catalogos")
        skus = [str(r[0]) for r in cursor.fetchall()]

        fecha = ultima + timedelta(days=1)
        hasta = datetime.now()
        total = 0
        while fecha <= hasta:
            mes = fecha.month
            ventas_dia = random.randint(15, 30) if mes in TEMPORADA_ALTA else random.randint(5, 15)
            for _ in range(ventas_dia):
                tipo     = random.choices(TIPO_TRANSAC, weights=[70, 15, 10, 5])[0]
                cantidad = random.randint(1, 5) if tipo == "venta" else random.randint(5, 30)
                cursor.execute("""
                    INSERT INTO transacciones (
                        receipt_id, sku_id, source_location_id, target_location_id,
                        quantity, sale_price, currency, type, transaction_date, transaction_date_process
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (f"AUTO_{total:08d}", random.choice(skus), "BODEGA_CENTRAL",
                      random.choice(tiendas), cantidad,
                      round(random.uniform(30000, 350000), 2) if tipo == "venta" else 0,
                      "COP", tipo, fecha, datetime.now()))
                total += 1
            fecha += timedelta(days=1)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS log_cargas (
                id SERIAL PRIMARY KEY, fecha_ejecucion TIMESTAMP DEFAULT NOW(),
                fecha_desde TIMESTAMP, fecha_hasta TIMESTAMP,
                transacciones INTEGER, estado VARCHAR(50)
            )
        """)
        cursor.execute("INSERT INTO log_cargas (fecha_desde, fecha_hasta, transacciones, estado) VALUES (%s,%s,%s,%s)",
                       (ultima, hasta, total, "exitoso"))
        conn.commit(); cursor.close(); conn.close()
        return True, total, hoy
    except Exception as e:
        return False, 0, datetime.now().date()

# ── Header ──
st.markdown(header("📦 Go Retail", "Sistema de inteligencia para abastecimiento de puntos de venta · Retail Colombia"), unsafe_allow_html=True)

# ── Carga incremental ──
with st.spinner("Verificando data del día..."):
    cargado, total_nuevos, fecha_hoy = verificar_y_cargar()

if cargado:
    st.success(f"✅ Carga incremental ejecutada · {total_nuevos:,} transacciones nuevas · {fecha_hoy.strftime('%d/%m/%Y')}")
else:
    st.info(f"ℹ️ Data del día ya cargada · {fecha_hoy.strftime('%d/%m/%Y')} · {total_nuevos:,} registros existentes")

engine = conectar_engine()

# ── Cargar todas las tablas necesarias ──
df_anomalias   = pd.DataFrame()
df_seg_skus    = pd.DataFrame()
df_seg_tiendas = pd.DataFrame()
df_eoq         = pd.DataFrame()

try:
    df_anomalias   = pd.read_sql("SELECT * FROM anomalias_inventario", engine)
except Exception: pass
try:
    df_seg_skus    = pd.read_sql("SELECT * FROM segmentacion_skus", engine)
except Exception: pass
try:
    df_seg_tiendas = pd.read_sql("SELECT * FROM segmentacion_tiendas", engine)
except Exception: pass
try:
    df_eoq         = pd.read_sql("SELECT * FROM eoq_resultados ORDER BY eoq DESC", engine)
except Exception: pass

# ── KPIs principales ──
st.markdown(section("Resumen Ejecutivo"), unsafe_allow_html=True)

if not df_anomalias.empty and not df_seg_skus.empty and not df_eoq.empty:
    quiebres    = len(df_anomalias[df_anomalias["tipo_anomalia"] == "🔴 Quiebre de stock"])
    riesgos     = len(df_anomalias[df_anomalias["tipo_anomalia"] == "🟠 Riesgo de quiebre"])
    sin_mov     = len(df_anomalias[df_anomalias["tipo_anomalia"] == "🔵 Sin movimiento"])
    pedir_ahora = len(df_eoq[df_eoq["estado_reposicion"] == "🔴 Pedir ahora"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(kpi("Total SKUs",        len(df_seg_skus),    "success"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Tiendas activas",   len(df_seg_tiendas), "info"),    unsafe_allow_html=True)
    with c3: st.markdown(kpi("Quiebres stock",    quiebres,            "danger",  "Acción inmediata"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Riesgo quiebre",    riesgos,             "warning", "Acción urgente"),   unsafe_allow_html=True)
    with c5: st.markdown(kpi("Sin movimiento",    sin_mov,             "neutral", "Revisar viabilidad"), unsafe_allow_html=True)
    with c6: st.markdown(kpi("Pedir ahora (EOQ)", pedir_ahora,         "danger",  "Pedido inmediato"), unsafe_allow_html=True)
else:
    st.warning("Ejecuta los modelos para ver los KPIs.")

# ── Ventas históricas ──
st.markdown(section("Evolución de Ventas Históricas"), unsafe_allow_html=True)
try:
    df_ventas = pd.read_sql("""
        SELECT DATE(transaction_date) AS fecha, SUM(quantity) AS cantidad
        FROM transacciones WHERE type = 'venta'
        GROUP BY DATE(transaction_date) ORDER BY fecha
    """, engine)
    df_ventas["fecha"] = pd.to_datetime(df_ventas["fecha"])
    fig = px.area(df_ventas, x="fecha", y="cantidad",
                  color_discrete_sequence=[COLORS["primary"]],
                  labels={"fecha": "Fecha", "cantidad": "Unidades vendidas"})
    fig.update_layout(**PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=10, b=0), height=280)
    fig.update_traces(fillcolor="rgba(91,160,51,0.15)", line_color=COLORS["primary"])
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Sin datos de ventas disponibles.")

# ── Resumen de alertas ──
st.markdown(section("Resumen de Alertas"), unsafe_allow_html=True)

if not df_anomalias.empty and not df_seg_skus.empty:
    col1, col2 = st.columns(2)
    with col1:
        resumen_anom = (
            df_anomalias[df_anomalias["es_anomalia"]]
            .groupby("tipo_anomalia").size()
            .reset_index(name="cantidad")
            .sort_values("cantidad", ascending=False)
        )
        fig_anom = px.bar(resumen_anom, x="tipo_anomalia", y="cantidad",
                          color="tipo_anomalia",
                          color_discrete_sequence=[COLORS["danger"], COLORS["warning"],
                                                   COLORS["info"], COLORS["neutral"]],
                          labels={"tipo_anomalia": "", "cantidad": "Registros"},
                          title="Alertas de inventario")
        fig_anom.update_layout(**PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=40, b=0),
                               height=280, showlegend=False)
        st.plotly_chart(fig_anom, use_container_width=True)

    with col2:
        abc = df_seg_skus.groupby("segmento_abc").agg(skus=("sku_id","count")).reset_index()
        fig_abc = px.pie(abc, names="segmento_abc", values="skus",
                         color_discrete_sequence=[COLORS["primary"], COLORS["warning"], COLORS["danger"]],
                         hole=0.5, title="Segmentación ABC de SKUs")
        fig_abc.update_layout(**PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=40, b=0), height=280)
        st.plotly_chart(fig_abc, use_container_width=True)
else:
    st.info("Ejecuta los modelos para ver las alertas.")

# ── Footer ──
st.divider()
st.caption("Go Retail v3.0 · Softline S.A. · Modelos: Prophet · LightGBM · K-Means · Isolation Forest · EOQ · Monte Carlo · Market Basket · Rentabilidad · Rotación · Eficiencia")