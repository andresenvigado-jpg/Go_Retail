import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import STYLE, COLORS, PLOTLY_TEMPLATE, kpi, section, header

st.set_page_config(page_title="Go Retail — Inventario", page_icon="📦", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)
load_dotenv()

@st.cache_resource
def conectar_engine():
    host     = os.getenv("DB_HOST")
    dbname   = os.getenv("DB_NAME")
    user     = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port     = os.getenv("DB_PORT", "5432")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require")

st.markdown(header("📦 Gestión de Inventario", "Anomalías · EOQ · Simulación Monte Carlo"), unsafe_allow_html=True)
engine = conectar_engine()

tab1, tab2, tab3 = st.tabs(["🚨 Anomalías", "🛒 EOQ — Pedido Óptimo", "🎲 Monte Carlo — Riesgo"])

# ── Tab 1: Anomalías ──
with tab1:
    st.markdown(section("Detección de Anomalías — Isolation Forest"), unsafe_allow_html=True)
    try:
        df_an = pd.read_sql("SELECT * FROM anomalias_inventario", engine)
        quiebres   = len(df_an[df_an["tipo_anomalia"] == "🔴 Quiebre de stock"])
        riesgos    = len(df_an[df_an["tipo_anomalia"] == "🟠 Riesgo de quiebre"])
        sobrestock = len(df_an[df_an["tipo_anomalia"] == "🟡 Sobrestock"])
        sin_mov    = len(df_an[df_an["tipo_anomalia"] == "🔵 Sin movimiento"])

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(kpi("Quiebres de stock", quiebres,   "danger",  "Acción inmediata"),       unsafe_allow_html=True)
        with c2: st.markdown(kpi("Riesgo de quiebre", riesgos,    "warning", "Acción urgente"),          unsafe_allow_html=True)
        with c3: st.markdown(kpi("Sobrestock",         sobrestock, "info",    "Revisar pedidos"),         unsafe_allow_html=True)
        with c4: st.markdown(kpi("Sin movimiento",     sin_mov,    "neutral", "Evaluar discontinuar"),    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            resumen = df_an[df_an["es_anomalia"]].groupby("tipo_anomalia").size().reset_index(name="cantidad")
            fig = px.bar(resumen, x="tipo_anomalia", y="cantidad", color="tipo_anomalia",
                         color_discrete_sequence=[COLORS["danger"], COLORS["warning"], COLORS["info"], COLORS["neutral"]],
                         labels={"tipo_anomalia": "", "cantidad": "Registros"})
            fig.update_layout(**PLOTLY_TEMPLATE, margin=dict(l=0,r=0,t=10,b=0), height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            filtro = st.selectbox("Filtrar por tipo", ["Todos"] + df_an["tipo_anomalia"].unique().tolist())
            df_fil = df_an if filtro == "Todos" else df_an[df_an["tipo_anomalia"] == filtro]
            df_show = df_fil[["tienda_id","sku_id","site_qty","min_stock","cobertura_dias","tipo_anomalia"]].copy()
            df_show.columns = ["Tienda","SKU","Stock","Mínimo","Cobertura días","Tipo"]
            df_show["Cobertura días"] = df_show["Cobertura días"].round(1)
            st.dataframe(df_show.sort_values("Stock").head(20), use_container_width=True, hide_index=True, height=300)
    except Exception:
        st.info("ℹ️ Ejecuta modelo_anomalias.py para ver las anomalías.")

# ── Tab 2: EOQ ──
with tab2:
    st.markdown(section("EOQ — Cantidad Óptima de Pedido"), unsafe_allow_html=True)
    try:
        df_eoq = pd.read_sql("SELECT * FROM eoq_resultados ORDER BY eoq DESC", engine)

        pedir_ahora  = len(df_eoq[df_eoq["estado_reposicion"] == "🔴 Pedir ahora"])
        pedir_pronto = len(df_eoq[df_eoq["estado_reposicion"] == "🟡 Pedir pronto"])
        stock_ok     = len(df_eoq[df_eoq["estado_reposicion"] == "🟢 Stock OK"])
        costo_total  = df_eoq["costo_total_optimizado"].sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(kpi("Pedir ahora",      pedir_ahora,               "danger",  "Urgente"),   unsafe_allow_html=True)
        with c2: st.markdown(kpi("Pedir pronto",     pedir_pronto,              "warning", "Próximo"),   unsafe_allow_html=True)
        with c3: st.markdown(kpi("Stock OK",          stock_ok,                  "success", "Estable"),   unsafe_allow_html=True)
        with c4: st.markdown(kpi("Costo optimizado", f"${costo_total:,.0f}",    "info",    "COP total"), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 SKUs que necesitan pedido inmediato**")
            urgentes = df_eoq[df_eoq["estado_reposicion"] == "🔴 Pedir ahora"][
                ["sku_id","tienda_id","eoq","punto_reorden","site_qty","dias_entre_pedidos"]
            ].nlargest(15, "eoq")
            urgentes.columns = ["SKU","Tienda","EOQ","Punto reorden","Stock actual","Días entre pedidos"]
            st.dataframe(urgentes, use_container_width=True, hide_index=True, height=360)

        with col2:
            df_cat = df_eoq.groupby("categoria").agg(
                eoq_prom=("eoq","mean"), skus=("sku_id","nunique")
            ).round(1).reset_index().sort_values("eoq_prom", ascending=False)
            fig = px.bar(df_cat, x="categoria", y="eoq_prom",
                         color="eoq_prom", color_continuous_scale=["#C8E6B0","#5BA033","#3D7A1F"],
                         labels={"categoria": "Categoría", "eoq_prom": "EOQ promedio"})
            fig.update_layout(**PLOTLY_TEMPLATE, margin=dict(l=0,r=0,t=10,b=0),
                              height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("ℹ️ Ejecuta modelo_eoq.py para ver el EOQ.")

# ── Tab 3: Monte Carlo ──
with tab3:
    st.markdown(section("Monte Carlo — Simulación de Riesgo de Quiebre"), unsafe_allow_html=True)
    try:
        df_mc = pd.read_sql("SELECT * FROM monte_carlo ORDER BY prob_quiebre DESC", engine)

        riesgo_alto  = len(df_mc[df_mc["nivel_riesgo"] == "🔴 Riesgo alto"])
        riesgo_medio = len(df_mc[df_mc["nivel_riesgo"] == "🟡 Riesgo medio"])
        riesgo_bajo  = len(df_mc[df_mc["nivel_riesgo"].isin(["🟢 Riesgo bajo","🟠 Riesgo bajo-medio"])])
        prob_prom    = df_mc["prob_quiebre"].mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(kpi("Riesgo alto",     riesgo_alto,          "danger",  "≥70% prob. quiebre"),  unsafe_allow_html=True)
        with c2: st.markdown(kpi("Riesgo medio",    riesgo_medio,         "warning", "40-70% prob. quiebre"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Riesgo bajo",     riesgo_bajo,          "success", "<15% prob. quiebre"),  unsafe_allow_html=True)
        with c4: st.markdown(kpi("Prob. promedio",  f"{prob_prom:.1f}%",  "info",    "Todos los SKUs"),       unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            df_crit = df_mc[["sku_id","tienda_id","prob_quiebre","stock_actual",
                              "demanda_p95","stock_recomendado","nivel_riesgo"]].head(15).copy()
            df_crit["prob_quiebre"] = df_crit["prob_quiebre"].round(1).astype(str) + "%"
            df_crit.columns = ["SKU","Tienda","Prob. quiebre","Stock actual","Demanda P95","Stock recomendado","Nivel"]
            st.dataframe(df_crit, use_container_width=True, hide_index=True, height=360)

        with col2:
            df_comp = df_mc.nlargest(15, "prob_quiebre")[["sku_id","stock_actual","stock_recomendado"]].copy()
            df_comp["sku_id"] = "SKU " + df_comp["sku_id"].astype(str)
            df_melt = df_comp.melt(id_vars="sku_id", var_name="tipo", value_name="unidades")
            df_melt["tipo"] = df_melt["tipo"].map({"stock_actual":"Stock actual","stock_recomendado":"Stock recomendado P95"})
            fig = px.bar(df_melt, x="sku_id", y="unidades", color="tipo", barmode="group",
                         color_discrete_sequence=[COLORS["danger"], COLORS["primary"]],
                         labels={"sku_id":"SKU","unidades":"Unidades","tipo":""})
            fig.update_layout(**PLOTLY_TEMPLATE, margin=dict(l=0,r=0,t=10,b=0), height=360)
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("ℹ️ Ejecuta modelo_monte_carlo.py para ver la simulación de riesgo.")

st.divider()
st.caption("Go Retail v3.0 · Módulo de Inventario")