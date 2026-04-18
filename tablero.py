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
# Sección Market Basket Analysis
# ─────────────────────────────────────────
st.divider()
st.subheader("🛍️ Market Basket — Productos que se compran juntos")

try:
    df_mb = pd.read_sql("SELECT * FROM market_basket ORDER BY lift DESC", conectar_engine())

    if not df_mb.empty:
        col_mb1, col_mb2 = st.columns(2)

        with col_mb1:
            st.markdown("**🔗 Top reglas de asociación más fuertes**")
            df_top_mb = df_mb[["sku_origen", "sku_destino", "confianza", "lift", "soporte"]].head(15).copy()
            df_top_mb["confianza"] = (df_top_mb["confianza"] * 100).round(1).astype(str) + "%"
            df_top_mb["soporte"]   = (df_top_mb["soporte"]   * 100).round(1).astype(str) + "%"
            df_top_mb["lift"]      = df_top_mb["lift"].round(2)
            df_top_mb.columns = ["Si compra SKU", "También lleva SKU", "Confianza", "Lift", "Soporte"]
            st.dataframe(df_top_mb, use_container_width=True, height=380, hide_index=True)

        with col_mb2:
            st.markdown("**📊 Distribución de Lift por regla**")
            fig_mb = px.histogram(
                df_mb,
                x="lift",
                nbins=20,
                color_discrete_sequence=["#1D9E75"],
                labels={"lift": "Lift", "count": "Número de reglas"}
            )
            fig_mb.add_vline(x=1, line_dash="dash", line_color="red",
                             annotation_text="Lift = 1 (sin relación)")
            fig_mb.add_vline(x=2, line_dash="dash", line_color="orange",
                             annotation_text="Lift = 2 (relación fuerte)")
            fig_mb.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=380
            )
            st.plotly_chart(fig_mb, use_container_width=True)

        # Métricas resumen
        c1, c2, c3 = st.columns(3)
        c1.metric("Total reglas encontradas", len(df_mb))
        c2.metric("Lift promedio",            f"{df_mb['lift'].mean():.2f}")
        c3.metric("Confianza promedio",        f"{(df_mb['confianza'].mean()*100):.1f}%")

except Exception:
    st.info("ℹ️ Ejecuta modelo_market_basket.py para ver los productos que se compran juntos.")

# ─────────────────────────────────────────
# Sección EOQ
# ─────────────────────────────────────────
st.divider()
st.subheader("🛒 EOQ — Cantidad óptima de pedido")

try:
    df_eoq = pd.read_sql("SELECT * FROM eoq_resultados", conectar_engine())

    if not df_eoq.empty:

        # Métricas EOQ
        pedir_ahora  = len(df_eoq[df_eoq["estado_reposicion"] == "🔴 Pedir ahora"])
        pedir_pronto = len(df_eoq[df_eoq["estado_reposicion"] == "🟡 Pedir pronto"])
        stock_ok     = len(df_eoq[df_eoq["estado_reposicion"] == "🟢 Stock OK"])
        costo_total  = df_eoq["costo_total_optimizado"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pedir ahora",  pedir_ahora,  delta=f"{pedir_ahora} urgentes",  delta_color="inverse")
        c2.metric("Pedir pronto", pedir_pronto, delta=f"{pedir_pronto} próximos",  delta_color="inverse")
        c3.metric("Stock OK",     stock_ok,     delta=f"{stock_ok} estables",      delta_color="normal")
        c4.metric("Costo optimizado total", f"${costo_total:,.0f} COP", delta_color="off")

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown("**🔴 SKUs que necesitan pedido inmediato**")
            urgentes = df_eoq[df_eoq["estado_reposicion"] == "🔴 Pedir ahora"][
                ["sku_id", "tienda_id", "eoq", "punto_reorden", "site_qty", "stock_seguridad", "dias_entre_pedidos"]
            ].nlargest(15, "eoq")
            urgentes.columns = ["SKU", "Tienda", "EOQ (und)", "Punto reorden", "Stock actual", "Stock seguridad", "Días entre pedidos"]
            st.dataframe(urgentes, use_container_width=True, height=350, hide_index=True)

        with col_e2:
            st.markdown("**📊 EOQ promedio por categoría**")
            df_cat = df_eoq.groupby("categoria").agg(
                eoq_prom        =("eoq",                   "mean"),
                costo_opt_total =("costo_total_optimizado", "sum"),
                skus            =("sku_id",                 "nunique")
            ).round(1).reset_index().sort_values("eoq_prom", ascending=False)

            fig_eoq = px.bar(
                df_cat,
                x="categoria",
                y="eoq_prom",
                color="eoq_prom",
                color_continuous_scale="Blues",
                labels={"categoria": "Categoría", "eoq_prom": "EOQ promedio (unidades)"}
            )
            fig_eoq.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=350,
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_eoq, use_container_width=True)

except Exception:
    st.info("ℹ️ Ejecuta modelo_eoq.py para ver los indicadores de cantidad óptima de pedido.")

# ─────────────────────────────────────────
# Sección Monte Carlo
# ─────────────────────────────────────────
st.divider()
st.subheader("🎲 Monte Carlo — Simulación de riesgo de quiebre")

try:
    df_mc = pd.read_sql("SELECT * FROM monte_carlo ORDER BY prob_quiebre DESC", conectar_engine())

    if not df_mc.empty:

        riesgo_alto  = len(df_mc[df_mc["nivel_riesgo"] == "🔴 Riesgo alto"])
        riesgo_medio = len(df_mc[df_mc["nivel_riesgo"] == "🟡 Riesgo medio"])
        riesgo_bajo  = len(df_mc[df_mc["nivel_riesgo"].isin(["🟢 Riesgo bajo", "🟠 Riesgo bajo-medio"])])
        prob_prom    = df_mc["prob_quiebre"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Riesgo alto",              riesgo_alto,  delta=f"{riesgo_alto} críticos",      delta_color="inverse")
        c2.metric("Riesgo medio",             riesgo_medio, delta=f"{riesgo_medio} a monitorear", delta_color="inverse")
        c3.metric("Riesgo bajo",              riesgo_bajo,  delta=f"{riesgo_bajo} estables",      delta_color="normal")
        c4.metric("Prob. quiebre promedio",   f"{prob_prom:.1f}%",                                delta_color="off")

        col_mc1, col_mc2 = st.columns(2)

        with col_mc1:
            st.markdown("**🚨 SKUs con mayor riesgo de quiebre**")
            df_criticos = df_mc[["sku_id", "tienda_id", "prob_quiebre", "stock_actual",
                                  "demanda_p95", "stock_recomendado", "dias_cobertura", "nivel_riesgo"]].head(15).copy()
            df_criticos["prob_quiebre"] = df_criticos["prob_quiebre"].round(1).astype(str) + "%"
            df_criticos.columns = ["SKU", "Tienda", "Prob. quiebre", "Stock actual",
                                    "Demanda P95", "Stock recomendado", "Días cobertura", "Nivel riesgo"]
            st.dataframe(df_criticos, use_container_width=True, height=380, hide_index=True)

        with col_mc2:
            st.markdown("**📊 Distribución de probabilidad de quiebre**")
            fig_mc = px.histogram(
                df_mc, x="prob_quiebre", nbins=20,
                color_discrete_sequence=["#E24B4A"],
                labels={"prob_quiebre": "Probabilidad de quiebre (%)", "count": "SKUs"}
            )
            fig_mc.add_vline(x=70, line_dash="dash", line_color="red",   annotation_text="Riesgo alto (70%)")
            fig_mc.add_vline(x=40, line_dash="dash", line_color="orange", annotation_text="Riesgo medio (40%)")
            fig_mc.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
            st.plotly_chart(fig_mc, use_container_width=True)

        st.markdown("**📦 Stock actual vs Stock recomendado P95 — Top 20 SKUs críticos**")
        df_comp = df_mc.nlargest(20, "prob_quiebre")[["sku_id", "stock_actual", "stock_recomendado"]].copy()
        df_comp["sku_id"] = "SKU " + df_comp["sku_id"].astype(str)
        df_melt = df_comp.melt(id_vars="sku_id", var_name="tipo", value_name="unidades")
        df_melt["tipo"] = df_melt["tipo"].map({"stock_actual": "Stock actual", "stock_recomendado": "Stock recomendado P95"})
        fig_comp = px.bar(
            df_melt, x="sku_id", y="unidades", color="tipo", barmode="group",
            color_discrete_sequence=["#E24B4A", "#1D9E75"],
            labels={"sku_id": "SKU", "unidades": "Unidades", "tipo": ""}
        )
        fig_comp.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
        st.plotly_chart(fig_comp, use_container_width=True)

except Exception:
    st.info("ℹ️ Ejecuta modelo_monte_carlo.py para ver la simulación de riesgo de quiebre.")

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.caption("Go Retail · Modelos: Prophet · LightGBM · K-Means · Isolation Forest · EOQ · Monte Carlo · Market Basket · Base de datos: Neon PostgreSQL")
