import warnings
warnings.filterwarnings("ignore")

import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modelos avanzados opcionales
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import STL
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False


# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="SIIMTUR",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

ARCHIVO_EXCEL = "datos_turisticos.xlsx"

# Redes y perfiles
ORCID_URL = "https://orcid.org/0000-0003-3594-0038"
SCOPUS_URL = "https://www.scopus.com/authid/detail.uri?authorId=57952261200"
SCHOLAR_URL = "https://scholar.google.com/citations?user=ZPSfApAAAAAJ&hl=es"
LINKEDIN_URL = "https://www.linkedin.com/in/cesar-omar-sepulveda-moreno-137a1210/"
EMAIL_URL = "mailto:cesar.sepulveda@uabc.edu.mx"

MESES = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Setiembre": 9, "Octubre": 10,
    "Noviembre": 11, "Diciembre": 12
}

ORDEN_MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]


# =========================================================
# ESTILO CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1.1rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 1rem 0 0.8rem 0;
        padding-left: 0.8rem;
        border-left: 4px solid #2563eb;
    }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border: 1px solid #e2e8f0;
        min-height: 110px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
        color: #1e3a8a;
    }
    .warning-card {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f97316;
        color: #9a3412;
    }
    .success-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
        color: #166534;
    }
    .profile-card {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border: 1px solid #e2e8f0;
        margin-top: 0.8rem;
    }
    .profile-title {
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.6rem;
        font-size: 1rem;
    }
    .profile-link {
        margin-bottom: 0.45rem;
        font-size: 0.9rem;
    }
    .profile-link a {
        color: #2563eb !important;
        text-decoration: none;
        font-weight: 500;
    }
    .profile-link a:hover {
        text-decoration: underline;
    }
    .footer {
        text-align: center;
        color: #64748b;
        padding: 1rem 0 2rem 0;
        font-size: 0.8rem;
    }
    .footer a {
        color: #2563eb;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# UTILIDADES
# =========================================================
def format_money(x):
    return "-" if pd.isna(x) else f"${x:,.0f}"

def format_pct(x):
    return "-" if pd.isna(x) else f"{x:.1f}%"

def format_number(x):
    return "-" if pd.isna(x) else f"{x:,.1f}"

def safe_divide(a, b):
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    elif isinstance(b, (int, float)) and b == 0:
        return np.nan
    return a / b

def calcular_metricas_calidad(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}

def normalizar_0_100(series):
    s = series.astype(float).copy()
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    if s.max() == s.min():
        return pd.Series(50.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min()) * 100

def descargar_csv(df, nombre_archivo):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{nombre_archivo}.csv">📥 Descargar CSV</a>'
    return href

def fallback_regresion_lineal(serie, horizon=6):
    train_size = int(len(serie) * 0.8)
    train, test = serie[:train_size], serie[train_size:]

    X_train = np.arange(len(train)).reshape(-1, 1)
    X_test = np.arange(len(train), len(serie)).reshape(-1, 1)
    X_future = np.arange(len(serie), len(serie) + horizon).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, train.values)
    predictions = model.predict(X_test)
    future_pred = model.predict(X_future)
    return train, test, predictions, future_pred


# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def cargar_datos():
    try:
        raw = pd.read_excel(ARCHIVO_EXCEL, sheet_name="Hoja1", header=None)
        encabezados = raw.iloc[1].tolist()
        df = raw.iloc[2:].copy()
        df.columns = encabezados

        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
        )

        for col in ["Estado", "Destino", "Mes"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        cols_numericas = [
            "Año", "Porcentaje de ocupación", "Estancia promedio", "Días reservados",
            "Listados reservados", "Ingresos mensuales", "Tarifa diaria promedio",
            "Ingreso por habitación disponible", "Listados 1 recámara", "Listados 2 recámaras",
            "Listados 3 recámaras", "Listados 4 recámaras", "Listados 5 recámaras", "Listados 6+ recámaras",
            "Percentil 25 ingresos", "Mediana ingresos", "Percentil 75 ingresos", "Percentil 90 ingresos"
        ]

        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Mes_num"] = df["Mes"].map(MESES)
        df["Fecha"] = pd.to_datetime(
            dict(year=df["Año"], month=df["Mes_num"], day=1),
            errors="coerce"
        )

        df = df.dropna(subset=["Destino", "Fecha"]).copy()

        # Corrección de estancia desde Hoja2
        try:
            corr = pd.read_excel(ARCHIVO_EXCEL, sheet_name="Hoja2")
            corr.columns = corr.columns.astype(str).str.strip()
            needed = ["Destino", "Mes", "Año", "Estancia promedio sugerida"]
            if all(c in corr.columns for c in needed):
                corr["Destino"] = corr["Destino"].astype(str).str.strip()
                corr["Mes"] = corr["Mes"].astype(str).str.strip()
                corr["Año"] = pd.to_numeric(corr["Año"], errors="coerce")
                corr["Estancia promedio sugerida"] = pd.to_numeric(corr["Estancia promedio sugerida"], errors="coerce")

                df = df.merge(
                    corr[needed],
                    on=["Destino", "Mes", "Año"],
                    how="left"
                )
                df["Estancia promedio"] = df["Estancia promedio sugerida"].fillna(df["Estancia promedio"])
                df.drop(columns=["Estancia promedio sugerida"], inplace=True)
        except Exception:
            pass

        # Oferta por tipo
        df["Oferta_1_recamara"] = df.get("Listados 1 recámara", 0).fillna(0)
        df["Oferta_2_recamaras"] = df.get("Listados 2 recámaras", 0).fillna(0)
        df["Oferta_3_recamaras"] = df.get("Listados 3 recámaras", 0).fillna(0)
        df["Oferta_4_plus"] = (
            df.get("Listados 4 recámaras", 0).fillna(0)
            + df.get("Listados 5 recámaras", 0).fillna(0)
            + df.get("Listados 6+ recámaras", 0).fillna(0)
        )

        df["Oferta_total"] = df[[
            "Oferta_1_recamara", "Oferta_2_recamaras", "Oferta_3_recamaras", "Oferta_4_plus"
        ]].sum(axis=1)

        # Indicadores derivados
        df["RevPAR"] = df["Ingreso por habitación disponible"]
        df["TRevPAR"] = safe_divide(df["Ingresos mensuales"], df["Oferta_total"])
        df["Conversion_Rate"] = safe_divide(df["Listados reservados"], df["Oferta_total"]) * 100

        df["Participacion_Oferta"] = safe_divide(
            df["Oferta_total"],
            df.groupby("Fecha")["Oferta_total"].transform("sum")
        ) * 100

        df["Participacion_Demanda"] = safe_divide(
            df["Días reservados"],
            df.groupby("Fecha")["Días reservados"].transform("sum")
        ) * 100

        df["Aporte_Mercado"] = df["Porcentaje de ocupación"] * safe_divide(
            df["Oferta_total"],
            df.groupby("Fecha")["Oferta_total"].transform("sum")
        )

        df["Presion_Mercado"] = safe_divide(df["Días reservados"], df["Oferta_total"])

        df = df.sort_values(["Destino", "Fecha"]).copy()

        df["YoY_Ocupacion"] = df.groupby("Destino")["Porcentaje de ocupación"].pct_change(12) * 100
        df["YoY_ADR"] = df.groupby("Destino")["Tarifa diaria promedio"].pct_change(12) * 100
        df["YoY_RevPAR"] = df.groupby("Destino")["RevPAR"].pct_change(12) * 100

        df["Volatilidad"] = df.groupby("Destino")["Porcentaje de ocupación"].transform(
            lambda x: x.rolling(12, min_periods=6).std()
        )

        occ_chg = df.groupby("Destino")["Porcentaje de ocupación"].pct_change()
        adr_chg = df.groupby("Destino")["Tarifa diaria promedio"].pct_change()
        df["Elasticidad"] = safe_divide(occ_chg, adr_chg)

        df["Mes_Cat"] = pd.Categorical(df["Mes"], categories=ORDEN_MESES, ordered=True)
        df["Trimestre"] = df["Fecha"].dt.quarter
        df["Anio"] = df["Fecha"].dt.year
        df["Mes_Sin"] = np.sin(2 * np.pi * df["Mes_num"] / 12)
        df["Mes_Cos"] = np.cos(2 * np.pi * df["Mes_num"] / 12)

        return df

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None


df = cargar_datos()
if df is None or df.empty:
    st.error("No se pudieron cargar los datos. Verifica que el archivo Excel existe en la carpeta.")
    st.stop()


# =========================================================
# SIDEBAR CON FILTROS ÚTILES
# =========================================================
with st.sidebar:
    st.markdown("## 🏨 SIIMTUR")
    st.caption("Inteligencia de Mercados Turísticos")
    st.markdown("---")

    estados = sorted(df["Estado"].dropna().unique().tolist()) if "Estado" in df.columns else []
    estados_sel = st.multiselect(
        "🗺️ Estado",
        estados,
        default=estados
    ) if estados else []

    df_temp = df.copy()
    if estados_sel:
        df_temp = df_temp[df_temp["Estado"].isin(estados_sel)]

    destinos = sorted(df_temp["Destino"].dropna().unique().tolist())
    destinos_sel = st.multiselect(
        "📍 Destinos",
        destinos,
        default=destinos[:4] if len(destinos) >= 4 else destinos
    )

    fechas = df["Fecha"].dt.date
    rango = st.date_input(
        "📅 Período",
        (fechas.min(), fechas.max())
    )

    trimestres = sorted(df["Trimestre"].dropna().unique().tolist()) if "Trimestre" in df.columns else [1, 2, 3, 4]
    trimestre_sel = st.multiselect(
        "🗓️ Trimestre",
        trimestres,
        default=trimestres
    )

    meses_disp = [m for m in ORDEN_MESES if m in df["Mes"].dropna().unique().tolist()]
    meses_sel = st.multiselect(
        "📆 Mes",
        meses_disp,
        default=meses_disp
    )

    ocup_min = float(df["Porcentaje de ocupación"].min())
    ocup_max = float(df["Porcentaje de ocupación"].max())
    rango_ocup = st.slider(
        "🏨 Rango de ocupación (%)",
        min_value=float(np.floor(ocup_min)),
        max_value=float(np.ceil(ocup_max)),
        value=(float(np.floor(ocup_min)), float(np.ceil(ocup_max)))
    )

    adr_min = float(df["Tarifa diaria promedio"].min())
    adr_max = float(df["Tarifa diaria promedio"].max())
    rango_adr = st.slider(
        "💰 Rango de ADR",
        min_value=float(np.floor(adr_min)),
        max_value=float(np.ceil(adr_max)),
        value=(float(np.floor(adr_min)), float(np.ceil(adr_max)))
    )

    metrica_principal = st.selectbox(
        "🎯 Métrica principal del dashboard",
        [
            "Porcentaje de ocupación",
            "Tarifa diaria promedio",
            "RevPAR",
            "Conversion_Rate",
            "Aporte_Mercado",
            "YoY_Ocupacion",
            "Presion_Mercado"
        ],
        index=0
    )

    top_n = st.slider("🔢 Top N destinos a mostrar", 3, 10, 5)

    # Solo para gráfico de estructura de oferta
    tipo_prop = st.multiselect(
        "🏠 Composición de oferta (solo gráfico de estructura)",
        ["1 recámara", "2 recámaras", "3 recámaras", "4+ recámaras"],
        default=["1 recámara", "2 recámaras", "3 recámaras", "4+ recámaras"]
    )

    st.markdown("---")
    st.caption(f"📊 {len(df):,} registros | {df['Destino'].nunique()} destinos")

    if not STATSMODELS_OK:
        st.info("📦 Para modelos avanzados (ETS/SARIMA): `pip install statsmodels`")

    st.markdown("---")
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-title">👨‍🔬 Dr. César Omar Sepúlveda Moreno</div>
            <div class="profile-link">📖 <a href="{ORCID_URL}" target="_blank">ORCID</a></div>
            <div class="profile-link">🎓 <a href="{SCHOLAR_URL}" target="_blank">Google Scholar</a></div>
            <div class="profile-link">📚 <a href="{SCOPUS_URL}" target="_blank">Scopus</a></div>
            <div class="profile-link">💼 <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a></div>
            <div class="profile-link">🐦 <a href="{TWITTER_URL}" target="_blank">Twitter/X</a></div>
            <div class="profile-link">✉️ <a href="{EMAIL_URL}">Email</a></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# APLICAR FILTROS REALES
# =========================================================
df_filt = df.copy()

if estados_sel:
    df_filt = df_filt[df_filt["Estado"].isin(estados_sel)]

if destinos_sel:
    df_filt = df_filt[df_filt["Destino"].isin(destinos_sel)]

if isinstance(rango, tuple) and len(rango) == 2:
    df_filt = df_filt[
        (df_filt["Fecha"] >= pd.to_datetime(rango[0])) &
        (df_filt["Fecha"] <= pd.to_datetime(rango[1]))
    ]

if trimestre_sel:
    df_filt = df_filt[df_filt["Trimestre"].isin(trimestre_sel)]

if meses_sel:
    df_filt = df_filt[df_filt["Mes"].isin(meses_sel)]

df_filt = df_filt[
    (df_filt["Porcentaje de ocupación"] >= rango_ocup[0]) &
    (df_filt["Porcentaje de ocupación"] <= rango_ocup[1])
]

df_filt = df_filt[
    (df_filt["Tarifa diaria promedio"] >= rango_adr[0]) &
    (df_filt["Tarifa diaria promedio"] <= rango_adr[1])
]

if df_filt.empty:
    st.warning("⚠️ No hay datos con los filtros seleccionados. Ajusta los criterios.")
    st.stop()


# =========================================================
# CONFIGURACIÓN DINÁMICA
# =========================================================
mapa_titulos = {
    "Porcentaje de ocupación": "Ocupación (%)",
    "Tarifa diaria promedio": "ADR (USD)",
    "RevPAR": "RevPAR (USD)",
    "Conversion_Rate": "Tasa de conversión (%)",
    "Aporte_Mercado": "Aporte al mercado",
    "YoY_Ocupacion": "Crecimiento YoY (%)",
    "Presion_Mercado": "Presión de mercado"
}
titulo_metrica = mapa_titulos.get(metrica_principal, metrica_principal)


# =========================================================
# ENCABEZADO
# =========================================================
st.markdown('<div class="main-title">SIIMTUR PROFESSIONAL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Plataforma integral de visualización, análisis y apoyo estratégico para mercados turísticos</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="insight-card">
        <strong>🔗 Enlaces académicos del autor:</strong>
        <a href="{ORCID_URL}" target="_blank">ORCID</a> ·
        <a href="{SCHOLAR_URL}" target="_blank">Google Scholar</a> ·
        <a href="{SCOPUS_URL}" target="_blank">Scopus</a> ·
        <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a> ·
        <a href="{TWITTER_URL}" target="_blank">Twitter/X</a>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PESTAÑAS
# =========================================================
tab_res, tab_desc, tab_intel, tab_corr, tab_exp, tab_pred, tab_autor = st.tabs([
    "📊 Resumen Ejecutivo",
    "📈 Análisis Descriptivo",
    "🧠 Inteligencia de Mercado",
    "🔗 Correlaciones",
    "🔬 Análisis Explicativo",
    "🔮 Predicciones",
    "👤 Autor y Metodología"
])


# =========================================================
# TAB 1: RESUMEN EJECUTIVO
# =========================================================
with tab_res:
    st.markdown('<div class="section-title">Indicadores Clave del Período</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🏨 OCUPACIÓN</div>
                <div class='metric-value'>{format_pct(df_filt["Porcentaje de ocupación"].mean())}</div>
                <div class='metric-sub'>Promedio del período</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>💰 ADR</div>
                <div class='metric-value'>{format_money(df_filt["Tarifa diaria promedio"].mean())}</div>
                <div class='metric-sub'>Tarifa diaria promedio</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>📊 REVPAR</div>
                <div class='metric-value'>{format_money(df_filt["RevPAR"].mean())}</div>
                <div class='metric-sub'>Ingreso por habitación disponible</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        valor_metrica = df_filt[metrica_principal].mean()
        if metrica_principal in ["Porcentaje de ocupación", "Conversion_Rate", "YoY_Ocupacion"]:
            valor_formateado = format_pct(valor_metrica)
        elif metrica_principal in ["Tarifa diaria promedio", "RevPAR"]:
            valor_formateado = format_money(valor_metrica)
        else:
            valor_formateado = format_number(valor_metrica)

        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🎯 MÉTRICA PRINCIPAL</div>
                <div class='metric-value'>{valor_formateado}</div>
                <div class='metric-sub'>{titulo_metrica}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Evolución del Mercado</div>', unsafe_allow_html=True)

    evol = df_filt.groupby("Fecha").agg({
        "Porcentaje de ocupación": "mean",
        "Tarifa diaria promedio": "mean",
        "RevPAR": "mean",
        metrica_principal: "mean"
    }).reset_index()

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Ocupación", "ADR", "RevPAR", titulo_metrica))
    fig.add_trace(go.Scatter(x=evol["Fecha"], y=evol["Porcentaje de ocupación"], mode="lines+markers", line=dict(color="#2563eb")), row=1, col=1)
    fig.add_trace(go.Scatter(x=evol["Fecha"], y=evol["Tarifa diaria promedio"], mode="lines+markers", line=dict(color="#10b981")), row=1, col=2)
    fig.add_trace(go.Scatter(x=evol["Fecha"], y=evol["RevPAR"], mode="lines+markers", line=dict(color="#f59e0b")), row=2, col=1)
    fig.add_trace(go.Scatter(x=evol["Fecha"], y=evol[metrica_principal], mode="lines+markers", line=dict(color="#ef4444")), row=2, col=2)
    fig.update_layout(height=560, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    top_ocup = df_filt.groupby("Destino")["Porcentaje de ocupación"].mean().idxmax()
    top_aporte = df_filt.groupby("Destino")["Aporte_Mercado"].mean().idxmax()
    mas_volatil = df_filt.groupby("Destino")["Volatilidad"].mean().idxmax()
    mejor_crecimiento = df_filt.groupby("Destino")["YoY_Ocupacion"].mean().idxmax()

    st.markdown(
        f"""
        <div class='insight-card'>
            <strong>📌 Insight Ejecutivo</strong><br>
            • <strong>{top_ocup}</strong> lidera en ocupación promedio del período.<br>
            • <strong>{top_aporte}</strong> es el principal aportador al mercado regional.<br>
            • <strong>{mas_volatil}</strong> presenta la mayor volatilidad observada.<br>
            • <strong>{mejor_crecimiento}</strong> muestra el mejor crecimiento interanual promedio.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TAB 2: DESCRIPTIVO
# =========================================================
with tab_desc:
    st.markdown('<div class="section-title">Indicador dinámico por destino</div>', unsafe_allow_html=True)

    indicador_descriptivo = st.selectbox(
        "Selecciona el indicador para visualizar",
        [
            "Porcentaje de ocupación",
            "Tarifa diaria promedio",
            "RevPAR",
            "Conversion_Rate",
            "Aporte_Mercado",
            "Presion_Mercado"
        ]
    )

    fig_ind = px.line(
        df_filt,
        x="Fecha",
        y=indicador_descriptivo,
        color="Destino",
        title=f"Evolución de {indicador_descriptivo}",
        template="plotly_white"
    )
    fig_ind.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig_ind, width="stretch")

    st.markdown('<div class="section-title">Series temporales complementarias</div>', unsafe_allow_html=True)

    fig_desc = make_subplots(rows=2, cols=2, subplot_titles=("Ocupación (%)", "ADR (USD)", "RevPAR (USD)", "Tasa de Conversión (%)"))
    colores = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

    for i, d in enumerate(destinos_sel[:6]):
        ddf = df_filt[df_filt["Destino"] == d]
        color = colores[i % len(colores)]
        fig_desc.add_trace(go.Scatter(x=ddf["Fecha"], y=ddf["Porcentaje de ocupación"], mode="lines", name=d, line=dict(color=color)), row=1, col=1)
        fig_desc.add_trace(go.Scatter(x=ddf["Fecha"], y=ddf["Tarifa diaria promedio"], mode="lines", name=d, line=dict(color=color), showlegend=False), row=1, col=2)
        fig_desc.add_trace(go.Scatter(x=ddf["Fecha"], y=ddf["RevPAR"], mode="lines", name=d, line=dict(color=color), showlegend=False), row=2, col=1)
        fig_desc.add_trace(go.Scatter(x=ddf["Fecha"], y=ddf["Conversion_Rate"], mode="lines", name=d, line=dict(color=color), showlegend=False), row=2, col=2)

    fig_desc.update_layout(height=620, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_desc, width="stretch")

    st.markdown('<div class="section-title">Patrón Estacional del Mercado</div>', unsafe_allow_html=True)
    estacional = df_filt.groupby("Mes_Cat")["Porcentaje de ocupación"].agg(["mean", "std"]).reset_index()
    estacional.columns = ["Mes", "Ocupación Promedio", "Desviación"]

    fig_est = px.bar(
        estacional,
        x="Mes",
        y="Ocupación Promedio",
        error_y="Desviación",
        title="Ocupación promedio por mes (con variabilidad)",
        template="plotly_white",
        color="Ocupación Promedio",
        color_continuous_scale="Blues"
    )
    fig_est.update_layout(height=450)
    st.plotly_chart(fig_est, width="stretch")

    st.markdown('<div class="section-title">Composición de oferta por tipo de propiedad</div>', unsafe_allow_html=True)
    mapa_oferta = {
        "1 recámara": "Oferta_1_recamara",
        "2 recámaras": "Oferta_2_recamaras",
        "3 recámaras": "Oferta_3_recamaras",
        "4+ recámaras": "Oferta_4_plus"
    }
    tipos_activos = [t for t in tipo_prop if t in mapa_oferta]

    if tipos_activos:
        oferta_tipo = pd.DataFrame({
            "Tipo": tipos_activos,
            "Oferta": [df_filt[mapa_oferta[t]].sum() for t in tipos_activos]
        })

        fig_pie = px.pie(
            oferta_tipo,
            values="Oferta",
            names="Tipo",
            hole=0.45,
            title="Estructura de la oferta seleccionada",
            template="plotly_white"
        )
        fig_pie.update_layout(height=420)
        st.plotly_chart(fig_pie, width="stretch")


# =========================================================
# TAB 3: INTELIGENCIA DE MERCADO
# =========================================================
with tab_intel:
    st.markdown('<div class="section-title">Índice SIIMTUR - Ranking Integral</div>', unsafe_allow_html=True)

    ranking = df_filt.groupby("Destino").agg({
        "Porcentaje de ocupación": "mean",
        "Tarifa diaria promedio": "mean",
        "RevPAR": "mean",
        "Aporte_Mercado": "mean",
        "YoY_Ocupacion": "mean",
        "Volatilidad": "mean",
        "Conversion_Rate": "mean",
        "Presion_Mercado": "mean"
    }).reset_index()

    ranking["Score_Ocup"] = normalizar_0_100(ranking["Porcentaje de ocupación"])
    ranking["Score_RevPAR"] = normalizar_0_100(ranking["RevPAR"])
    ranking["Score_Aporte"] = normalizar_0_100(ranking["Aporte_Mercado"])
    ranking["Score_Crecimiento"] = normalizar_0_100(ranking["YoY_Ocupacion"].fillna(0))
    ranking["Score_Conversion"] = normalizar_0_100(ranking["Conversion_Rate"].fillna(0))
    ranking["Score_Riesgo"] = 100 - normalizar_0_100(ranking["Volatilidad"].fillna(ranking["Volatilidad"].median()))

    ranking["Indice_SIIMTUR"] = (
        ranking["Score_Ocup"] * 0.25 +
        ranking["Score_RevPAR"] * 0.20 +
        ranking["Score_Crecimiento"] * 0.20 +
        ranking["Score_Aporte"] * 0.15 +
        ranking["Score_Conversion"] * 0.10 +
        ranking["Score_Riesgo"] * 0.10
    )

    ranking = ranking.sort_values("Indice_SIIMTUR", ascending=False)

    fig_rank = px.bar(
        ranking.head(top_n),
        x="Indice_SIIMTUR",
        y="Destino",
        orientation="h",
        color="Indice_SIIMTUR",
        color_continuous_scale="Viridis",
        title=f"Top {top_n} destinos según Índice SIIMTUR",
        template="plotly_white",
        labels={"Indice_SIIMTUR": "Índice de Desempeño (0-100)", "Destino": ""}
    )
    fig_rank.update_layout(height=550)
    st.plotly_chart(fig_rank, width="stretch")

    st.markdown('<div class="section-title">Ranking dinámico por métrica elegida</div>', unsafe_allow_html=True)

    ranking_base = df_filt.groupby("Destino").agg({
        "Porcentaje de ocupación": "mean",
        "Tarifa diaria promedio": "mean",
        "RevPAR": "mean",
        "Aporte_Mercado": "mean",
        "YoY_Ocupacion": "mean",
        "Volatilidad": "mean",
        "Conversion_Rate": "mean",
        "Presion_Mercado": "mean"
    }).reset_index()

    ranking_metrica = ranking_base.sort_values(metrica_principal, ascending=False).head(top_n)

    fig_rank_metric = px.bar(
        ranking_metrica,
        x=metrica_principal,
        y="Destino",
        orientation="h",
        color=metrica_principal,
        color_continuous_scale="Viridis",
        title=f"Top {top_n} destinos según {titulo_metrica}",
        template="plotly_white"
    )
    fig_rank_metric.update_layout(height=500)
    st.plotly_chart(fig_rank_metric, width="stretch")

    st.markdown('<div class="section-title">Mapa Competitivo: Ocupación vs ADR</div>', unsafe_allow_html=True)
    fig_bubble = px.scatter(
        ranking,
        x="Porcentaje de ocupación",
        y="Tarifa diaria promedio",
        size="RevPAR",
        color="Indice_SIIMTUR",
        text="Destino",
        title="Posicionamiento competitivo de destinos",
        template="plotly_white",
        labels={"Porcentaje de ocupación": "Ocupación (%)", "Tarifa diaria promedio": "ADR (USD)"}
    )
    fig_bubble.update_traces(textposition="top center")
    fig_bubble.add_hline(y=ranking["Tarifa diaria promedio"].median(), line_dash="dash", line_color="gray", opacity=0.5)
    fig_bubble.add_vline(x=ranking["Porcentaje de ocupación"].median(), line_dash="dash", line_color="gray", opacity=0.5)
    fig_bubble.update_layout(height=550)
    st.plotly_chart(fig_bubble, width="stretch")

    st.markdown('<div class="section-title">Alertas Estratégicas</div>', unsafe_allow_html=True)

    alertas_contador = 0
    for _, row in ranking.iterrows():
        alertas = []
        if row["Porcentaje de ocupación"] < 35:
            alertas.append("⚠️ baja ocupación (<35%)")
        if pd.notna(row["YoY_Ocupacion"]) and row["YoY_Ocupacion"] < -5:
            alertas.append("📉 contracción anual (>5%)")
        if pd.notna(row["Volatilidad"]) and row["Volatilidad"] > 15:
            alertas.append("🎢 alta volatilidad estacional")
        if pd.notna(row["Conversion_Rate"]) and row["Conversion_Rate"] < 10:
            alertas.append("🔄 baja tasa de conversión")

        if alertas:
            alertas_contador += 1
            st.markdown(
                f"<div class='warning-card'><strong>{row['Destino']}</strong>: " + " | ".join(alertas) + "</div>",
                unsafe_allow_html=True
            )

    if alertas_contador == 0:
        st.markdown(
            "<div class='success-card'>✅ No se detectaron alertas críticas en los destinos seleccionados.</div>",
            unsafe_allow_html=True
        )


# =========================================================
# TAB 4: CORRELACIONES
# =========================================================
with tab_corr:
    st.markdown('<div class="section-title">Matriz de Correlación entre Variables</div>', unsafe_allow_html=True)

    vars_corr = st.multiselect(
        "Selecciona las variables a analizar:",
        [
            "Porcentaje de ocupación", "Tarifa diaria promedio", "RevPAR",
            "Estancia promedio", "Días reservados", "Oferta_total",
            "Conversion_Rate", "Presion_Mercado", "Participacion_Oferta"
        ],
        default=["Porcentaje de ocupación", "Tarifa diaria promedio", "RevPAR", "Estancia promedio"]
    )

    if len(vars_corr) >= 2:
        corr = df_filt[vars_corr].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            template="plotly_white",
            title="Matriz de Correlación"
        )
        fig_corr.update_layout(height=550)
        st.plotly_chart(fig_corr, width="stretch")

        st.markdown('<div class="section-title">Relación Bivariada Interactiva</div>', unsafe_allow_html=True)
        col_x, col_y = st.columns(2)
        with col_x:
            x_var = st.selectbox("Variable X", vars_corr, index=0)
        with col_y:
            y_var = st.selectbox("Variable Y", vars_corr, index=1 if len(vars_corr) > 1 else 0)

        fig_sc = px.scatter(
            df_filt,
            x=x_var,
            y=y_var,
            color="Destino",
            size="RevPAR",
            hover_data=["Fecha", "Destino"],
            template="plotly_white",
            title=f"Relación entre {x_var} y {y_var}"
        )
        fig_sc.update_layout(height=500)
        st.plotly_chart(fig_sc, width="stretch")


# =========================================================
# TAB 5: EXPLICATIVO
# =========================================================
with tab_exp:
    st.markdown('<div class="section-title">Modelo Explicativo por Destino</div>', unsafe_allow_html=True)

    destino_exp = st.selectbox("Selecciona un destino para análisis profundo:", destinos_sel)
    objetivo = st.selectbox(
        "Variable objetivo a explicar:",
        ["Porcentaje de ocupación", "RevPAR", "Tarifa diaria promedio", "Conversion_Rate"]
    )

    dfe = df_filt[df_filt["Destino"] == destino_exp].copy().sort_values("Fecha")

    if len(dfe) >= 8:
        vars_exp = ["Estancia promedio", "Oferta_total", "Presion_Mercado", "Participacion_Demanda", "Mes_Sin", "Mes_Cos"]
        vars_exp = [v for v in vars_exp if v in dfe.columns and v != objetivo and dfe[v].notna().sum() > 5]

        if len(vars_exp) >= 3:
            X = dfe[vars_exp].ffill().dropna()
            y = dfe.loc[X.index, objetivo].dropna()
            X = X.loc[y.index]

            if len(X) > 5:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                metricas = calcular_metricas_calidad(y, y_pred)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² (Poder Explicativo)", f"{metricas['R²']:.3f}")
                with col2:
                    st.metric("MAE (Error Absoluto)", f"{metricas['MAE']:.2f}")
                with col3:
                    st.metric("RMSE", f"{metricas['RMSE']:.2f}")

                coef_df = pd.DataFrame({
                    "Variable": vars_exp,
                    "Coeficiente": model.coef_,
                    "Impacto": np.where(model.coef_ >= 0, "Positivo (+)", "Negativo (-)")
                }).sort_values("Coeficiente", ascending=False)

                fig_coef = px.bar(
                    coef_df,
                    x="Coeficiente",
                    y="Variable",
                    orientation="h",
                    color="Impacto",
                    color_discrete_map={"Positivo (+)": "#10b981", "Negativo (-)": "#ef4444"},
                    title=f"Impacto de variables en {objetivo} - {destino_exp}",
                    template="plotly_white"
                )
                fig_coef.update_layout(height=450)
                st.plotly_chart(fig_coef, width="stretch")

                mejor_var = coef_df.iloc[0]["Variable"]
                mejor_impacto = coef_df.iloc[0]["Impacto"]

                st.markdown(
                    f"""
                    <div class='insight-card'>
                        <strong>📊 Interpretación del modelo para {destino_exp}:</strong><br>
                        • El modelo explica <strong>{metricas['R²']*100:.1f}%</strong> de la variabilidad en {objetivo}.<br>
                        • La variable más influyente es <strong>{mejor_var}</strong> con impacto {mejor_impacto}.<br>
                        • Error promedio del modelo: <strong>{metricas['MAE']:.2f}</strong> unidades.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown('<div class="section-title">Segmentación de Destinos (Clustering)</div>', unsafe_allow_html=True)

    cluster_vars = ["Porcentaje de ocupación", "Tarifa diaria promedio", "RevPAR", "Estancia promedio", "Conversion_Rate"]
    cluster_vars_exist = [v for v in cluster_vars if v in df_filt.columns]
    cluster_base = df_filt.groupby("Destino")[cluster_vars_exist].mean().dropna()

    if len(cluster_base) >= 3:
        scaler = StandardScaler()
        Xc = scaler.fit_transform(cluster_base)
        n_clusters = min(4, len(cluster_base))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_base["Cluster"] = km.fit_predict(Xc).astype(str)

        fig_cluster = px.scatter(
            cluster_base.reset_index(),
            x="Porcentaje de ocupación",
            y="Tarifa diaria promedio",
            size="RevPAR",
            color="Cluster",
            text="Destino",
            template="plotly_white",
            title="Segmentación de Destinos por Desempeño",
            labels={"Porcentaje de ocupación": "Ocupación (%)", "Tarifa diaria promedio": "ADR (USD)"}
        )
        fig_cluster.update_traces(textposition="top center")
        fig_cluster.update_layout(height=500)
        st.plotly_chart(fig_cluster, width="stretch")


# =========================================================
# TAB 6: PREDICCIONES
# =========================================================
with tab_pred:
    st.markdown('<div class="section-title">Predicción de Ocupación</div>', unsafe_allow_html=True)

    destino_pred = st.selectbox("Selecciona un destino para forecast:", destinos_sel)
    modelo_sel = st.selectbox(
        "Selecciona el modelo predictivo:",
        ["Regresión Lineal"] + (["ETS (Holt-Winters)", "SARIMA"] if STATSMODELS_OK else [])
    )

    serie = (
        df_filt[df_filt["Destino"] == destino_pred]
        .sort_values("Fecha")
        .set_index("Fecha")["Porcentaje de ocupación"]
        .dropna()
    )

    if len(serie) >= 12:
        predictions = None
        future_pred = None
        train = None
        test = None

        if modelo_sel == "Regresión Lineal":
            train, test, predictions, future_pred = fallback_regresion_lineal(serie, 6)

        elif modelo_sel == "ETS (Holt-Winters)" and STATSMODELS_OK:
            try:
                train_size = int(len(serie) * 0.8)
                train, test = serie[:train_size], serie[train_size:]
                ets = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
                fit = ets.fit()
                predictions = fit.forecast(len(test))
                fit_full = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=12).fit()
                future_pred = fit_full.forecast(6)
            except Exception as e:
                st.warning(f"ETS no pudo ajustarse: {str(e)[:100]}. Usando regresión lineal.")
                train, test, predictions, future_pred = fallback_regresion_lineal(serie, 6)

        elif modelo_sel == "SARIMA" and STATSMODELS_OK:
            try:
                train_size = int(len(serie) * 0.8)
                train, test = serie[:train_size], serie[train_size:]
                sar = SARIMAX(
                    train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit = sar.fit(disp=False)
                predictions = fit.forecast(len(test))

                sar_full = SARIMAX(
                    serie,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit_full = sar_full.fit(disp=False)
                future_pred = fit_full.forecast(6)
            except Exception as e:
                st.warning(f"SARIMA no pudo ajustarse: {str(e)[:100]}. Usando regresión lineal.")
                train, test, predictions, future_pred = fallback_regresion_lineal(serie, 6)

        if predictions is not None and len(test) > 0:
            metricas = calcular_metricas_calidad(test.values, np.array(predictions))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE (Error Absoluto)", f"{metricas['MAE']:.2f}%")
            with col2:
                st.metric("RMSE", f"{metricas['RMSE']:.2f}%")
            with col3:
                st.metric("R² (Precisión)", f"{metricas['R²']:.3f}")

            future_dates = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1), periods=6, freq="MS")

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=train.index, y=train, mode="lines+markers", name="Entrenamiento", line=dict(color="#2563eb")))
            fig_pred.add_trace(go.Scatter(x=test.index, y=test, mode="lines+markers", name="Real (Validación)", line=dict(color="#10b981")))
            fig_pred.add_trace(go.Scatter(x=test.index, y=predictions, mode="lines+markers", name="Predicción", line=dict(dash="dash", color="#f59e0b")))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines+markers", name="Forecast 6 meses", line=dict(dash="dot", color="#ef4444")))
            fig_pred.update_layout(
                title=f"Predicción de Ocupación - {destino_pred} ({modelo_sel})",
                xaxis_title="Fecha",
                yaxis_title="Ocupación (%)",
                height=520,
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_pred, width="stretch")

            pred_df = pd.DataFrame({
                "Período": [f"Mes +{i+1}" for i in range(6)],
                "Fecha": future_dates.strftime("%B %Y"),
                "Pronóstico (%)": np.round(np.array(future_pred), 1),
                "Límite Inferior (95%)": np.round(np.array(future_pred) - 5, 1),
                "Límite Superior (95%)": np.round(np.array(future_pred) + 5, 1)
            })

            st.markdown("#### 📋 Predicciones a 6 meses")
            st.dataframe(pred_df, width="stretch", hide_index=True)

            tendencia = "creciente" if future_pred[-1] > future_pred[0] else "decreciente"
            cambio = abs(future_pred[-1] - future_pred[0])

            st.markdown(
                f"""
                <div class='success-card'>
                    <strong>🔮 Interpretación del forecast:</strong><br>
                    • El modelo {modelo_sel} proyecta una tendencia <strong>{tendencia}</strong> para {destino_pred}.<br>
                    • Se espera una variación de <strong>{cambio:.1f} puntos porcentuales</strong> en los próximos 6 meses.<br>
                    • La ocupación proyectada al final del período es de <strong>{future_pred[-1]:.1f}%</strong>.
                </div>
                """,
                unsafe_allow_html=True
            )

            if STATSMODELS_OK and len(serie) >= 24:
                try:
                    st.markdown('<div class="section-title">Descomposición Estacional (STL)</div>', unsafe_allow_html=True)
                    stl = STL(serie, period=12)
                    res = stl.fit()

                    fig_stl = make_subplots(rows=3, cols=1, subplot_titles=("Tendencia", "Estacionalidad", "Residuo"))
                    fig_stl.add_trace(go.Scatter(x=res.trend.index, y=res.trend, mode="lines", line=dict(color="#2563eb")), row=1, col=1)
                    fig_stl.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, mode="lines", line=dict(color="#10b981")), row=2, col=1)
                    fig_stl.add_trace(go.Scatter(x=res.resid.index, y=res.resid, mode="lines", line=dict(color="#f59e0b")), row=3, col=1)
                    fig_stl.update_layout(height=650, template="plotly_white", showlegend=False)
                    st.plotly_chart(fig_stl, width="stretch")
                except Exception:
                    pass
    else:
        st.warning(f"Se requieren al menos 12 meses de datos para {destino_pred}. Actualmente: {len(serie)} meses.")


# =========================================================
# TAB 7: AUTOR Y METODOLOGÍA
# =========================================================
with tab_autor:
    st.markdown('<div class="section-title">Perfil Académico del Investigador</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-title">👨‍🔬 Dr. César Omar Sepúlveda Moreno</div>
            <p>Profesor Investigador especializado en análisis de datos, competitividad e inteligencia de mercados turísticos. Desarrollador de SIIMTUR.</p>
            <div class="profile-link">📖 <a href="{ORCID_URL}" target="_blank">ORCID: 0000-0003-3594-0038</a></div>
            <div class="profile-link">🎓 <a href="{SCHOLAR_URL}" target="_blank">Google Scholar</a></div>
            <div class="profile-link">📚 <a href="{SCOPUS_URL}" target="_blank">Scopus Author ID: 57952261200</a></div>
            <div class="profile-link">💼 <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a></div>
            <div class="profile-link">✉️ <a href="{EMAIL_URL}">cesar.sepulveda@uabc.edu.mx</a></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Metodología y Modelos Utilizados</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📊 ANÁLISIS DESCRIPTIVO</div>
            <div class="metric-sub">• Series temporales<br>• Estacionalidad<br>• Estructura de oferta<br>• KPIs de mercado</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-label">🔗 ANÁLISIS CORRELACIONAL</div>
            <div class="metric-sub">• Matriz de Pearson<br>• Scatter plots interactivos<br>• Identificación de relaciones</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🔬 ANÁLISIS EXPLICATIVO</div>
            <div class="metric-sub">• Regresión Lineal Múltiple<br>• Coeficientes e impactos<br>• Clustering (K-Means)<br>• R², MAE, RMSE</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-label">🔮 MODELOS PREDICTIVOS</div>
            <div class="metric-sub">• Regresión Lineal<br>• ETS (Holt-Winters)<br>• SARIMA<br>• STL</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Indicadores Construidos (SIIMTUR)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
        <strong>📐 Indicadores Propietarios:</strong><br>
        • <strong>Índice SIIMTUR</strong>: ranking integral que combina ocupación, RevPAR, crecimiento, aporte de mercado, conversión y riesgo.<br>
        • <strong>Aporte de Mercado</strong>: ocupación × participación de oferta.<br>
        • <strong>Presión de Mercado</strong>: días reservados / oferta total.<br>
        • <strong>Elasticidad Precio-Demanda</strong>: cambio en ocupación vs cambio en ADR.<br>
        • <strong>Volatilidad</strong>: desviación estándar móvil de la ocupación.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    f"""
    <div class="footer">
        <div><strong>SIIMTUR PROFESSIONAL</strong> · Sistema Integral de Inteligencia de Mercados Turísticos</div>
        <div style="margin-top: 8px;">
            <a href="{ORCID_URL}" target="_blank">ORCID</a> ·
            <a href="{SCHOLAR_URL}" target="_blank">Google Scholar</a> ·
            <a href="{SCOPUS_URL}" target="_blank">Scopus</a> ·
            <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a> ·
            <a href="{TWITTER_URL}" target="_blank">Twitter/X</a>
        </div>
        <div style="margin-top: 8px; font-size: 0.7rem;">
            Dr. César Omar Sepúlveda Moreno · Profesor Investigador · Análisis de datos, competitividad e inteligencia de mercados turísticos
        </div>
        <div style="margin-top: 8px; font-size: 0.7rem;">
            Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
