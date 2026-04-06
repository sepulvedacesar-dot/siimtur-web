import streamlit as st
import streamlit.components.v1 as components

GA_ID = "G-M2LFB8Q2RL"

components.html(f"""
<!DOCTYPE html>
<html>
<head>
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
gtag('js', new Date());
gtag('config', '{GA_ID}', {{
  'page_path': window.location.pathname
}});
</script>
</head>
<body></body>
</html>
""", height=0)

import warnings
warnings.filterwarnings("ignore")

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

# =========================================================
# MODELOS AVANZADOS OPCIONALES
# =========================================================
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
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
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
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

def weighted_mean(values, weights):
    values = pd.Series(values)
    weights = pd.Series(weights)
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])

def compute_hhi_from_totals(totals_dict):
    s = pd.Series(totals_dict, dtype=float).fillna(0)
    total = s.sum()
    if total <= 0:
        return np.nan
    shares = s / total
    return (shares.pow(2).sum()) * 10000

def same_month_last_year(ts):
    return pd.to_datetime(ts) - pd.DateOffset(years=1)

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluar_pronostico(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)

    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

def diagnosticar_serie(serie, seasonal_period=12):
    resultados = {
        "n_obs": len(serie),
        "estacionaria": None,
        "p_value_adf": np.nan,
        "seasonality_strength": np.nan,
        "seasonal_period": seasonal_period
    }

    serie = serie.dropna()

    if len(serie) >= 12 and STATSMODELS_OK:
        try:
            adf_result = adfuller(serie)
            resultados["p_value_adf"] = adf_result[1]
            resultados["estacionaria"] = adf_result[1] < 0.05
        except Exception:
            pass

    if len(serie) >= 24 and STATSMODELS_OK:
        try:
            stl = STL(serie, period=seasonal_period, robust=True)
            res = stl.fit()
            var_resid = np.var(res.resid)
            var_seasonal_resid = np.var(res.seasonal + res.resid)
            if var_seasonal_resid > 0:
                resultados["seasonality_strength"] = max(0, 1 - (var_resid / var_seasonal_resid))
        except Exception:
            pass

    return resultados

def seasonal_naive_forecast(train, test_len, horizon=6, seasonal_period=12):
    train = pd.Series(train).dropna()
    if len(train) < seasonal_period:
        raise ValueError("No hay suficientes datos para seasonal naive")

    last_season = train.iloc[-seasonal_period:].values
    pred_test = np.resize(last_season, test_len)
    future_pred = np.resize(last_season, horizon)

    return np.array(pred_test), np.array(future_pred)

def naive_forecast(train, test_len, horizon=6):
    last_value = float(pd.Series(train).dropna().iloc[-1])
    pred_test = np.repeat(last_value, test_len)
    future_pred = np.repeat(last_value, horizon)
    return np.array(pred_test), np.array(future_pred)

def linear_trend_forecast(train, full_series, test_len, horizon=6):
    train = pd.Series(train).dropna()
    full_series = pd.Series(full_series).dropna()

    X_train = np.arange(len(train)).reshape(-1, 1)
    X_test = np.arange(len(train), len(train) + test_len).reshape(-1, 1)
    X_future = np.arange(len(full_series), len(full_series) + horizon).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, train.values)
    pred_test = model.predict(X_test)

    model_full = LinearRegression()
    X_full = np.arange(len(full_series)).reshape(-1, 1)
    model_full.fit(X_full, full_series.values)
    future_pred = model_full.predict(X_future)

    return np.array(pred_test), np.array(future_pred)

def generar_indice_futuro(serie, horizon=6):
    return pd.date_range(
        start=serie.index[-1] + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

def seleccionar_mejor_modelo(serie, horizon=6, seasonal_period=12):
    serie = pd.Series(serie).dropna().astype(float)

    if len(serie) < 18:
        raise ValueError("Se requieren al menos 18 observaciones para selección automática robusta.")

    test_len = min(horizon, max(4, len(serie) // 5))
    train = serie.iloc[:-test_len]
    test = serie.iloc[-test_len:]

    diagnostico = diagnosticar_serie(serie, seasonal_period=seasonal_period)
    candidatos = []

    def registrar_modelo(nombre, pred_test, future_pred):
        metricas = evaluar_pronostico(test.values, pred_test)
        candidatos.append({
            "Modelo": nombre,
            "MAE": metricas["MAE"],
            "RMSE": metricas["RMSE"],
            "MAPE": metricas["MAPE"],
            "R2": metricas["R2"],
            "pred_test": np.array(pred_test),
            "future_pred": np.array(future_pred)
        })

    # Modelos base
    try:
        pred_test, future_pred = naive_forecast(train, len(test), horizon=horizon)
        registrar_modelo("Naive", pred_test, future_pred)
    except Exception:
        pass

    try:
        if len(train) >= seasonal_period:
            pred_test, future_pred = seasonal_naive_forecast(train, len(test), horizon=horizon, seasonal_period=seasonal_period)
            registrar_modelo("Seasonal Naive", pred_test, future_pred)
    except Exception:
        pass

    try:
        pred_test, future_pred = linear_trend_forecast(train, serie, len(test), horizon=horizon)
        registrar_modelo("Regresión lineal", pred_test, future_pred)
    except Exception:
        pass

    if STATSMODELS_OK:
        # Suavizamiento simple
        try:
            fit = SimpleExpSmoothing(train).fit()
            pred_test = fit.forecast(len(test))
            fit_full = SimpleExpSmoothing(serie).fit()
            future_pred = fit_full.forecast(horizon)
            registrar_modelo("SES", pred_test, future_pred)
        except Exception:
            pass

        # Holt
        try:
            fit = Holt(train, exponential=False, damped_trend=False).fit()
            pred_test = fit.forecast(len(test))
            fit_full = Holt(serie, exponential=False, damped_trend=False).fit()
            future_pred = fit_full.forecast(horizon)
            registrar_modelo("Holt", pred_test, future_pred)
        except Exception:
            pass

        # Holt amortiguado
        try:
            fit = Holt(train, exponential=False, damped_trend=True).fit()
            pred_test = fit.forecast(len(test))
            fit_full = Holt(serie, exponential=False, damped_trend=True).fit()
            future_pred = fit_full.forecast(horizon)
            registrar_modelo("Holt amortiguado", pred_test, future_pred)
        except Exception:
            pass

        # Holt-Winters aditivo
        try:
            if len(train) >= 24:
                fit = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_period
                ).fit()
                pred_test = fit.forecast(len(test))

                fit_full = ExponentialSmoothing(
                    serie,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_period
                ).fit()
                future_pred = fit_full.forecast(horizon)
                registrar_modelo("Holt-Winters aditivo", pred_test, future_pred)
        except Exception:
            pass

        # Holt-Winters multiplicativo
        try:
            if len(train) >= 24 and (train > 0).all() and (serie > 0).all():
                fit = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="mul",
                    seasonal_periods=seasonal_period
                ).fit()
                pred_test = fit.forecast(len(test))

                fit_full = ExponentialSmoothing(
                    serie,
                    trend="add",
                    seasonal="mul",
                    seasonal_periods=seasonal_period
                ).fit()
                future_pred = fit_full.forecast(horizon)
                registrar_modelo("Holt-Winters multiplicativo", pred_test, future_pred)
        except Exception:
            pass

        # ARIMA / SARIMA
        arima_orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)]
        seasonal_orders = [
            (1, 0, 0, 12),
            (0, 1, 1, 12),
            (1, 1, 1, 12)
        ]

        for order in arima_orders:
            try:
                fit = ARIMA(train, order=order).fit()
                pred_test = fit.forecast(len(test))

                fit_full = ARIMA(serie, order=order).fit()
                future_pred = fit_full.forecast(horizon)
                registrar_modelo(f"ARIMA{order}", pred_test, future_pred)
            except Exception:
                pass

            if len(train) >= 24:
                for sorder in seasonal_orders:
                    try:
                        fit = SARIMAX(
                            train,
                            order=order,
                            seasonal_order=sorder,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        ).fit(disp=False)

                        pred_test = fit.forecast(len(test))

                        fit_full = SARIMAX(
                            serie,
                            order=order,
                            seasonal_order=sorder,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        ).fit(disp=False)

                        future_pred = fit_full.forecast(horizon)
                        registrar_modelo(f"SARIMA{order}x{sorder}", pred_test, future_pred)
                    except Exception:
                        pass

    if len(candidatos) == 0:
        raise ValueError("No se pudo ajustar ningún modelo candidato.")

    tabla = pd.DataFrame(candidatos).drop(columns=["pred_test", "future_pred"])
    tabla = tabla.sort_values(["RMSE", "MAPE"], ascending=[True, True]).reset_index(drop=True)

    mejor_nombre = tabla.iloc[0]["Modelo"]
    mejor = [c for c in candidatos if c["Modelo"] == mejor_nombre][0]

    return {
        "train": train,
        "test": test,
        "diagnostico": diagnostico,
        "leaderboard": tabla,
        "best_model": mejor
    }

def resumen_por_fecha(df_in, metrica_principal):
    registros = []
    for fecha, g in df_in.groupby("Fecha"):
        registros.append({
            "Fecha": fecha,
            "Ocupacion_Regional": weighted_mean(g["Porcentaje de ocupación"], g["Oferta_total"]),
            "ADR_Promedio": g["Tarifa diaria promedio"].mean(),
            "RevPAR_Promedio": g["RevPAR"].mean(),
            "Metrica_Principal": (
                weighted_mean(g["Porcentaje de ocupación"], g["Oferta_total"])
                if metrica_principal == "Porcentaje de ocupación"
                else g[metrica_principal].mean()
            )
        })
    return pd.DataFrame(registros).sort_values("Fecha")


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
            "Año",
            "Porcentaje de ocupación",
            "Estancia promedio",
            "Días reservados",
            "Listados reservados",
            "Ingresos mensuales",
            "Tarifa diaria promedio",
            "Ingreso por habitación disponible",
            "Listados 1 recámara",
            "Listados 2 recámaras",
            "Listados 3 recámaras",
            "Listados 4 recámaras",
            "Listados 5 recámaras",
            "Listados 6+ recámaras",
            "Percentil 25 ingresos",
            "Mediana ingresos",
            "Percentil 75 ingresos",
            "Percentil 90 ingresos"
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

        df["Oferta_total"] = df[
            ["Oferta_1_recamara", "Oferta_2_recamaras", "Oferta_3_recamaras", "Oferta_4_plus"]
        ].sum(axis=1)

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

        shares_hhi = df[
            ["Oferta_1_recamara", "Oferta_2_recamaras", "Oferta_3_recamaras", "Oferta_4_plus"]
        ].div(df["Oferta_total"].replace(0, np.nan), axis=0)
        df["HHI_Recamaras"] = shares_hhi.pow(2).sum(axis=1) * 10000

        df["Spread_Ingresos"] = df["Percentil 75 ingresos"] - df["Percentil 25 ingresos"]
        df["Ratio_P90_P50"] = safe_divide(df["Percentil 90 ingresos"], df["Mediana ingresos"])

        df = df.sort_values(["Destino", "Fecha"]).copy()

        df["YoY_Ocupacion"] = df.groupby("Destino")["Porcentaje de ocupación"].pct_change(12) * 100
        df["YoY_ADR"] = df.groupby("Destino")["Tarifa diaria promedio"].pct_change(12) * 100
        df["YoY_RevPAR"] = df.groupby("Destino")["RevPAR"].pct_change(12) * 100
        df["YoY_Oferta"] = df.groupby("Destino")["Oferta_total"].pct_change(12) * 100

        df["Vol_Ocupacion_12m"] = df.groupby("Destino")["Porcentaje de ocupación"].transform(
            lambda x: x.rolling(12, min_periods=6).std()
        )

        df["Vol_RevPAR_12m"] = df.groupby("Destino")["RevPAR"].transform(
            lambda x: x.rolling(12, min_periods=6).std()
        )

        df["Volatilidad"] = df["Vol_Ocupacion_12m"]

        occ_chg = df.groupby("Destino")["Porcentaje de ocupación"].pct_change()
        adr_chg = df.groupby("Destino")["Tarifa diaria promedio"].pct_change()
        df["Elasticidad"] = safe_divide(occ_chg, adr_chg)

        rolling_peak_revpar = df.groupby("Destino")["RevPAR"].cummax()
        df["Drawdown_RevPAR"] = safe_divide(df["RevPAR"], rolling_peak_revpar) - 1

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
# SIDEBAR
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
            "Presion_Mercado",
            "YoY_Oferta",
            "HHI_Recamaras"
        ],
        index=0
    )

    top_n = st.slider("🔢 Top N destinos a mostrar", 3, 10, 5)

    tipo_prop = st.multiselect(
        "🏠 Composición de oferta (solo gráfico de estructura)",
        ["1 recámara", "2 recámaras", "3 recámaras", "4+ recámaras"],
        default=["1 recámara", "2 recámaras", "3 recámaras", "4+ recámaras"]
    )

    st.markdown("---")
    st.caption(f"📊 {len(df):,} registros | {df['Destino'].nunique()} destinos")

    if not STATSMODELS_OK:
        st.info("📦 Para modelos avanzados (ETS/SARIMA/ARIMA): `pip install statsmodels`")

    st.markdown("---")
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-title">👨‍🔬 Dr. César Omar Sepúlveda Moreno</div>
            <div class="profile-link">📖 <a href="{ORCID_URL}" target="_blank">ORCID</a></div>
            <div class="profile-link">🎓 <a href="{SCHOLAR_URL}" target="_blank">Google Scholar</a></div>
            <div class="profile-link">📚 <a href="{SCOPUS_URL}" target="_blank">Scopus</a></div>
            <div class="profile-link">💼 <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a></div>
            <div class="profile-link">✉️ <a href="{EMAIL_URL}">Email</a></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FILTROS
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
    "YoY_Ocupacion": "Crecimiento YoY ocupación (%)",
    "Presion_Mercado": "Presión de mercado",
    "YoY_Oferta": "Crecimiento YoY oferta (%)",
    "HHI_Recamaras": "HHI recámaras"
}
titulo_metrica = mapa_titulos.get(metrica_principal, metrica_principal)


# =========================================================
# ENCABEZADO
# =========================================================
st.markdown('<div class="main-title">SIIMTUR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Sistema Integral de Inteligencia de Mercados Turísticos</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="insight-card">
        <strong>🔗 Enlaces académicos del autor:</strong>
        <a href="{ORCID_URL}" target="_blank">ORCID</a> ·
        <a href="{SCHOLAR_URL}" target="_blank">Google Scholar</a> ·
        <a href="{SCOPUS_URL}" target="_blank">Scopus</a> ·
        <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
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

    ultima_fecha = df_filt["Fecha"].max()
    df_ultimo = df_filt[df_filt["Fecha"] == ultima_fecha].copy()

    fecha_base = same_month_last_year(ultima_fecha)
    df_base = df_filt[df_filt["Fecha"] == fecha_base].copy()

    ocupacion_kpi = weighted_mean(df_ultimo["Porcentaje de ocupación"], df_ultimo["Oferta_total"])

    ocupacion_base = weighted_mean(
        df_base["Porcentaje de ocupación"],
        df_base["Oferta_total"]
    ) if not df_base.empty else np.nan

    yoy_ocupacion_kpi = ((ocupacion_kpi / ocupacion_base) - 1) * 100 if pd.notna(ocupacion_base) and ocupacion_base != 0 else np.nan

    oferta_actual = df_ultimo["Oferta_total"].sum()
    oferta_base = df_base["Oferta_total"].sum() if not df_base.empty else np.nan
    yoy_oferta_kpi = ((oferta_actual / oferta_base) - 1) * 100 if pd.notna(oferta_base) and oferta_base != 0 else np.nan

    hhi_kpi = compute_hhi_from_totals({
        "Oferta_1_recamara": df_ultimo["Oferta_1_recamara"].sum(),
        "Oferta_2_recamaras": df_ultimo["Oferta_2_recamaras"].sum(),
        "Oferta_3_recamaras": df_ultimo["Oferta_3_recamaras"].sum(),
        "Oferta_4_plus": df_ultimo["Oferta_4_plus"].sum()
    })

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🏨 OCUPACIÓN REGIONAL</div>
                <div class='metric-value'>{format_pct(ocupacion_kpi)}</div>
                <div class='metric-sub'>Último mes disponible · ponderada por oferta</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>📈 YOY OCUPACIÓN</div>
                <div class='metric-value'>{format_pct(yoy_ocupacion_kpi)}</div>
                <div class='metric-sub'>Vs. mismo mes del año previo</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🏢 YOY OFERTA</div>
                <div class='metric-value'>{format_pct(yoy_oferta_kpi)}</div>
                <div class='metric-sub'>Cambio interanual de oferta total</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🧩 HHI RECÁMARAS</div>
                <div class='metric-value'>{format_number(hhi_kpi)}</div>
                <div class='metric-sub'>Concentración de oferta por tipo</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Evolución del Mercado</div>', unsafe_allow_html=True)

    evol = resumen_por_fecha(df_filt, metrica_principal)

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Ocupación regional", "ADR", "RevPAR", titulo_metrica))
    fig.add_trace(go.Scatter(x=evol["Fecha"], y=evol["Ocupacion_Regional"], mode="lines+markers", line=dict(color="#2563eb")), row=1, col=1)
    fig.add_trace(go.Scatter(x=evol["Fecha"], y=evol["ADR_Promedio"], mode="lines+markers", line=dict(color="#10b981")), row=1, col=2)
    fig.add_trace(go.Scatter(x=evol["Fecha"], y=evol["RevPAR_Promedio"], mode="lines+markers", line=dict(color="#f59e0b")), row=2, col=1)
    fig.add_trace(go.Scatter(x=evol["Fecha"], y=evol["Metrica_Principal"], mode="lines+markers", line=dict(color="#ef4444")), row=2, col=2)
    fig.update_layout(height=560, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    ranking_ocup = {}
    for destino, g in df_filt.groupby("Destino"):
        ranking_ocup[destino] = weighted_mean(g["Porcentaje de ocupación"], g["Oferta_total"])

    top_ocup = max(ranking_ocup, key=ranking_ocup.get) if ranking_ocup else "-"
    top_aporte = df_filt.groupby("Destino")["Aporte_Mercado"].mean().idxmax()
    mas_volatil = df_filt.groupby("Destino")["Vol_Ocupacion_12m"].mean().idxmax()
    mejor_crecimiento = df_filt.groupby("Destino")["YoY_Ocupacion"].mean().idxmax()

    st.markdown(
        f"""
        <div class='insight-card'>
            <strong>📌 Insight Ejecutivo</strong><br>
            • <strong>{top_ocup}</strong> lidera en ocupación ponderada por oferta.<br>
            • <strong>{top_aporte}</strong> es el principal aportador al mercado regional.<br>
            • <strong>{mas_volatil}</strong> presenta la mayor volatilidad reciente.<br>
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
            "Presion_Mercado",
            "YoY_Ocupacion",
            "YoY_Oferta"
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
    estacional = df_filt.groupby("Mes_Cat", observed=False)["Porcentaje de ocupación"].agg(["mean", "std"]).reset_index()
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
        "YoY_Oferta": "mean",
        "Vol_Ocupacion_12m": "mean",
        "Vol_RevPAR_12m": "mean",
        "Conversion_Rate": "mean",
        "Presion_Mercado": "mean",
        "HHI_Recamaras": "mean",
        "Spread_Ingresos": "mean",
        "Ratio_P90_P50": "mean",
        "Drawdown_RevPAR": "mean"
    }).reset_index()

    ranking["Score_Ocup"] = normalizar_0_100(ranking["Porcentaje de ocupación"])
    ranking["Score_RevPAR"] = normalizar_0_100(ranking["RevPAR"])
    ranking["Score_Aporte"] = normalizar_0_100(ranking["Aporte_Mercado"])
    ranking["Score_Crecimiento"] = normalizar_0_100(ranking["YoY_Ocupacion"].fillna(0))
    ranking["Score_Conversion"] = normalizar_0_100(ranking["Conversion_Rate"].fillna(0))
    ranking["Score_Riesgo"] = 100 - normalizar_0_100(ranking["Vol_Ocupacion_12m"].fillna(ranking["Vol_Ocupacion_12m"].median()))

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
        "YoY_Oferta": "mean",
        "Vol_Ocupacion_12m": "mean",
        "Conversion_Rate": "mean",
        "Presion_Mercado": "mean",
        "HHI_Recamaras": "mean"
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

    st.markdown('<div class="section-title">Indicadores estructurales adicionales</div>', unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        hhi_prom = df_filt["HHI_Recamaras"].mean()
        st.metric("HHI promedio", f"{hhi_prom:.1f}" if pd.notna(hhi_prom) else "-")

    with col_b:
        spread_prom = df_filt["Spread_Ingresos"].mean()
        st.metric("Spread ingresos", format_money(spread_prom) if pd.notna(spread_prom) else "-")

    with col_c:
        ratio_prom = df_filt["Ratio_P90_P50"].mean()
        st.metric("Ratio P90/P50", f"{ratio_prom:.2f}" if pd.notna(ratio_prom) else "-")

    with col_d:
        dd_prom = df_filt["Drawdown_RevPAR"].mean()
        st.metric("Drawdown RevPAR", format_pct(dd_prom * 100) if pd.notna(dd_prom) else "-")

    st.markdown('<div class="section-title">Alertas Estratégicas</div>', unsafe_allow_html=True)

    alertas_contador = 0
    for _, row in ranking.iterrows():
        alertas = []
        if row["Porcentaje de ocupación"] < 35:
            alertas.append("⚠️ baja ocupación (<35%)")
        if pd.notna(row["YoY_Ocupacion"]) and row["YoY_Ocupacion"] < -5:
            alertas.append("📉 contracción anual (>5%)")
        if pd.notna(row["Vol_Ocupacion_12m"]) and row["Vol_Ocupacion_12m"] > 15:
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
            "Porcentaje de ocupación",
            "Tarifa diaria promedio",
            "RevPAR",
            "Estancia promedio",
            "Días reservados",
            "Oferta_total",
            "Conversion_Rate",
            "Presion_Mercado",
            "Participacion_Oferta",
            "YoY_Ocupacion",
            "YoY_Oferta",
            "HHI_Recamaras",
            "Spread_Ingresos",
            "Ratio_P90_P50",
            "Vol_RevPAR_12m"
        ],
        default=[
            "Porcentaje de ocupación",
            "Tarifa diaria promedio",
            "RevPAR",
            "Estancia promedio"
        ]
    )

    if len(vars_corr) >= 2:
        corr = df_filt[vars_corr].corr(numeric_only=True)
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
        vars_exp = [
            "Estancia promedio",
            "Oferta_total",
            "Presion_Mercado",
            "Participacion_Demanda",
            "Mes_Sin",
            "Mes_Cos",
            "YoY_Oferta",
            "HHI_Recamaras"
        ]
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

    cluster_vars = [
        "Porcentaje de ocupación",
        "Tarifa diaria promedio",
        "RevPAR",
        "Estancia promedio",
        "Conversion_Rate",
        "YoY_Ocupacion",
        "YoY_Oferta"
    ]
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
    st.markdown('<div class="section-title">Predicción automática de ocupación</div>', unsafe_allow_html=True)

    destino_pred = st.selectbox("Selecciona un destino para forecast:", destinos_sel)

    modo_modelo = st.selectbox(
        "Modo de selección del modelo:",
        [
            "Automático (mejor modelo)",
            "Regresión lineal",
            "ETS (Holt-Winters)",
            "SARIMA"
        ],
        index=0
    )

    serie = (
        df_filt[df_filt["Destino"] == destino_pred]
        .sort_values("Fecha")
        .set_index("Fecha")["Porcentaje de ocupación"]
        .dropna()
        .astype(float)
    )

    if len(serie) >= 18:
        if modo_modelo == "Automático (mejor modelo)":
            try:
                resultado = seleccionar_mejor_modelo(serie, horizon=6, seasonal_period=12)

                train = resultado["train"]
                test = resultado["test"]
                diagnostico = resultado["diagnostico"]
                leaderboard = resultado["leaderboard"]
                best = resultado["best_model"]

                pred_test = best["pred_test"]
                future_pred = best["future_pred"]
                nombre_modelo = best["Modelo"]

                future_dates = generar_indice_futuro(serie, horizon=6)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Modelo seleccionado", nombre_modelo)
                with col2:
                    st.metric("RMSE", f"{best['RMSE']:.2f}")
                with col3:
                    st.metric("MAPE", "-" if pd.isna(best["MAPE"]) else f"{best['MAPE']:.1f}%")
                with col4:
                    st.metric("R²", "-" if pd.isna(best["R2"]) else f"{best['R2']:.3f}")

                st.markdown('<div class="section-title">Diagnóstico de la serie</div>', unsafe_allow_html=True)

                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    est_txt = "Sí" if diagnostico["estacionaria"] is True else "No" if diagnostico["estacionaria"] is False else "-"
                    st.metric("¿Estacionaria?", est_txt)
                with col_d2:
                    st.metric("p-value ADF", "-" if pd.isna(diagnostico["p_value_adf"]) else f"{diagnostico['p_value_adf']:.4f}")
                with col_d3:
                    st.metric("Fuerza estacional", "-" if pd.isna(diagnostico["seasonality_strength"]) else f"{diagnostico['seasonality_strength']:.3f}")

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines+markers", name="Entrenamiento", line=dict(color="#2563eb")))
                fig_pred.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="Real", line=dict(color="#10b981")))
                fig_pred.add_trace(go.Scatter(x=test.index, y=pred_test, mode="lines+markers", name="Predicción validación", line=dict(dash="dash", color="#f59e0b")))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines+markers", name="Forecast 6 meses", line=dict(dash="dot", color="#ef4444")))
                fig_pred.update_layout(
                    title=f"Mejor modelo para {destino_pred}: {nombre_modelo}",
                    xaxis_title="Fecha",
                    yaxis_title="Ocupación (%)",
                    height=520,
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_pred, width="stretch")

                st.markdown('<div class="section-title">Comparativo de modelos</div>', unsafe_allow_html=True)
                st.dataframe(leaderboard, width="stretch", hide_index=True)

                pred_df = pd.DataFrame({
                    "Período": [f"Mes +{i+1}" for i in range(6)],
                    "Fecha": future_dates.strftime("%B %Y"),
                    "Pronóstico (%)": np.round(np.array(future_pred), 1)
                })
                st.markdown("#### 📋 Pronóstico a 6 meses")
                st.dataframe(pred_df, width="stretch", hide_index=True)

                if pd.notna(best["R2"]):
                    if best["R2"] < 0:
                        interpretacion = "El modelo seleccionado aún supera al resto comparado, pero su poder predictivo es bajo y la serie presenta comportamiento difícil de anticipar."
                    elif best["R2"] < 0.3:
                        interpretacion = "La capacidad predictiva es baja."
                    elif best["R2"] < 0.6:
                        interpretacion = "La capacidad predictiva es moderada."
                    else:
                        interpretacion = "La capacidad predictiva es alta."
                else:
                    interpretacion = "La interpretación se apoya principalmente en RMSE y MAPE."

                st.markdown(
                    f"""
                    <div class='insight-card'>
                        <strong>Interpretación automática:</strong><br>
                        • SIIMTUR evaluó múltiples modelos y seleccionó <strong>{nombre_modelo}</strong> por presentar el menor error de validación.<br>
                        • {interpretacion}<br>
                        • RMSE: <strong>{best['RMSE']:.2f}</strong> |
                          MAPE: <strong>{'-' if pd.isna(best['MAPE']) else f"{best['MAPE']:.1f}%"}</strong> |
                          R²: <strong>{'-' if pd.isna(best['R2']) else f"{best['R2']:.3f}"}</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if STATSMODELS_OK and len(serie) >= 24:
                    try:
                        st.markdown('<div class="section-title">Descomposición estacional (STL)</div>', unsafe_allow_html=True)
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

            except Exception as e:
                st.error(f"No fue posible seleccionar automáticamente el mejor modelo: {e}")

        else:
            predictions = None
            future_pred = None
            train = None
            test = None
            nombre_modelo = modo_modelo

            if modo_modelo == "Regresión lineal":
                train, test, predictions, future_pred = linear_trend_forecast(
                    serie.iloc[:-max(4, len(serie) // 5)],
                    serie,
                    max(4, len(serie) // 5),
                    horizon=6
                )
                train = serie.iloc[:-max(4, len(serie) // 5)]
                test = serie.iloc[-max(4, len(serie) // 5):]

            elif modo_modelo == "ETS (Holt-Winters)" and STATSMODELS_OK:
                try:
                    train_size = int(len(serie) * 0.8)
                    train, test = serie[:train_size], serie[train_size:]
                    ets = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
                    fit = ets.fit()
                    predictions = fit.forecast(len(test))
                    fit_full = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=12).fit()
                    future_pred = fit_full.forecast(6)
                except Exception as e:
                    st.warning(f"ETS no pudo ajustarse: {str(e)[:100]}")

            elif modo_modelo == "SARIMA" and STATSMODELS_OK:
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
                    st.warning(f"SARIMA no pudo ajustarse: {str(e)[:100]}")

            if predictions is not None and future_pred is not None and test is not None and len(test) > 0:
                metricas = evaluar_pronostico(test.values, np.array(predictions))
                future_dates = generar_indice_futuro(serie, horizon=6)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{metricas['RMSE']:.2f}")
                with col2:
                    st.metric("MAPE", "-" if pd.isna(metricas["MAPE"]) else f"{metricas['MAPE']:.1f}%")
                with col3:
                    st.metric("R²", "-" if pd.isna(metricas["R2"]) else f"{metricas['R2']:.3f}")

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines+markers", name="Entrenamiento", line=dict(color="#2563eb")))
                fig_pred.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="Real", line=dict(color="#10b981")))
                fig_pred.add_trace(go.Scatter(x=test.index, y=np.array(predictions), mode="lines+markers", name="Predicción", line=dict(dash="dash", color="#f59e0b")))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=np.array(future_pred), mode="lines+markers", name="Forecast 6 meses", line=dict(dash="dot", color="#ef4444")))
                fig_pred.update_layout(
                    title=f"Predicción de ocupación - {destino_pred} ({nombre_modelo})",
                    xaxis_title="Fecha",
                    yaxis_title="Ocupación (%)",
                    height=520,
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_pred, width="stretch")
    else:
        st.warning(f"Se requieren al menos 18 meses de datos para selección automática robusta. Actualmente: {len(serie)} meses.")


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
            <div class="metric-sub">• Naive<br>• Seasonal Naive<br>• SES / Holt / Holt-Winters<br>• ARIMA / SARIMA<br>• STL</div>
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
        • <strong>Vol_Ocupacion_12m</strong>: volatilidad móvil de la ocupación.<br>
        • <strong>Vol_RevPAR_12m</strong>: volatilidad móvil de RevPAR.<br>
        • <strong>HHI_Recamaras</strong>: concentración de la oferta por tipo de recámaras.<br>
        • <strong>Ratio_P90_P50</strong>: relación entre percentil 90 y mediana de ingresos.<br>
        • <strong>Drawdown_RevPAR</strong>: caída relativa de RevPAR respecto a su máximo acumulado.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    f"""
    <div class="footer">
        <div><strong>SIIMTUR</strong> · Sistema Integral de Inteligencia de Mercados Turísticos</div>
        <div style="margin-top: 8px;">
            <a href="{ORCID_URL}" target="_blank">ORCID</a> ·
            <a href="{SCHOLAR_URL}" target="_blank">Google Scholar</a> ·
            <a href="{SCOPUS_URL}" target="_blank">Scopus</a> ·
            <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
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
