import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.stats.diagnostic import linear_rainbow, linear_reset

from utils import (
    clean_commodity_series,
    format_id,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_dataset,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Komoditas Pangan Dashboard", layout="wide")

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / analisis deskriptif",
    title="📊 Analisis Deskriptif",
    caption="Ringkasan statistik dan visualisasi pola data harga sebelum pemodelan dilakukan.",
)

RANDOM_STATE = 42

# ============================================================
# VALIDASI & PEMBERSIHAN DATA
# ============================================================
df, date_column, commodity_column = require_dataset()

working_df = df[[date_column, commodity_column]].copy()
working_df[date_column] = pd.to_datetime(working_df[date_column], dayfirst=True, errors="coerce")
working_df = working_df.dropna(subset=[date_column]).sort_values(date_column)
st.session_state.df = working_df

df[commodity_column] = clean_commodity_series(df, commodity_column)
df = df.dropna(subset=[commodity_column])

harga = df[commodity_column]


# ============================================================
# UJI NORMALITAS
# ============================================================
def normality_tests(series: pd.Series, label: str):
    x = pd.Series(series).dropna().astype(float)

    if len(x) > 5000:
        sample = x.sample(5000, random_state=RANDOM_STATE)
        note = "Shapiro menggunakan sampel acak 5.000 observasi."
    else:
        sample = x
        note = "Shapiro menggunakan seluruh observasi."

    shapiro_stat, shapiro_p = stats.shapiro(sample)
    jb = stats.jarque_bera(x)

    if len(x) >= 8:
        dagostino = stats.normaltest(x)
        dagostino_stat, dagostino_p = dagostino.statistic, dagostino.pvalue
    else:
        dagostino_stat, dagostino_p = np.nan, np.nan

    return {
        "Variabel": label,
        "N": len(x),
        "Shapiro": shapiro_stat,
        "Shapiro p-value": shapiro_p,
        "Jarque-Bera": jb.statistic,
        "JB p-value": jb.pvalue,
        "D'Agostino": dagostino_stat,
        "D'Agostino p-value": dagostino_p,
        "Kesimpulan": "Normal" if shapiro_p > 0.05 else "Tidak Normal",
        "Catatan": note,
    }


# ============================================================
# UJI LINEARITAS
# ============================================================
def build_lag_regression_data(series, n_lags=5):
    frame = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = series.shift(lag)
    return frame.dropna()


# ============================================================
# METRIK RINGKAS
# ============================================================
mean_val = harga.mean()
std_val = harga.std()
skew_val = stats.skew(harga)
kurt_val = stats.kurtosis(harga, fisher=False)

# -------- rata-rata kenaikan per bulan --------
plot_df = df.sort_values(date_column).copy()
monthly = plot_df.set_index(date_column)[commodity_column].resample("MS").mean()
monthly_change = monthly.pct_change().mean() * 100

if pd.isna(monthly_change):
    mean_delta = "-"
elif monthly_change > 0:
    mean_delta = f"▲ {monthly_change:.2f}% / bulan"
elif monthly_change < 0:
    mean_delta = f"▼ {abs(monthly_change):.2f}% / bulan"
else:
    mean_delta = "Tidak berubah"
mean_color = "off"

# -------- volatilitas --------
cv = std_val / mean_val
if cv < 0.10:
    volatility_delta, volatility_color = "▼ Rendah", "inverse"
elif cv < 0.20:
    volatility_delta, volatility_color = "■ Sedang", "off"
else:
    volatility_delta, volatility_color = "▲ Tinggi", "normal"

# -------- skewness --------
if abs(skew_val) < 0.50:
    skew_delta, skew_color = "● Simetris", "normal"
elif skew_val > 0:
    skew_delta, skew_color = "▶ Miring ke kanan", "inverse"
else:
    skew_delta, skew_color = "◀ Miring ke kiri", "off"

# -------- kurtosis --------
if kurt_val < 3:
    kurt_delta, kurt_color = "▼ Platykurtic", "inverse"
elif kurt_val <= 3.5:
    kurt_delta, kurt_color = "● Mesokurtic", "off"
else:
    kurt_delta, kurt_color = "▲ Leptokurtic", "normal"

# ============================================================
# METRIC CARDS
# ============================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rata-rata Harga", f"Rp {format_id(mean_val, 0)}", mean_delta, delta_color=mean_color)
m2.metric("Volatilitas", format_id(std_val, 1), volatility_delta, delta_color=volatility_color)
m3.metric("Skewness", f"{skew_val:.2f}", skew_delta, delta_color=skew_color)
m4.metric("Kurtosis", f"{kurt_val:.2f}", kurt_delta, delta_color=kurt_color)

st.write("")

# ============================================================
# TREN HARGA HISTORIS
# ============================================================
with st.container(border=True):
    st.markdown("#### 📈 Tren Harga Historis")

    plot_df = df.sort_values(date_column)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[date_column],
            y=plot_df[commodity_column],
            mode="lines",
            name=commodity_column,
            line=dict(color="#FF4B4B", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.08)",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        yaxis_title="Rp/kg",
        xaxis_title=None,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

st.write("")

# ============================================================
# TABEL STATISTIK & HISTOGRAM
# ============================================================
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### Tabel Statistik Deskriptif")
        stat_table = pd.DataFrame(
            {
                "Statistik": ["Mean", "Median", "Std. Deviasi", "Minimum", "Maksimum", "Skewness", "Kurtosis"],
                "Nilai": [
                    f"{mean_val:,.0f}".replace(",", "."),
                    f"{harga.median():,.0f}".replace(",", "."),
                    f"{std_val:,.1f}".replace(",", "."),
                    f"{harga.min():,.0f}".replace(",", "."),
                    f"{harga.max():,.0f}".replace(",", "."),
                    f"{skew_val:.2f}",
                    f"{kurt_val:.2f}",
                ],
            }
        )
        st.dataframe(stat_table, use_container_width=True, hide_index=True)

with col2:
    with st.container(border=True):
        st.markdown("#### Distribusi Harga (Histogram)")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df[commodity_column], nbinsx=8, marker_color="#2B6CB0"))
        fig_hist.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            xaxis_title="Rp/kg",
            yaxis_title="Frekuensi",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

st.write("")

# ============================================================
# UJI NORMALITAS
# ============================================================
with st.container(border=True):
    st.markdown("### 📋 Uji Normalitas")

    hasil_normalitas = pd.DataFrame(
        [normality_tests(harga, "Level Harga"), normality_tests(harga.diff(), "First Difference")]
    )
    st.dataframe(hasil_normalitas, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Q-Q Plot Level Harga")
        fig = plt.figure(figsize=(5, 5))
        sm.qqplot(harga.dropna(), line="45", fit=True, ax=fig.add_subplot(111))
        st.pyplot(fig)

    with col2:
        st.markdown("#### Q-Q Plot First Difference")
        fig = plt.figure(figsize=(5, 5))
        sm.qqplot(harga.diff().dropna(), line="45", fit=True, ax=fig.add_subplot(111))
        st.pyplot(fig)

st.write("")

# ============================================================
# UJI LINEARITAS
# ============================================================
with st.container(border=True):
    st.markdown("### 📈 Uji Linearitas")

    n_lags = min(5, max(1, len(harga) // 50))
    lin_data = build_lag_regression_data(harga, n_lags=n_lags)

    X = sm.add_constant(lin_data.drop(columns="y"))
    y = lin_data["y"]
    model = sm.OLS(y, X).fit()

    rainbow_stat, rainbow_p = linear_rainbow(model)
    reset = linear_reset(model, power=2, use_f=True)

    hasil_linearitas = pd.DataFrame(
        {
            "Uji": ["Rainbow Test", "Ramsey RESET"],
            "Statistik": [rainbow_stat, float(reset.fvalue)],
            "p-value": [rainbow_p, float(reset.pvalue)],
            "Keputusan": [
                "Linear" if rainbow_p > 0.05 else "Tidak Linear",
                "Linear" if float(reset.pvalue) > 0.05 else "Tidak Linear",
            ],
        }
    )
    st.dataframe(hasil_linearitas, use_container_width=True, hide_index=True)

    st.markdown("#### Scatter Plot Lag-1")
    st.write("")
    fig, ax = plt.subplots(
        figsize=(12, 5),
        dpi=150,
        constrained_layout=True
    )

    x = lin_data["lag_1"]
    y = lin_data["y"]
    ax.scatter(x, y, alpha=0.45)

    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = np.polyval(coef, xx)
    ax.plot(xx, yy, linewidth=2)

    ax.set_xlabel("Harga t-1")
    ax.set_ylabel("Harga t")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
    st.write("") 
    
with st.expander("Lihat Ringkasan Model OLS"):
    st.text(model.summary().as_text())
