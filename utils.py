"""
utils.py
Kumpulan fungsi bersama yang dipakai di seluruh halaman KomoditasAI Dashboard:
- inisialisasi session_state
- styling (CSS) global & sidebar navigasi
- guard/validasi alur halaman (dataset & model harus tersedia)
- pembersihan kolom harga komoditas
- fungsi metrik evaluasi model & formatting angka ala Indonesia

Disatukan di sini supaya setiap halaman (homepage.py, pages/*.py) tidak perlu
mengulang kode yang sama, dan perubahan tampilan/perilaku cukup dilakukan
di satu tempat.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================================
# SESSION STATE
# ======================================================================

# Nilai default seluruh session_state yang dipakai lintas halaman.
SESSION_DEFAULTS = {
    "df": None,
    "original_df": None,
    "dataset_name": None,
    "date_column": None,
    "commodity_column": None,
    "analysis_range": None,
    "model_result": None,
    "model_data": None,
    "model_params": None,
}


def init_session_state():
    """Pastikan seluruh key session_state sudah terdaftar sebelum dipakai halaman mana pun."""
    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)


# ======================================================================
# STYLING (CSS GLOBAL)
# ======================================================================

_CUSTOM_CSS = """
<style>

/* ==========================================================
   GLOBAL
========================================================== */

html{
    font-size:14px;
}

body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"]{
    background:#FFFFFF;
    color:#31333F;
    font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}

/* ==========================================================
   STREAMLIT HEADER
========================================================== */

header[data-testid="stHeader"]{
    background:#FFFFFF !important;
    border-bottom:1px solid #E6E6E9;
}

[data-testid="stToolbar"]{
    background:#FFFFFF !important;
}

/* ==========================================================
   MAIN CONTAINER
========================================================== */

.block-container{
    max-width:1400px;
    padding:1rem 2rem 1.5rem;
}

/* ==========================================================
   TYPOGRAPHY
========================================================== */

h1{
    font-size:2.6rem !important;
    font-weight:800 !important;
    color:#31333F;
    margin-bottom:.5rem;
}

h2{
    font-size:2rem !important;
    font-weight:700 !important;
    color:#31333F;
    margin-top:1.4rem;
    margin-bottom:.6rem;
}

h3{
    font-size:1.45rem !important;
    font-weight:700 !important;
    color:#31333F;
}

h4{
    font-size:1.15rem !important;
    font-weight:600 !important;
    color:#31333F;
}

p,
span,
label,
li{
    font-size:13px;
    line-height:1.7;
    color:#31333F;
}

/* ==========================================================
   HERO TITLE (Homepage)
========================================================== */

.hero-title{
    display:block;
    text-align:center;
    font-size:44px;
    font-weight:800;
    line-height:1.25;
    color:#31333F;
}

/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"]{
    width:235px;
    background:#F5F6FA;
    border-right:1px solid #E6E6E9;
}

section[data-testid="stSidebarContent"]{
    padding:16px;
}

.sidebar-divider{
    border-top:1px solid #E6E6E9;
    margin:14px 0;
}

div[data-testid="stSidebarNav"]{
    display:none;
}

.sidebar-title{
    font-size:18px;
    font-weight:700;
    margin-bottom:10px;
}

/* ==========================================================
   PAGE LINK
========================================================== */

div[data-testid="stPageLink"]{
    margin-bottom:4px;
}

div[data-testid="stPageLink"] a{
    padding:8px 10px;
    border-radius:10px;
    font-size:14px;
    font-weight:500;
    color:#31333F;
    transition:.2s;
}

div[data-testid="stPageLink"] a:hover{
    background:#E8ECF5;
}

div[data-testid="stPageLink"][aria-current="page"] a{
    background:#FFF1EF;
    color:#E84C4C;
    font-weight:600;
}

div[data-testid="stPageLink"] a::before{
    content:"•";
    color:#9AA0A6;
    margin-right:6px;
}

div[data-testid="stPageLink"][aria-current="page"] a::before{
    color:#E84C4C;
}

/* ==========================================================
   CARD
========================================================== */
.card{
    background:#FFFFFF;
    border:1px solid #E6E6E9;
    border-radius:16px;
    padding:22px;
    min-height:260px;
}

.stMarkdown .card h3{
    font-size:28px !important;
    font-weight:700 !important;
    margin-bottom:12px !important;
    color:#31333F !important;
}

.stMarkdown .card p{
    font-size:14px !important;
    line-height:1.8 !important;
    color:#31333F !important;
}

/* ==========================================================
   INFO BOX
========================================================== */

div[data-testid="stInfo"]{
    border-radius:14px;
    border:1px solid #E6E6E9;
    padding:.8rem 1rem;
    font-size:14px;
}

div[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:18px;
}

/* ==========================================================
   METRIC
========================================================== */

[data-testid="stMetricLabel"]{
    font-size:14px !important;
}

[data-testid="stMetricValue"]{
    font-size:28px !important;
    font-weight:700;
}

[data-testid="stMetricDelta"]{
    font-size:13px !important;
}

[data-testid="stMetricDelta"] svg{
    display:none;
}

/* ==========================================================
   BUTTON
========================================================== */

.stButton button{
    border-radius:10px;
    font-size:14px;
    padding:.45rem 1rem;
}

/* ==========================================================
   INPUT
========================================================== */

.stTextInput label,
.stSelectbox label,
.stNumberInput label,
.stDateInput label,
.stRadio label,
.stCheckbox label{
    font-size:14px;
    font-weight:600;
}

.stTextInput input,
.stNumberInput input{
    font-size:14px;
}

.stSelectbox div[data-baseweb="select"]{
    font-size:14px;
}

/* ==========================================================
   DATAFRAME
========================================================== */

[data-testid="stDataFrame"]{
    font-size:14px;
}

/* ==========================================================
   TABS
========================================================== */

button[data-baseweb="tab"]{
    font-size:14px;
    padding:10px 18px;
}

</style>
"""

_LOGO_SMALL_HTML = """
<div style="
    margin-top:-15px;
    width:48px;
    height:48px;
    border-radius:12px;
    background:linear-gradient(135deg,#FF4B4B,#FFB199);
    display:flex;
    justify-content:center;
    align-items:center;
    color:white;
    font-weight:700;
    font-size:20px;">
    Rp
</div>
"""

# path halaman & label navigasi (urutan sesuai konsep tampilan awal)
NAV_ITEMS = [
    ("homepage.py", "Homepage"),
    ("pages/input_data.py", "Input Dataset"),
    ("pages/analisis_desk.py", "Analisis Deskriptif"),
    ("pages/input_params.py", "Input Parameter"),
    ("pages/output.py", "Output"),
]


def inject_custom_css():
    """Suntikkan CSS global (card, sidebar, page link, info box, container)."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar KomoditasAI: logo, judul, navigasi, dan footer."""
    with st.sidebar:
        c1, c2 = st.columns([0.75, 3.25], vertical_alignment="center")
        with c1:
            st.markdown(_LOGO_SMALL_HTML, unsafe_allow_html=True)
        with c2:
            st.markdown(
                "**KomoditasAI**  \n"
                "<span style='color:#6E6E80;font-size:13px'>Stacking Ensemble Dashboard</span>",
                unsafe_allow_html=True,
            )

        st.markdown('<div style="text-align:center;font-size:15px;font-weight:700;margin-top:8px;color:#31333F"></div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='sidebar-title'>Navigasi</div>",
            unsafe_allow_html=True
        )
        

        for path, label in NAV_ITEMS:
            st.page_link(path, label=label)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.caption("© 2026 • KomoditasAI Dashboard")


def page_header(breadcrumb: str, title: str, caption: str = ""):
    """Header konsisten untuk tiap halaman: breadcrumb, judul, dan sub-judul."""
    st.markdown(f"`{breadcrumb}`")
    st.title(title)
    if caption:
        st.caption(caption)


def setup_page(page_title: str, page_icon: str, breadcrumb: str, title: str, caption: str = ""):
    """
    Satu pemanggilan untuk seluruh boilerplate awal sebuah halaman:
    page_config -> session_state -> CSS -> sidebar -> header.
    Dipanggil paling atas, tepat setelah import.
    """
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    init_session_state()
    inject_custom_css()
    render_sidebar()
    page_header(breadcrumb, title, caption)


# ======================================================================
# GUARD / VALIDASI ALUR HALAMAN
# ======================================================================

def require_dataset():
    """
    Pastikan dataset & pemetaan kolom (tanggal, harga) sudah tersedia.
    Menghentikan halaman dengan pesan yang konsisten jika belum siap.
    """
    if st.session_state.get("df") is None or len(st.session_state.df) == 0:
        st.warning("Silakan unggah dan pilih dataset terlebih dahulu pada halaman **Input Dataset**.")
        st.stop()
    if st.session_state.get("date_column") is None:
        st.error("Kolom tanggal belum ditentukan pada halaman Input Dataset.")
        st.stop()
    if st.session_state.get("commodity_column") is None:
        st.error("Kolom harga belum ditentukan pada halaman Input Dataset.")
        st.stop()

    return (
        st.session_state.df.copy(),
        st.session_state.date_column,
        st.session_state.commodity_column,
    )


def require_trained_model():
    """Pastikan proses training (Input Parameter) sudah pernah dijalankan."""
    if st.session_state.get("model_result") is None:
        st.warning("Silakan jalankan proses training terlebih dahulu pada halaman **Input Parameter**.")
        st.stop()

    return (
        st.session_state.model_result,
        st.session_state.model_data,
        st.session_state.model_params,
    )


# ======================================================================
# PEMBERSIHAN DATA HARGA KOMODITAS
# ======================================================================

def clean_commodity_series(df: pd.DataFrame, commodity_column: str) -> pd.Series:
    """
    Bersihkan kolom harga komoditas dari format Rupiah (mis. "Rp12.345,67")
    menjadi nilai numerik (float), dan ubah placeholder ("-", "nan", dst) menjadi NaN.
    """
    cleaned = (
        df[commodity_column]
        .astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace(["-", "", "nan", "None"], np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ======================================================================
# FORMATTING ANGKA (GAYA INDONESIA)
# ======================================================================

def format_id(value, decimal: int = 0) -> str:
    """Format angka dengan pemisah ribuan '.' dan desimal ',' (gaya Indonesia)."""
    return f"{value:,.{decimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_rupiah(value) -> str:
    """Format angka menjadi teks Rupiah, mis. 12345.6 -> 'Rp12.345,6'."""
    text = format_id(value, decimal=2).rstrip("0").rstrip(",")
    return f"Rp{text}"


# ======================================================================
# METRIK EVALUASI MODEL (dipakai di Input Parameter & Output)
# ======================================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


def mase(y_true, y_pred, insample) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)
    scale = np.mean(np.abs(np.diff(insample)))
    if scale <= 1e-12:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_prediction(y_true, y_pred, insample, model_name: str) -> dict:
    """Ringkasan metrik evaluasi (RMSE, MAE, MAPE, sMAPE, MASE, R2, Bias) untuk satu model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": mape_safe(y_true, y_pred),
        "sMAPE (%)": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, insample),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }
