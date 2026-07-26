"""
pages/input_data.py
Halaman untuk mengunggah data historis harga komoditas.
"""

import pandas as pd
import streamlit as st

from utils import init_session_state, inject_custom_css, render_sidebar, page_header

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Input Data", page_icon="📥", layout="wide")

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / input data",
    title="📥 Input Data",
    caption="Unggah data historis harga komoditas, atau gunakan data contoh yang tersedia pada sistem.",
)

# ============================================================
# UPLOAD DATASET
# ============================================================
with st.container(border=True):

    st.subheader("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Unggah dataset harga komoditas",
        type=["csv", "xlsx"],
        help="Format yang didukung: CSV dan Excel",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Simpan dataset asli & dataset aktif (masih sama pada awalnya)
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.dataset_name = uploaded_file.name

            st.success(f"Dataset **{uploaded_file.name}** berhasil dimuat.")
            st.markdown("---")

            # -------- RINGKASAN DATASET --------
            st.subheader("📑 Ringkasan Dataset")

            c1, c2, c3 = st.columns(3)
            c1.metric("Jumlah Baris", f"{len(df):,}")
            c2.metric("Jumlah Kolom", len(df.columns))
            c3.metric("Missing Value", int(df.isna().sum().sum()))

            st.markdown("---")

            # -------- PEMETAAN KOLOM --------
            st.subheader("🗂️ Pemetaan Kolom")

            col1, col2 = st.columns(2)
            with col1:
                date_column = st.selectbox("Kolom Tanggal", df.columns, index=0)
            with col2:
                commodity_column = st.selectbox(
                    "Kolom Harga Komoditas",
                    df.columns,
                    index=1 if len(df.columns) > 1 else 0,
                )

            st.session_state.date_column = date_column
            st.session_state.commodity_column = commodity_column

            # -------- RENTANG ANALISIS --------
            st.markdown("---")
            st.subheader("📅 Rentang Analisis")

            df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors="coerce")
            df = df.dropna(subset=[date_column]).sort_values(date_column)
            st.session_state.df = df

            min_date = df[date_column].min().date()
            max_date = df[date_column].max().date()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Tanggal Awal", min_value=min_date, max_value=max_date, value=None)
            with col2:
                end_date = st.date_input("Tanggal Akhir", min_value=min_date, max_value=max_date, value=None)

            if start_date and end_date:
                if start_date > end_date:
                    st.error("Tanggal awal tidak boleh melebihi tanggal akhir.")
                else:
                    st.session_state.analysis_range = (start_date, end_date)

                    filtered_df = df[
                        (df[date_column].dt.date >= start_date) & (df[date_column].dt.date <= end_date)
                    ].copy()

                    # dataset yang dipakai seluruh aplikasi
                    st.session_state.df = filtered_df

            st.caption(
                f"Rentang data tersedia: **{min_date.strftime('%d %b %Y')}** "
                f"hingga **{max_date.strftime('%d %b %Y')}**"
            )
            st.info(
                f"Data tersedia dari **{min_date.strftime('%d %b %Y')}** "
                f"sampai **{max_date.strftime('%d %b %Y')}**."
            )

        except Exception as e:
            st.error(e)

# ============================================================
# PRATINJAU DATA
# ============================================================
with st.container(border=True):
    st.markdown("#### Pratinjau Data")

    df = st.session_state.df
    if df is not None and len(df):
        preview = df.copy()
        if st.session_state.date_column in preview.columns:
            preview = preview[[st.session_state.date_column, st.session_state.commodity_column]]
        st.dataframe(preview, height=500, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Belum ada data dimuat. Gunakan panel di sebelah kiri untuk mengunggah "
            "berkas atau memakai data contoh."
        )
