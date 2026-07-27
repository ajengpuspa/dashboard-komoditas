import streamlit as st

from utils import init_session_state, inject_custom_css, render_sidebar

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Komoditas Pangan Dashboard", layout="wide")

init_session_state()
inject_custom_css()
render_sidebar()

# ============================================================
# MAIN PAGE
# ============================================================
st.markdown("`app / homepage`")

with st.container(border=True):

    # ---- LOGO ----
    st.markdown(
        """
        <div style="display:flex;justify-content:center;margin-top:10px;">
            <div style="
                width:72px;
                height:72px;
                border-radius:18px;
                background:linear-gradient(135deg,#FF4B4B,#FFB199);
                display:flex;
                justify-content:center;
                align-items:center;
                color:white;
                font-size:34px;
                font-weight:800;">
                Rp
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- TITLE ----
    st.markdown(
        """
        <div style="text-align:center;font-size:25px;font-weight:700;margin-top:8px;color:#31333F;">
            Integrasi Model Kecerdasan Buatan dan Stokastik
            Melalui Stacking Ensemble Learning untuk
            Prediksi Harga dan Risiko Komoditas Pangan
            di Indonesia
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align:center;font-size:17px;font-weight:700;margin-top:8px;color:#31333F;">
            Disusun oleh: Mohammad Idhom, Trimono, Ajeng Puspa, Shafira Amanda
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align:center;font-size:16px;color:#6E6E80;margin-bottom:36px;">
            Sains Data — Fakultas Ilmu Komputer —
            UPN ''Veteran'' Jawa Timur, 2026
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- FLOW ----
    flow_steps = [
        ("📂 Data Historis", "Data harga komoditas\n(Data Train & Test)"),
        ("🧠 Base Learners", "SARIMA, SVR, dan Random Forest"),
        ("📊 Meta Learner", "Linear Regression"),
        ("📈 Prediksi Akhir", "Hasil Stacking\nEnsemble"),
    ]

    cols = st.columns([2.3, 0.5, 2.3, 0.5, 2.3, 0.5, 2.3])
    step_cols = cols[0::2]
    arrow_cols = cols[1::2]

    for col, (label, desc) in zip(step_cols, flow_steps):
        with col:
            st.info(f"##### {label}\n\n{desc}")

    for col in arrow_cols:
        with col:
            st.markdown("<h1 style='text-align:center;padding-top:30px;'>➜</h1>", unsafe_allow_html=True)

st.write("")

# ============================================================
# INFO CARDS
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <h3>🎯 Tujuan Aplikasi</h3>
            <p>
            Menghasilkan prediksi harga komoditas pangan yang lebih akurat
            melalui pendekatan <b>Stacking Ensemble Learning</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h3>🌾 Cakupan Komoditas</h3>
            <p>Cabai rawit merah, telur, dan bawang merah — dapat disesuaikan pada menu Input Data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <h3>🧩 Meta Learner</h3>
            <p>
            Linear Regression mempelajari bobot optimal
            dari prediksi SARIMA, SVR,
            dan Random Forest sehingga menghasilkan
            prediksi akhir yang lebih stabil
            dibandingkan model tunggal.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("### 🔄 Alur Stacking Ensemble Learning")

st.markdown(
    """
1. Data historis dibagi menjadi **data latih** dan **data uji**.
2. Model **SARIMA**, **SVR**, dan **Random Forest** dilatih secara independen menggunakan data latih.
3. Masing-masing model menghasilkan prediksi pada data uji.
4. Seluruh hasil prediksi dijadikan fitur masukan bagi **Linear Regression** sebagai Meta Learner.
5. Meta Learner mempelajari kombinasi bobot terbaik sehingga menghasilkan **prediksi akhir (Stacking Ensemble)** yang lebih akurat dan stabil.
6. Prediksi akhir dapat digunakan untuk analisis risiko harga komoditas di masa depan.
"""
)

    # ---- TITLE ----
    st.markdown(
        """
        <h1 style="
            text-align:center;
            font-size:44px;
            font-weight:700;
            color:#31333F;
            line-height:1.28;
            max-width:1200px;
            margin:24px auto 0px auto;
        ">
        Integrasi Model Kecerdasan Buatan dan Stokastik
        Melalui Stacking Ensemble Learning untuk
        Prediksi Harga dan Risiko Komoditas Pangan
        di Indonesia
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align:center;font-size:20px;font-weight:700;margin-top:8px;color:#31333F;">
            Disusun oleh: Mohammad Idhom, Trimono, Ajeng Puspa, Shafira Amanda
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align:center;font-size:16px;color:#6E6E80;margin-bottom:36px;">
            Sains Data — Fakultas Ilmu Komputer —
            UPN ''Veteran'' Jawa Timur, 2026
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- FLOW ----
    flow_steps = [
        ("📂 Data Historis", "Data harga komoditas\n(Data Train & Test)"),
        ("🧠 Base Learners", "SARIMA, SVR, dan Random Forest"),
        ("📊 Meta Learner", "Linear Regression"),
        ("📈 Prediksi Akhir", "Hasil Stacking\nEnsemble"),
    ]

    cols = st.columns([2.3, 0.5, 2.3, 0.5, 2.3, 0.5, 2.3])
    step_cols = cols[0::2]
    arrow_cols = cols[1::2]

    for col, (label, desc) in zip(step_cols, flow_steps):
        with col:
            st.info(f"##### {label}\n\n{desc}")

    for col in arrow_cols:
        with col:
            st.markdown("<h1 style='text-align:center;padding-top:30px;'>➜</h1>", unsafe_allow_html=True)

st.write("")

# ============================================================
# INFO CARDS
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <h3>🎯 Tujuan Aplikasi</h3>
            <p>
            Menghasilkan prediksi harga komoditas pangan yang lebih akurat
            melalui pendekatan <b>Stacking Ensemble Learning</b> dengan
            mengombinasikan prediksi dari model SARIMA, SVR,
            dan Random Forest menggunakan
            <b>Linear Regression sebagai Meta Learner</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <h3>🌾 Cakupan Komoditas</h3>
            <p>Cabai rawit merah, telur, dan bawang merah — dapat disesuaikan pada menu Input Data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <h3>🧩 Meta Learner</h3>
            <p>
            Linear Regression mempelajari bobot optimal
            dari prediksi SARIMA, SVR,
            dan Random Forest sehingga menghasilkan
            prediksi akhir yang lebih stabil
            dibandingkan model tunggal.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### 🔄 Alur Stacking Ensemble Learning")

st.markdown(
    """
1. Data historis dibagi menjadi **data latih** dan **data uji**.
2. Model **SARIMA**, **SVR**, dan **Random Forest** dilatih secara independen menggunakan data latih.
3. Masing-masing model menghasilkan prediksi pada data uji.
4. Seluruh hasil prediksi dijadikan fitur masukan bagi **Linear Regression** sebagai Meta Learner.
5. Meta Learner mempelajari kombinasi bobot terbaik sehingga menghasilkan **prediksi akhir (Stacking Ensemble)** yang lebih akurat dan stabil.
6. Prediksi akhir dapat digunakan untuk analisis risiko harga komoditas di masa depan.
"""
)
