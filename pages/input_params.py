import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import loguniform, randint
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import (
    clean_commodity_series,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_dataset,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Input Parameter Model — KomoditasAI", page_icon="⚙️", layout="wide")

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / input parameter",
    title="⚙️ Input Parameter Model",
    caption=(
        "Konfigurasi pembagian data, feature engineering, tuning model, "
        "dan parameter forecasting sebelum model dijalankan."
    ),
)

# ============================================================
# VALIDASI DATASET & PERSIAPAN SERIES
# ============================================================
df, date_column, commodity_column = require_dataset()

df[date_column] = pd.to_datetime(df[date_column], errors="coerce", dayfirst=True)
df[commodity_column] = clean_commodity_series(df, commodity_column)
df = df.dropna(subset=[date_column, commodity_column]).sort_values(date_column)

harga = df.set_index(date_column)[commodity_column].astype(float).sort_index()
harga = harga[~harga.index.duplicated(keep="last")]

# ============================================================
# DATASET SUMMARY
# ============================================================
with st.container(border=True):
    st.markdown("#### 📄 Dataset Aktif")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Komoditas", commodity_column)
    c2.metric("Observasi", f"{len(harga):,}".replace(",", "."))
    c3.metric("Tanggal Mulai", harga.index.min().strftime("%d %b %Y"))
    c4.metric("Tanggal Akhir", harga.index.max().strftime("%d %b %Y"))

st.write("")

# ============================================================
# GLOBAL CONFIG
# ============================================================
RANDOM_STATE = 42
END_DATE = harga.index.max()
MIN_LAG, MAX_LAG = 1, 30


def make_time_series_features(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """
    Semua fitur hanya memakai informasi sebelum waktu t.
    Rolling statistics menggunakan shift(1) untuk mencegah data leakage.
    """
    s = series.astype(float).copy()
    frame = pd.DataFrame(index=s.index)
    frame["target"] = s

    for lag in range(1, max_lag + 1):
        frame[f"lag_{lag}"] = s.shift(lag)

    shifted = s.shift(1)
    for window in [3, 5, 7, 10, 14, 21, 30]:
        frame[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window).std()
        frame[f"roll_min_{window}"] = shifted.rolling(window).min()
        frame[f"roll_max_{window}"] = shifted.rolling(window).max()

    frame["ewm_mean_5"] = shifted.ewm(span=5, adjust=False).mean()
    frame["ewm_mean_14"] = shifted.ewm(span=14, adjust=False).mean()
    frame["diff_1"] = s.shift(1) - s.shift(2)
    frame["diff_5"] = s.shift(1) - s.shift(6)

    idx = pd.DatetimeIndex(frame.index)
    frame["day_of_week"] = idx.dayofweek
    frame["day_of_month"] = idx.day
    frame["month"] = idx.month
    frame["quarter"] = idx.quarter
    frame["day_of_year_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def make_svr_estimator(C=100.0, gamma="scale", epsilon=0.05):
    x_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon, cache_size=1000)),
        ]
    )
    return TransformedTargetRegressor(regressor=x_pipeline, transformer=StandardScaler())


def valid_tscv(n_samples, requested_splits):
    n_splits = min(requested_splits, max(2, n_samples // 60))
    n_splits = min(n_splits, n_samples - 1)
    return TimeSeriesSplit(n_splits=n_splits)


def infer_seasonal_period(index):
    index = pd.DatetimeIndex(index)
    weekday_ratio = np.mean(index.dayofweek < 5)
    weekend_ratio = np.mean(index.dayofweek >= 5)
    if weekday_ratio > 0.90 and weekend_ratio < 0.10:
        return 5
    return 7


def adf_test(series):
    result = adfuller(pd.Series(series).dropna(), autolag="AIC")
    return {
        "Statistik ADF": result[0],
        "ADF p-value": result[1],
        "ADF lag": result[2],
        "Kesimpulan ADF": "Stasioner" if result[1] < 0.05 else "Belum stasioner",
    }


def kpss_test(series):
    result = kpss(pd.Series(series).dropna(), regression="c", nlags="auto")
    return {
        "Statistik KPSS": result[0],
        "KPSS p-value": result[1],
        "KPSS lag": result[2],
        "Kesimpulan KPSS": "Stasioner" if result[1] > 0.05 else "Belum stasioner",
    }


def sarima_grid_search(series, seasonal_period, d_value, profile="balanced"):
    """
    Pemilihan parameter SARIMA berdasarkan AIC pada data train.
    Kandidat yang gagal konvergen dilewati.
    """
    y = np.asarray(series, dtype=float)

    if profile == "thorough":
        p_values = q_values = [0, 1, 2, 3]
        seasonal_values = [0, 1, 2]
        max_complexity = 7
    else:
        p_values = q_values = [0, 1, 2]
        seasonal_values = [0, 1]
        max_complexity = 5

    d_values = list(dict.fromkeys([d_value, 0, 1]))
    D_values = [0, 1]

    candidates = []
    for p, d, q, P, D, Q in product(p_values, d_values, q_values, seasonal_values, D_values, seasonal_values):
        if p + q + P + Q > max_complexity:
            continue
        if p == q == P == Q == 0 and d == D == 0:
            continue
        candidates.append(((p, d, q), (P, D, Q, seasonal_period)))

    records = []
    best_result, best_aic, best_spec = None, np.inf, None

    print(f"Jumlah kandidat SARIMA: {len(candidates)}")

    for i, (order, seasonal_order) in enumerate(candidates, start=1):
        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                trend="c" if order[1] == 0 else "n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=150, method="lbfgs")

            records.append(
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "AIC": result.aic,
                    "BIC": result.bic,
                    "Converged": bool(result.mle_retvals.get("converged", True)),
                }
            )

            if np.isfinite(result.aic) and result.aic < best_aic:
                best_aic, best_result, best_spec = result.aic, result, (order, seasonal_order)

        except Exception as e:
            st.write(order, seasonal_order)
            st.code(str(e))
            continue

        if i % 20 == 0:
            print(f"  Kandidat diproses: {i}/{len(candidates)}")

    if len(records) == 0:
        raise RuntimeError("Tidak ada kandidat SARIMA yang berhasil difit.")

    result_table = pd.DataFrame(records).sort_values("AIC")

    if best_result is None:
        raise RuntimeError("Seluruh kandidat SARIMA gagal. Coba tetapkan SARIMA_SEASONAL_PERIOD=5.")

    return best_spec, best_result, result_table


# ============================================================
# 1. TRAIN / TEST SPLIT
# ============================================================
st.markdown("### 1. 📊 Pembagian Data Train & Test")

DEFAULT_TRAIN_END_DATE = pd.Timestamp("2025-12-31")

with st.container(border=True):
    split_method = st.radio("Metode Pembagian Data", ["Persentase", "Tanggal (Advanced)"], horizontal=True)

    if split_method == "Persentase":
        train_ratio = st.slider("Proporsi Data Train (%)", min_value=50, max_value=95, value=80, step=5)
        test_ratio = 100 - train_ratio

        split_index = int(len(harga) * train_ratio / 100)
        train_series = harga.iloc[:split_index].copy()
        test_series = harga.iloc[split_index:].copy()

        split_note = f"Data dibagi secara kronologis {train_ratio}% train dan {test_ratio}% test."

    else:
        train_end_date = st.date_input(
            "Tanggal Akhir Data Train",
            value=min(DEFAULT_TRAIN_END_DATE.date(), harga.index.max().date()),
            min_value=harga.index.min().date(),
            max_value=harga.index.max().date(),
        )

        train_series = harga.loc[harga.index <= pd.Timestamp(train_end_date)].copy()
        test_series = harga.loc[harga.index > pd.Timestamp(train_end_date)].copy()

        total = len(harga)
        train_ratio = round(len(train_series) / total * 100, 1)
        test_ratio = round(len(test_series) / total * 100, 1)
        split_note = "Data dibagi berdasarkan tanggal yang dipilih."

# -------- validasi --------
if len(train_series) == 0 or len(test_series) == 0:
    st.error("Pembagian data menghasilkan train atau test kosong. Silakan ubah rasio atau tanggal.")
    st.stop()

effective_train_end = train_series.index.max()
effective_test_start = test_series.index.min()

# -------- ringkasan --------
st.info(split_note)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Train", f"{len(train_series):,}".replace(",", "."))
c2.metric("Test", f"{len(test_series):,}".replace(",", "."))
c3.metric("Train Sampai", effective_train_end.strftime("%d %b %Y"))
c4.metric("Test Mulai", effective_test_start.strftime("%d %b %Y"))

# -------- visualisasi --------
with st.expander("📈 Visualisasi Pembagian Data", expanded=False):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(train_series.index, train_series, label="Train", linewidth=1.3)
    ax.plot(test_series.index, test_series, label="Test", linewidth=1.3)
    ax.axvline(effective_test_start, color="red", linestyle="--", linewidth=1.3)
    ax.set_title("Pembagian Data Train dan Test")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel(f"Harga {commodity_column}")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

st.write("")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
st.markdown("### 2. ⚙️ Feature Engineering")

with st.container(border=True):
    st.markdown(
        """
        Feature engineering dilakukan secara otomatis sebelum proses
        pelatihan model.

        Lag optimum akan dipilih menggunakan **Support Vector Regression
        (SVR) Baseline** dengan evaluasi **TimeSeriesSplit Cross Validation
        (CV RMSE)**.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Lag Selection")
        st.success(
            """
            **Automatic**

            Lag optimum akan dipilih secara otomatis
            setelah proses pelatihan dimulai.
            """
        )
        st.info(
            """
            Metode :

            • SVR Baseline

            • Cross Validation RMSE
            """
        )

    with col2:
        st.markdown("#### Cross Validation")
        cv_splits = st.slider(
            "TimeSeriesSplit (Fold)", min_value=3, max_value=10, value=5, step=1, key="cv_split"
        )

        effective_cv = min(cv_splits, max(2, len(train_series) // 60))
        effective_cv = min(effective_cv, len(train_series) - 1)

        st.metric("Effective Fold", effective_cv)
        st.caption("Jumlah fold akan disesuaikan apabila jumlah data tidak mencukupi.")

# ============================================================
# 3. SARIMA CONFIGURATION
# ============================================================
st.markdown("### 3. 📈 Konfigurasi SARIMA")

recommended_seasonal = infer_seasonal_period(train_series.index)
adf_result = adfuller(train_series.dropna(), autolag="AIC")
recommended_d = 0 if adf_result[1] < 0.05 else 1

col1, col2 = st.columns(2)

# -------- seasonal period --------
with col1:
    with st.container(border=True):
        st.markdown("#### 📅 Seasonal Period")
        st.metric("Suggested Seasonal Period", recommended_seasonal)

        auto_season = st.toggle("Gunakan nilai otomatis", value=True, key="auto_seasonal_period")

        if auto_season:
            seasonal_period = recommended_seasonal
            st.number_input("Seasonal Period", value=seasonal_period, disabled=True)
        else:
            seasonal_period = st.number_input(
                "Seasonal Period", min_value=2, max_value=365, value=recommended_seasonal, step=1
            )

        st.caption("Periodisitas musiman yang digunakan pada proses pencarian parameter SARIMA.")

# -------- differencing --------
with col2:
    with st.container(border=True):
        st.markdown("#### 🔄 Differencing")
        st.metric("ADF p-value", f"{adf_result[1]:.4f}")
        st.write("**Interpretasi:** " + ("Data sudah stasioner." if recommended_d == 0 else "Data belum stasioner."))

        auto_d = st.toggle("Gunakan nilai otomatis", value=True, key="auto_differencing")

        if auto_d:
            d_value = recommended_d
            st.number_input("Differencing (d)", value=d_value, disabled=True)
        else:
            d_value = st.number_input("Differencing (d)", min_value=0, max_value=2, value=recommended_d, step=1)

        st.caption("Nilai differencing yang digunakan dalam proses grid search SARIMA.")

# ============================================================
# 4. MACHINE LEARNING CONFIGURATION
# ============================================================
st.markdown("### 4. 🤖 Machine Learning")

SVR_ITERATIONS_BY_PROFILE = {"Fast": 15, "Balanced": 28, "Thorough": 50}
RF_ITERATIONS_BY_PROFILE = {"Fast": 15, "Balanced": 24, "Thorough": 40}

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### Support Vector Regression (SVR)")
        svr_profile = st.radio(
            "Search Profile", ["Fast", "Balanced", "Thorough"], index=1, horizontal=True, key="svr_profile"
        )
        svr_iterations = SVR_ITERATIONS_BY_PROFILE[svr_profile]

        st.metric("Random Search Iteration", svr_iterations)
        st.caption("Jumlah iterasi RandomizedSearchCV yang digunakan untuk proses tuning SVR.")

with col2:
    with st.container(border=True):
        st.markdown("#### Random Forest")
        rf_profile = st.radio(
            "Search Profile", ["Fast", "Balanced", "Thorough"], index=1, horizontal=True, key="rf_profile"
        )
        rf_iterations = RF_ITERATIONS_BY_PROFILE[rf_profile]

        st.metric("Random Search Iteration", rf_iterations)
        st.caption("Jumlah iterasi RandomizedSearchCV yang digunakan untuk proses tuning Random Forest.")

# ============================================================
# RUN MODEL
# ============================================================
jalankan = st.button("▶ Jalankan Model", type="primary", use_container_width=True)

if jalankan:
    params = {
        "commodity_column": commodity_column,
        "date_column": date_column,
        "train_end": effective_train_end,
        "test_start": effective_test_start,
        "train_size": len(train_series),
        "test_size": len(test_series),
        "max_lag": MAX_LAG,
        "svr_iterations": svr_iterations,
        "rf_iterations": rf_iterations,
        "seasonal_period": seasonal_period,
        "d": d_value,
        "svr_profile": svr_profile,
        "rf_profile": rf_profile,
        "cv_splits": effective_cv,
        "random_state": RANDOM_STATE,
    }
    st.session_state.model_params = params

    with st.spinner("Mempersiapkan data dan menjalankan proses tuning model..."):

        # -------- SVR lag optimization --------
        lag_results = []
        max_candidate = min(MAX_LAG, max(MIN_LAG, len(train_series) // 10))

        for lag in range(MIN_LAG, max_candidate + 1):
            feature_data = make_time_series_features(train_series, max_lag=lag)
            if len(feature_data) < 80:
                continue

            X_lag = feature_data.drop(columns="target")
            y_lag = feature_data["target"]
            cv = valid_tscv(len(X_lag), cv_splits)

            scores = cross_val_score(
                make_svr_estimator(), X_lag, y_lag, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
            )

            lag_results.append(
                {
                    "Lag": lag,
                    "CV RMSE": -scores.mean(),
                    "CV RMSE Std": scores.std(),
                    "Jumlah fitur": X_lag.shape[1],
                    "Jumlah observasi": len(X_lag),
                }
            )

        lag_table = pd.DataFrame(lag_results).sort_values("CV RMSE")

        if lag_table.empty:
            st.error("Penentuan lag gagal karena data terlalu sedikit.")
            st.stop()

        optimal_lag = int(lag_table.iloc[0]["Lag"])
        best_cv_rmse = float(lag_table.iloc[0]["CV RMSE"])

        fig, ax = plt.subplots(figsize=(9, 4))
        lag_plot = lag_table.sort_values("Lag")
        ax.plot(lag_plot["Lag"], lag_plot["CV RMSE"], marker="o")
        ax.axvline(optimal_lag, linestyle="--", color="red", label=f"Optimal Lag = {optimal_lag}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("CV RMSE")
        ax.set_title("Lag Optimization using TimeSeriesSplit")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.close(fig)

        # -------- stationarity --------
        stationarity_results = pd.DataFrame(
            [
                {"Transformasi": "Level", **adf_test(train_series), **kpss_test(train_series)},
                {
                    "Transformasi": "First difference",
                    **adf_test(train_series.diff()),
                    **kpss_test(train_series.diff()),
                },
            ]
        )

        # -------- supervised data --------
        supervised = make_time_series_features(harga, max_lag=optimal_lag)
        X_all = supervised.drop(columns="target")
        y_all = supervised["target"]

        train_mask = X_all.index <= effective_train_end
        test_mask = (X_all.index >= effective_test_start) & (X_all.index <= END_DATE)

        X_train = X_all.loc[train_mask].copy()
        y_train = y_all.loc[train_mask].copy()
        X_test = X_all.loc[test_mask].copy()
        y_test = y_all.loc[test_mask].copy()

        if len(X_train) < 100 or len(X_test) < 10:
            st.error(f"Data supervised tidak cukup. Train={len(X_train)}, Test={len(X_test)}")
            st.stop()

    # ============================================================
    # 1. SARIMA GRID SEARCH
    # ============================================================
    st.markdown("### 🔎 Mencari Parameter SARIMA Terbaik")

    with st.spinner("Melakukan grid search SARIMA..."):
        start = time.time()
        best_sarima_spec, fitted_sarima_train, sarima_search_table = sarima_grid_search(
            train_series, seasonal_period=seasonal_period, d_value=d_value, profile="balanced"
        )
        best_sarima_order, best_sarima_seasonal_order = best_sarima_spec
        elapsed = (time.time() - start) / 60

    st.success(f"SARIMA selesai dalam {elapsed:.2f} menit.")

    c1, c2 = st.columns(2)
    c1.metric("Best Order", str(best_sarima_order))
    c2.metric("Seasonal Order", str(best_sarima_seasonal_order))

    with st.expander("Seluruh kandidat SARIMA"):
        st.dataframe(sarima_search_table, use_container_width=True, hide_index=True)

    # ============================================================
    # SVR RANDOM SEARCH
    # ============================================================
    st.markdown("### 🤖 Tuning Support Vector Regression")

    svr_search_space = {
        "regressor__svr__C": loguniform(1e-1, 2e3),
        "regressor__svr__gamma": loguniform(1e-5, 1.0),
        "regressor__svr__epsilon": loguniform(1e-3, 0.5),
    }

    svr_search = RandomizedSearchCV(
        estimator=make_svr_estimator(),
        param_distributions=svr_search_space,
        n_iter=svr_iterations,
        scoring="neg_root_mean_squared_error",
        cv=valid_tscv(len(X_train), effective_cv),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    with st.spinner("Melakukan tuning SVR..."):
        start = time.time()
        svr_search.fit(X_train, y_train)
        elapsed = (time.time() - start) / 60

    best_svr = svr_search.best_estimator_
    st.success(f"Tuning SVR selesai ({elapsed:.2f} menit)")

    col1, col2 = st.columns(2)
    col1.metric("Best CV RMSE", f"{-svr_search.best_score_:.3f}")
    col2.metric("Jumlah Iterasi", svr_iterations)

    with st.expander("Best Parameter SVR"):
        st.json(svr_search.best_params_)

    # ============================================================
    # RANDOM FOREST RANDOM SEARCH
    # ============================================================
    st.markdown("### 🌲 Tuning Random Forest")

    rf_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    rf_search_space = {
        "n_estimators": randint(300, 1001),
        "max_depth": [None, 5, 8, 12, 16, 24, 32],
        "min_samples_split": randint(2, 16),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", 0.5, 0.75, 1.0],
        "bootstrap": [True],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_search_space,
        n_iter=rf_iterations,
        scoring="neg_root_mean_squared_error",
        cv=valid_tscv(len(X_train), effective_cv),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    with st.spinner("Melakukan tuning Random Forest..."):
        start = time.time()
        rf_search.fit(X_train, y_train)
        elapsed = (time.time() - start) / 60

    best_rf = rf_search.best_estimator_
    st.success(f"Tuning Random Forest selesai ({elapsed:.2f} menit)")

    col1, col2 = st.columns(2)
    col1.metric("Best CV RMSE", f"{-rf_search.best_score_:.3f}")
    col2.metric("Jumlah Iterasi", rf_iterations)

    with st.expander("Best Parameter Random Forest"):
        st.json(rf_search.best_params_)

    # -------- simpan hasil model ke session state --------
    st.session_state.model_result = {
        "best_svr": best_svr,
        "best_rf": best_rf,
        "best_sarima_order": best_sarima_order,
        "best_sarima_seasonal_order": best_sarima_seasonal_order,
        "fitted_sarima_train": fitted_sarima_train,
        "sarima_search_table": sarima_search_table,
        "svr_best_params": svr_search.best_params_,
        "svr_best_score": -svr_search.best_score_,
        "rf_best_params": rf_search.best_params_,
        "rf_best_score": -rf_search.best_score_,
    }

    # ============================================================
    # RESULT SUMMARY
    # ============================================================
    st.markdown("### 🔎 Hasil Konfigurasi Otomatis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimal Lag SVR", optimal_lag)
    c2.metric("SARIMA Seasonal Period", seasonal_period)
    c3.metric("Train", f"{len(X_train):,}".replace(",", "."))
    c4.metric("Test", f"{len(X_test):,}".replace(",", "."))

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.pyplot(fig)

    with st.expander("Lihat hasil uji stasioneritas"):
        st.dataframe(stationarity_results, use_container_width=True, hide_index=True)

    with st.expander("Lihat kandidat lag SVR"):
        st.dataframe(lag_table, use_container_width=True, hide_index=True)

    # -------- simpan hasil preprocessing --------
    st.session_state.model_data = {
        "harga": harga,
        "train_series": train_series,
        "test_series": test_series,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "optimal_lag": optimal_lag,
        "stationarity_results": stationarity_results,
        "seasonal_period": seasonal_period,
        "effective_train_end": effective_train_end,
        "effective_test_start": effective_test_start,
        "best_cv_rmse": best_cv_rmse,
        "best_svr": best_svr,
        "best_rf": best_rf,
        "best_sarima_order": best_sarima_order,
        "best_sarima_seasonal_order": best_sarima_seasonal_order,
    }
