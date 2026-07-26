import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import (
    evaluate_prediction,
    format_rupiah,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_trained_model,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Output Model — KomoditasAI", page_icon="📈", layout="wide")

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / output",
    title="📈 Hasil Forecasting",
    caption="Pelatihan model, evaluasi performa, stacking ensemble, serta forecasting harga komoditas.",
)


# ============================================================
# FUNGSI SARIMA ONE-STEP-AHEAD
# ============================================================
def fit_sarima_array(values, order, seasonal_order):
    values = np.asarray(values, dtype=float)
    return SARIMAX(
        values,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 else "n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=150, method="lbfgs")


def append_observation(result, actual):
    try:
        return result.append(np.asarray([actual], dtype=float), refit=False)
    except Exception:
        return None


def sarima_one_step_predictions(history_values, actual_future_values, order, seasonal_order, refit_every=0):
    history = list(np.asarray(history_values, dtype=float))
    future = np.asarray(actual_future_values, dtype=float)
    predictions = []

    result = fit_sarima_array(history, order, seasonal_order)

    for step, actual in enumerate(future, start=1):
        pred = float(result.forecast(steps=1)[0])
        predictions.append(pred)
        history.append(float(actual))

        appended = append_observation(result, actual)
        need_refit = appended is None or (refit_every and step % refit_every == 0)

        result = fit_sarima_array(history, order, seasonal_order) if need_refit else appended

    return np.asarray(predictions)


# ============================================================
# VALIDASI & MUAT HASIL TRAINING
# ============================================================
trained, data, params = require_trained_model()

best_svr = trained["best_svr"]
best_rf = trained["best_rf"]
best_sarima_order = trained["best_sarima_order"]
best_sarima_seasonal_order = trained["best_sarima_seasonal_order"]

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
train_series, test_series = data["train_series"], data["test_series"]
harga = data["harga"]

cv_splits = params["cv_splits"]
RANDOM_STATE = params["random_state"]

# ============================================================
# 1. FIT BASE MODELS
# ============================================================
st.markdown("### 1. 🔄 Membangun Model Terbaik")

with st.spinner("Melatih model terbaik..."):
    best_svr.fit(X_train, y_train)
    best_rf.fit(X_train, y_train)

st.success("Seluruh base learner berhasil dibangun.")

# ============================================================
# 2. PREDIKSI BASE LEARNERS
# ============================================================
st.markdown("### 2. 📈 Prediksi Base Learners")

sarima_test_pred = pd.Series(
    sarima_one_step_predictions(
        history_values=train_series.values,
        actual_future_values=test_series.values,
        order=best_sarima_order,
        seasonal_order=best_sarima_seasonal_order,
    ),
    index=test_series.index,
)

svr_train_pred = pd.Series(best_svr.predict(X_train), index=X_train.index)
svr_test_pred = pd.Series(best_svr.predict(X_test), index=X_test.index)

rf_train_pred = pd.Series(best_rf.predict(X_train), index=X_train.index)
rf_test_pred = pd.Series(best_rf.predict(X_test), index=X_test.index)

st.success("Prediksi seluruh base learner selesai.")

# ============================================================
# 3. HASIL PREDIKSI BASE LEARNERS
# ============================================================
st.markdown("### 3. 📋 Hasil Prediksi Base Learners")

prediction_df = pd.DataFrame(
    {
        "Actual": test_series,
        "SARIMA": sarima_test_pred,
        "SVR": svr_test_pred,
        "Random Forest": rf_test_pred,
    }
)
st.dataframe(prediction_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test_series.index, test_series, linewidth=2, label="Actual")
ax.plot(sarima_test_pred.index, sarima_test_pred, label="SARIMA")
ax.plot(svr_test_pred.index, svr_test_pred, label="SVR")
ax.plot(rf_test_pred.index, rf_test_pred, label="Random Forest")
ax.set_title("Perbandingan Prediksi Base Learners")
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig)
plt.close(fig)

# ============================================================
# 4. EVALUASI BASE LEARNERS
# ============================================================
st.markdown("### 4. 📊 Evaluasi Base Learners")

sarima_metric = evaluate_prediction(test_series, sarima_test_pred, train_series, "SARIMA")
svr_metric = evaluate_prediction(y_test, svr_test_pred, y_train, "SVR")
rf_metric = evaluate_prediction(y_test, rf_test_pred, y_train, "Random Forest")

metric_table = pd.DataFrame([sarima_metric, svr_metric, rf_metric])

METRIC_FORMAT = {
    "RMSE": "{:.2f}",
    "MAE": "{:.2f}",
    "MAPE (%)": "{:.2f}",
    "sMAPE (%)": "{:.2f}",
    "MASE": "{:.3f}",
    "R2": "{:.4f}",
    "Bias": "{:.2f}",
}

st.dataframe(metric_table.style.format(METRIC_FORMAT), use_container_width=True, hide_index=True)

best_model = metric_table.sort_values("RMSE").iloc[0]
st.success(f"Model dengan RMSE terbaik adalah **{best_model['Model']}** dengan RMSE sebesar **{best_model['RMSE']:.2f}**.")

# ============================================================
# 5. OUT OF FOLD PREDICTION
# ============================================================
st.markdown("### 5. 🧩 Membangun Meta Feature (OOF)")

tscv = TimeSeriesSplit(n_splits=cv_splits)

oof_svr = pd.Series(index=X_train.index, dtype=float)
oof_rf = pd.Series(index=X_train.index, dtype=float)
oof_sarima = pd.Series(index=train_series.index, dtype=float)

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # -------- SVR --------
    svr_fold = clone(best_svr)
    svr_fold.fit(X_tr, y_tr)
    oof_svr.iloc[val_idx] = svr_fold.predict(X_val)

    # -------- Random Forest --------
    rf_fold = clone(best_rf)
    rf_fold.fit(X_tr, y_tr)
    oof_rf.iloc[val_idx] = rf_fold.predict(X_val)

    # -------- SARIMA --------
    sarima_train = train_series.loc[train_series.index <= y_tr.index.max()]

    validation_start = y_val.index.min()
    first_val_position = train_series.index.get_loc(validation_start)

    history_values = train_series.iloc[:first_val_position].values
    future_values = train_series.loc[y_val.index].values

    oof_sarima.loc[y_val.index] = sarima_one_step_predictions(
        history_values=history_values,
        actual_future_values=future_values,
        order=best_sarima_order,
        seasonal_order=best_sarima_seasonal_order,
    )

oof_table = pd.DataFrame({"SARIMA": oof_sarima, "SVR": oof_svr, "RF": oof_rf, "Target": y_train}).dropna()

st.success(f"OOF berhasil dibuat ({len(oof_table)} observasi)")
st.dataframe(oof_table.head(), use_container_width=True)

# ============================================================
# META LEARNER
# ============================================================
meta_learner = LinearRegression()
meta_learner.fit(oof_table[["SARIMA", "SVR", "RF"]], oof_table["Target"])

meta_test = pd.DataFrame(
    {"SARIMA": sarima_test_pred, "SVR": svr_test_pred, "RF": rf_test_pred}
).loc[y_test.index]

ensemble_pred = pd.Series(meta_learner.predict(meta_test), index=y_test.index)

coef_table = pd.DataFrame(
    {
        "Base Learner": ["Support Vector Regression", "Random Forest", "SARIMA"],
        "Koefisien": meta_learner.coef_,
    }
)
coef_table["Koefisien"] = coef_table["Koefisien"].round(4)

st.markdown("#### Bobot Meta Learner")
st.dataframe(coef_table, use_container_width=True, hide_index=True)
st.metric("Intercept", round(meta_learner.intercept_, 4))

# ============================================================
# 6. STACKING ENSEMBLE PREDICTION
# ============================================================
st.markdown("### 6. 🚀 Prediksi Stacking Ensemble")

prediction_result = pd.DataFrame(
    {
        "Actual": y_test,
        "SARIMA": sarima_test_pred,
        "SVR": svr_test_pred,
        "Random Forest": rf_test_pred,
        "Stacking Ensemble": ensemble_pred,
    }
)
st.dataframe(prediction_result, use_container_width=True)

# ============================================================
# VISUALISASI STACKING ENSEMBLE
# ============================================================
st.markdown("### 📈 Hasil Prediksi Stacking Ensemble")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(prediction_result.index, prediction_result["Actual"], linewidth=2.5, label="Actual")
ax.plot(prediction_result.index, prediction_result["Stacking Ensemble"], linewidth=2.5, label="Stacking Ensemble")
ax.set_title("Perbandingan Aktual dan Prediksi Stacking Ensemble")
ax.set_xlabel("Periode")
ax.set_ylabel("Harga")
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)

stacking_metric = evaluate_prediction(y_test, ensemble_pred, y_train, "Stacking Ensemble")
metric_table = pd.concat([metric_table, pd.DataFrame([stacking_metric])], ignore_index=True)

st.markdown("### 7. 📊 Perbandingan Performa Model")
st.dataframe(metric_table.style.format(METRIC_FORMAT), use_container_width=True, hide_index=True)

# ============================================================
# 8. ANALISIS RISIKO (VALUE AT RISK)
# ============================================================
st.markdown("### 8. ⚠️ Analisis Risiko (Value at Risk)")

meta_oof_pred = pd.Series(
    meta_learner.predict(oof_table[["SARIMA", "SVR", "RF"]]),
    index=oof_table.index,
)
abs_error = (oof_table["Target"] - meta_oof_pred).abs()

var90 = abs_error.quantile(0.90)
var95 = abs_error.quantile(0.95)
var99 = abs_error.quantile(0.99)

var_table = pd.DataFrame({"Hari": range(1, 6)})
var_table["VaR 90%"] = var90 * np.sqrt(var_table["Hari"])
var_table["VaR 95%"] = var95 * np.sqrt(var_table["Hari"])
var_table["VaR 99%"] = var99 * np.sqrt(var_table["Hari"])

st.dataframe(
    var_table.style.format({"VaR 90%": "Rp{:,.2f}", "VaR 95%": "Rp{:,.2f}", "VaR 99%": "Rp{:,.2f}"}),
    use_container_width=True,
    hide_index=True,
)

fig, ax = plt.subplots(figsize=(10, 5))
for col in ["VaR 90%", "VaR 95%", "VaR 99%"]:
    ax.plot(var_table["Hari"], var_table[col], marker="o", linewidth=2, label=col)
ax.set_title("Value at Risk 1–5 Hari")
ax.set_xlabel("Horizon Risiko (Hari)")
ax.set_ylabel("Nilai VaR (Rp)")
ax.set_xticks(var_table["Hari"])
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.info(
    f"""
Pada tingkat kepercayaan **95%**, estimasi risiko maksimum
kesalahan prediksi model stacking selama **1 hari**
adalah sekitar **{format_rupiah(var95)}**.

Apabila horizon prediksi diperpanjang menjadi **5 hari**,
risiko meningkat menjadi sekitar
**{format_rupiah(var_table.iloc[-1]['VaR 95%'])}**
dengan pendekatan *square-root-of-time*.
"""
)
