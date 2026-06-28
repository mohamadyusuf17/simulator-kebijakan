import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# ── LANGKAH 1: Siapkan model & baseline ──────────────────────
X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
y_train = np.array([50, 80, 110, 90, 150])

model = LinearRegression().fit(X_train, y_train)

baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]

# ── LANGKAH 2: Fungsi simulasi what-if ───────────────────────
def run_simulation(new_iklan, new_diskon):
    intervention_input = np.array([[new_iklan, new_diskon]])
    prediction = model.predict(intervention_input)[0]
    delta_y = prediction - baseline_pred
    return prediction, delta_y

# ── LANGKAH 3: UI Streamlit ──────────────────────────────────
st.title("🚀 Simulator Kebijakan Keuntungan Toko")
st.write("Gunakan slider untuk menguji skenario **What-If**.")

st.sidebar.header("Tuas Kebijakan (Intervensi)")
iklan_slider  = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 50, 10)
diskon_slider = st.sidebar.slider("Besaran Diskon (%)",    0, 50, 10)

hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

col1, col2 = st.columns(2)
col1.metric("Prediksi Keuntungan", f"Rp {hasil_pred:.2f} Jt", f"{delta:+.2f} Jt")
col2.metric("Baseline Saat Ini",   f"Rp {baseline_pred:.2f} Jt")

data_plot = pd.DataFrame({
    'Skenario'   : ['Baseline', 'Intervensi'],
    'Keuntungan' : [baseline_pred, hasil_pred]
})
st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan')

if delta > 0:
    st.success(f"Skenario ini meningkatkan keuntungan sebesar Rp {delta:.2f} Juta.")
elif delta < 0:
    st.warning(f"Skenario ini MENURUNKAN keuntungan sebesar Rp {abs(delta):.2f} Juta!")
else:
    st.info("Tidak ada perubahan dari kondisi baseline.")
    
    # run = python -m streamlit run app.py