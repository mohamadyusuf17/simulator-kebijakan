import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

st.set_page_config(page_title="SimKebijakan", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
:root {
    --bg:#060D1F;--card:#0E1E38;--deep:#0A1628;
    --border:rgba(212,175,55,0.18);--gold:#D4AF37;--gold2:#E8CC6A;
    --slate:#8892B0;--white:#EEF0F7;--green:#22C55E;--red:#EF4444;--blue:#38BDF8;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg);}
section[data-testid="stSidebar"]{background:var(--deep)!important;border-right:1px solid var(--border);}
.hdr{display:flex;justify-content:space-between;align-items:flex-end;padding-bottom:1rem;border-bottom:1px solid var(--border);margin-bottom:1.4rem;}
.badge-tag{display:inline-block;background:rgba(212,175,55,.1);border:1px solid rgba(212,175,55,.3);border-radius:4px;padding:.12rem .6rem;font-size:.6rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin-bottom:.4rem;}
.ttl{font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:700;color:var(--white);line-height:1;margin:0;}
.ttl span{color:var(--gold);}
.sub{font-size:.74rem;color:var(--slate);margin-top:.35rem;}
.auth{text-align:right;}
.auth-name{font-size:.85rem;font-weight:600;color:var(--gold2);}
.auth-npm{font-size:.68rem;color:var(--slate);margin-top:.1rem;}
.sec{font-family:'Space Grotesk',sans-serif;font-size:.65rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--gold);display:flex;align-items:center;gap:.5rem;margin-bottom:.9rem;}
.sec::after{content:'';flex:1;height:1px;background:var(--border);}
.mc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.85rem 1.1rem;position:relative;overflow:hidden;display:flex;align-items:center;gap:1rem;margin-bottom:.7rem;}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--gold),transparent);}
.mc-icon{font-size:1.4rem;flex-shrink:0;}
.mc-body{flex:1;min-width:0;}
.mc-lbl{font-size:.58rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--slate);}
.mc-val{font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:var(--gold);line-height:1.1;}
.mc-val.green{color:var(--green);}
.mc-val.red{color:var(--red);}
.mc-sub{font-size:.62rem;color:var(--slate);margin-top:.15rem;}
.sc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.85rem 1.1rem;position:relative;overflow:hidden;}
.sc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--gold),transparent);}
.pill{display:inline-flex;align-items:center;gap:.3rem;padding:.22rem .8rem;border-radius:999px;font-size:.75rem;font-weight:700;margin-bottom:.35rem;}
.pill.g{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);color:var(--green);}
.pill.r{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:var(--red);}
.pill.y{background:rgba(212,175,55,.1);border:1px solid rgba(212,175,55,.3);color:var(--gold);}
.sc-txt{font-size:.68rem;color:var(--slate);line-height:1.5;}
.bl{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.8rem 1rem;font-size:.73rem;color:var(--slate);line-height:1.9;}
.bl b{color:var(--gold2);}
.sb-lbl{font-size:.6rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--gold);margin-bottom:.35rem;}
.divider{border:none;border-top:1px solid var(--border);margin:1.2rem 0;}
.footer{text-align:center;font-size:.67rem;color:var(--slate);border-top:1px solid var(--border);padding:.9rem 0 .4rem;margin-top:.8rem;letter-spacing:.4px;}

/* ── Ranking Table (M16 SAW) ── */
.rank-table{width:100%;border-collapse:collapse;font-size:.73rem;}
.rank-table th{color:var(--gold);font-size:.58rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;padding:.55rem .8rem;border-bottom:1px solid var(--border);text-align:left;}
.rank-table td{padding:.55rem .8rem;color:var(--white);border-bottom:1px solid rgba(212,175,55,.07);}
.rank-table tr.best td{background:rgba(212,175,55,.06);}
.rank-badge{display:inline-block;width:1.4rem;height:1.4rem;border-radius:50%;background:var(--gold);color:#060D1F;font-size:.65rem;font-weight:700;text-align:center;line-height:1.4rem;}
.rank-badge.s2{background:rgba(212,175,55,.3);color:var(--gold2);}
.rank-badge.s3{background:rgba(136,146,176,.2);color:var(--slate);}

/* ── XAI Bar ── */
.xai-row{display:flex;align-items:center;gap:.7rem;margin-bottom:.55rem;}
.xai-lbl{font-size:.68rem;color:var(--slate);width:7rem;flex-shrink:0;text-align:right;}
.xai-bar-wrap{flex:1;background:rgba(136,146,176,.12);border-radius:4px;height:.55rem;overflow:hidden;}
.xai-bar-inner{height:100%;border-radius:4px;transition:width .4s;}
.xai-val{font-size:.68rem;font-weight:700;width:3rem;text-align:right;}
.xai-pos{color:var(--green);}
.xai-neg{color:var(--red);}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ① PERSISTENSI MODEL (M15)
# ─────────────────────────────────────────────
MODEL_PATH  = "model_kebijakan_v1.joblib"
SCALER_PATH = "scaler_kebijakan_v1.joblib"

@st.cache_resource
def load_or_train_model():
    X_train = np.array([[5,10],[10,20],[15,5],[20,25],[25,15]])
    y_train = np.array([50,80,110,90,150])
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model   = LinearRegression().fit(X_scaled, y_train)
    if not os.path.exists(MODEL_PATH):
        joblib.dump(model, MODEL_PATH)
    if not os.path.exists(SCALER_PATH):
        joblib.dump(scaler, SCALER_PATH)
    return model, scaler, X_train

model, scaler, X_train_ref = load_or_train_model()

B_IKLAN = 10; B_DISKON = 10
baseline_raw = np.array([[B_IKLAN, B_DISKON]])
baseline     = model.predict(scaler.transform(baseline_raw))[0]
ic = round(model.coef_[0], 2)
dc = round(model.coef_[1], 2)

DARK="#0E1E38"; DARKER="#0A1628"; GOLD="#D4AF37"

# ─────────────────────────────────────────────
# ② DETEKSI DRIFT (M15)
# ─────────────────────────────────────────────
def check_data_drift(new_data, train_data, threshold=15.0):
    drift = np.abs(np.mean(new_data) - np.mean(train_data))
    return drift > threshold, round(drift, 2)

# ─────────────────────────────────────────────
# ③ ANONYMIZATION (M15)
# ─────────────────────────────────────────────
def clean_sensitive_data(df_input):
    cols_pii = ['Nama_Operator', 'NIK_Petugas', 'Email', 'No_HP']
    return df_input.drop(columns=[c for c in cols_pii if c in df_input.columns])

# ─────────────────────────────────────────────
# ④ SAW — MCDM (M16 NEW)
# ─────────────────────────────────────────────
# Bobot AHP sudah disesuaikan dengan koefisien ML:
#   Iklan paling berpengaruh → bobot tertinggi
SAW_WEIGHTS = np.array([0.50, 0.30, 0.20])   # [Profit-ML, Efisiensi, Biaya]
SAW_TYPES   = ["benefit", "benefit", "cost"]  # cost = semakin kecil semakin baik

ALTERNATIF = ["Strategi A\n(Iklan Agresif)", "Strategi B\n(Diskon Sedang)", "Strategi C\n(Konservatif)"]

def hitung_saw(pred_profit, iklan_val, diskon_val):
    """
    Matriks keputusan 3 alternatif × 3 kriteria:
      - Kolom 0 (Profit-ML)  : output prediksi ML per skenario
      - Kolom 1 (Efisiensi)  : asumsi statis per strategi
      - Kolom 2 (Biaya Total): total pengeluaran per skenario
    """
    # Prediksi profit masing-masing alternatif
    p_a = model.predict(scaler.transform([[min(iklan_val*1.4, 50), diskon_val]]))[0]
    p_b = pred_profit   # skenario aktif = Alternatif B (tengah)
    p_c = model.predict(scaler.transform([[max(iklan_val*0.6, 0), diskon_val*0.7]]))[0]

    matriks = np.array([
        [p_a,  85.0, iklan_val*1.4 + diskon_val*0.8],   # A: iklan tinggi, biaya besar
        [p_b,  90.0, iklan_val     + diskon_val      ],   # B: skenario aktif slider
        [p_c,  70.0, iklan_val*0.6 + diskon_val*0.7 ],   # C: konservatif, biaya kecil
    ])

    # Normalisasi SAW
    norm = np.zeros_like(matriks, dtype=float)
    for j in range(matriks.shape[1]):
        if SAW_TYPES[j] == "benefit":
            denom = matriks[:, j].max()
            norm[:, j] = matriks[:, j] / denom if denom != 0 else 0
        else:  # cost
            denom = matriks[:, j].min()
            norm[:, j] = denom / matriks[:, j] if denom != 0 else 0

    skor = norm @ SAW_WEIGHTS
    ranking = np.argsort(skor)[::-1]   # descending
    return matriks, norm, skor, ranking

# ─────────────────────────────────────────────
# ⑤ XAI — KONTRIBUSI FITUR (M16 NEW)
# ─────────────────────────────────────────────
def hitung_xai(iklan_val, diskon_val):
    """
    Pendekatan XAI berbasis koefisien LinearRegression (pengganti SHAP
    untuk model linear): kontribusi = koef × nilai_terstandarisasi.
    Nilai positif = mendorong profit naik. Nilai negatif = menekan profit.
    """
    x_scaled = scaler.transform([[iklan_val, diskon_val]])[0]
    kontribusi = model.coef_ * x_scaled         # shape (2,)
    intercept  = model.intercept_
    labels = ["Anggaran Iklan", "Besaran Diskon"]
    return labels, kontribusi, intercept

# ─────────────────────────────────────────────
# CHART HELPERS (M14/M15 — tidak diubah)
# ─────────────────────────────────────────────
def gauge(val):
    color="#EF4444" if val/200<.4 else GOLD if val/200<.7 else "#22C55E"
    fig=go.Figure(go.Indicator(
        mode="gauge+number+delta",value=val,
        delta={"reference":baseline,"valueformat":".1f",
               "increasing":{"color":"#22C55E"},"decreasing":{"color":"#EF4444"}},
        number={"suffix":" Jt","font":{"size":42,"color":GOLD,"family":"Space Grotesk"}},
        gauge={"axis":{"range":[0,200],"tickfont":{"color":"#8892B0","size":10},"nticks":6},
               "bar":{"color":color,"thickness":.22},"bgcolor":DARK,"bordercolor":"#142444",
               "steps":[{"range":[0,80],"color":"#0A1628"},{"range":[80,140],"color":"#0E1830"},
                        {"range":[140,200],"color":"#101F38"}],
               "threshold":{"line":{"color":GOLD,"width":2},"thickness":.72,"value":baseline}},
        title={"text":"Speedometer Profit<br><span style='font-size:10px;color:#8892B0'>KEUNTUNGAN PREDIKSI</span>",
               "font":{"color":"#EEF0F7","size":13}},
    ))
    fig.update_layout(paper_bgcolor=DARK,plot_bgcolor=DARK,
        margin=dict(t=70,b=10,l=20,r=20),height=300,
        font_color="#EEF0F7",font_family="Inter")
    return fig

def bar_chart(base, pred):
    fig=go.Figure(go.Bar(
        x=["Baseline","Skenario Aktif"],y=[base,pred],
        marker_color=[DARKER,GOLD],
        marker_line_color=["#8892B0","#E8CC6A"],marker_line_width=1.2,
        text=[f"Rp {base:.1f}",f"Rp {pred:.1f}"],
        textposition="outside",textfont_color="#EEF0F7",width=0.38,
    ))
    fig.update_layout(paper_bgcolor=DARK,plot_bgcolor=DARKER,
        font_color="#EEF0F7",font_family="Inter",
        yaxis=dict(gridcolor="rgba(136,146,176,.1)",title="Keuntungan (Jt Rp)",
                   title_font_color="#8892B0",title_font_size=10,
                   tickfont_color="#8892B0",zeroline=False),
        xaxis=dict(tickfont_color="#EEF0F7",tickfont_size=12),
        margin=dict(t=15,b=10,l=45,r=10),height=260,showlegend=False,bargap=.4)
    return fig

def sens_chart():
    r=np.linspace(0,50,25)
    id_=[model.predict(scaler.transform([[v,B_DISKON]]))[0]-baseline for v in r]
    dd_=[model.predict(scaler.transform([[B_IKLAN,v]]))[0]-baseline for v in r]
    fig=go.Figure()
    for y,name,col,dash,fc in [
        (id_,"Iklan",GOLD,"solid","rgba(212,175,55,.07)"),
        (dd_,"Diskon","#38BDF8","dot","rgba(56,189,248,.06)"),
    ]:
        fig.add_trace(go.Scatter(x=r,y=y,name=name,
            line={"color":col,"width":2.2,"dash":dash},
            fill="tozeroy",fillcolor=fc,mode="lines"))
    fig.add_hline(y=0,line_color="rgba(239,68,68,.45)",line_dash="dash",line_width=1)
    fig.update_layout(
        paper_bgcolor=DARK,plot_bgcolor=DARKER,
        font_color="#EEF0F7",font_family="Inter",
        legend=dict(bgcolor="rgba(10,22,40,.8)",bordercolor="rgba(212,175,55,.2)",
                    borderwidth=1,font_size=10,font_color="#EEF0F7"),
        yaxis=dict(gridcolor="rgba(136,146,176,.1)",title="Δ Keuntungan (Jt)",
                   title_font_color="#8892B0",title_font_size=10,tickfont_color="#8892B0",zeroline=False),
        xaxis=dict(gridcolor="rgba(136,146,176,.08)",title="Nilai Variabel",
                   title_font_color="#8892B0",title_font_size=10,tickfont_color="#8892B0"),
        margin=dict(t=15,b=40,l=50,r=10),height=260)
    return fig

def saw_chart(skor, ranking):
    """Bar chart horizontal untuk skor SAW tiap alternatif."""
    labels = ["Strategi A\n(Iklan Agresif)", "Strategi B\n(Diskon Sedang)", "Strategi C\n(Konservatif)"]
    colors = [GOLD if i == ranking[0] else "#38BDF8" if i == ranking[1] else "#8892B0" for i in range(3)]
    fig = go.Figure(go.Bar(
        x=skor, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{s:.4f}" for s in skor],
        textposition="outside", textfont_color="#EEF0F7",
        width=0.5,
    ))
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=DARKER,
        font_color="#EEF0F7", font_family="Inter",
        xaxis=dict(gridcolor="rgba(136,146,176,.1)", range=[0, 1.1],
                   tickfont_color="#8892B0", title="Skor SAW", title_font_color="#8892B0"),
        yaxis=dict(tickfont_color="#EEF0F7", tickfont_size=11),
        margin=dict(t=10, b=30, l=10, r=60), height=220, showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Tuas Kebijakan")
    st.markdown('<hr style="border-color:rgba(212,175,55,.2);margin:.6rem 0 1rem">', unsafe_allow_html=True)

    st.markdown('<p class="sb-lbl">Anggaran Iklan</p>', unsafe_allow_html=True)
    iklan = st.slider("iklan", 0, 50, B_IKLAN, label_visibility="collapsed")
    st.markdown(f'<p style="font-size:.68rem;color:#8892B0;margin:-.3rem 0 .9rem">Rp <b style="color:{GOLD}">{iklan} Juta</b></p>', unsafe_allow_html=True)

    st.markdown('<p class="sb-lbl">Besaran Diskon</p>', unsafe_allow_html=True)
    diskon = st.slider("diskon", 0, 50, B_DISKON, label_visibility="collapsed")
    st.markdown(f'<p style="font-size:.68rem;color:#8892B0;margin:-.3rem 0 1rem"><b style="color:{GOLD}">{diskon}%</b></p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(212,175,55,.2);margin:0 0 .9rem">', unsafe_allow_html=True)
    st.markdown(f"""<div class="bl"><b>📌 Baseline</b><br>
    Iklan &nbsp;: Rp {B_IKLAN} Juta<br>Diskon : {B_DISKON}%<br>
    Profit &nbsp;: <b>Rp {baseline:.1f} Jt</b></div>""", unsafe_allow_html=True)
    st.markdown('<p style="font-size:.62rem;color:#8892B0;margin-top:.7rem;text-align:center">⬦ Garis emas gauge = baseline</p>', unsafe_allow_html=True)

    # Drift Monitor
    st.markdown('<hr style="border-color:rgba(212,175,55,.2);margin:.9rem 0">', unsafe_allow_html=True)
    st.markdown('<p class="sb-lbl">🔍 Monitor Drift (M15)</p>', unsafe_allow_html=True)
    current_input = np.array([[iklan, diskon]])
    is_drift, drift_val = check_data_drift(current_input, X_train_ref)
    if is_drift:
        st.warning(f"⚠️ Drift terdeteksi: **{drift_val}**\nInput jauh dari profil data latih. Hasil mungkin kurang akurat.")
    else:
        st.success(f"✅ Sistem Stabil (drift: {drift_val})\nInput sesuai profil data latih.")

    st.markdown(f"""<div class="bl" style="margin-top:.5rem">
    <b>📦 Model File</b><br>
    model_kebijakan_v1.joblib<br>
    scaler_kebijakan_v1.joblib<br>
    <span style="color:#22C55E">✓ Tersimpan di disk</span></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────
pred  = round(model.predict(scaler.transform([[iklan, diskon]]))[0], 2)
delta = round(pred - baseline, 2)
sign  = "▲" if delta >= 0 else "▼"
dcls  = "green" if delta >= 0 else "red"

if   delta >  5: pill_cls, pill_lbl, narasi = "g", "✅ Untung", "Skenario menguntungkan. Pertimbangkan diterapkan."
elif delta < -5: pill_cls, pill_lbl, narasi = "r", "⚠️ Rugi",   "Skenario merugikan. Tinjau ulang anggaran."
else:            pill_cls, pill_lbl, narasi = "y", "➖ Netral",  "Dampak minimal. Coba variasikan slider lebih jauh."

dominant = "Anggaran Iklan" if abs(ic) > abs(dc) else "Besaran Diskon"

# SAW & XAI
matriks_saw, norm_saw, skor_saw, ranking_saw = hitung_saw(pred, iklan, diskon)
xai_labels, xai_kontribusi, xai_intercept   = hitung_xai(iklan, diskon)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <div>
    <div class="badge-tag">Pemodelan &amp; Simulasi · Minggu 16 — UAS Edition</div>
    <p class="ttl">📊 Sim<span>Kebijakan</span></p>
    <p class="sub">Simulator Interaktif What-If — Integrasi ML · XAI · SPK (SAW) · MLOps</p>
  </div>
  <div class="auth">
    <div class="auth-name">Mohamad Yusuf</div>
    <div class="auth-npm">NPM 2313020037</div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 1 — GAUGE + CARDS
# ─────────────────────────────────────────────
st.markdown('<p class="sec">Ringkasan Simulasi</p>', unsafe_allow_html=True)
col_g, col_cards = st.columns([1.6, 1])
with col_g:
    st.plotly_chart(gauge(pred), use_container_width=True)
with col_cards:
    st.markdown(f"""
    <div class="mc"><div class="mc-icon">💰</div><div class="mc-body">
    <div class="mc-lbl">Prediksi Profit</div>
    <div class="mc-val">Rp {pred:.1f} <span style="font-size:.9rem">Jt</span></div>
    <div class="mc-sub">Juta Rupiah</div></div></div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mc"><div class="mc-icon">📈</div><div class="mc-body">
    <div class="mc-lbl">Delta (Δ) vs Baseline</div>
    <div class="mc-val {dcls}">{sign} {abs(delta):.1f} <span style="font-size:.9rem">Jt</span></div>
    <div class="mc-sub">Selisih dari nilai baseline</div></div></div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sc"><div class="mc-lbl">Status Skenario</div>
    <div><span class="pill {pill_cls}">{pill_lbl}</span></div>
    <div class="sc-txt">{narasi}</div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 2 — CHART VISUAL
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="sec">Analisis Visual</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown('<p style="font-size:.78rem;font-weight:600;color:#EEF0F7;margin-bottom:.4rem">Perbandingan Skenario</p>', unsafe_allow_html=True)
    st.plotly_chart(bar_chart(baseline, pred), use_container_width=True)
with c2:
    st.markdown('<p style="font-size:.78rem;font-weight:600;color:#EEF0F7;margin-bottom:.4rem">Peta Sensitivitas Variabel</p>', unsafe_allow_html=True)
    st.plotly_chart(sens_chart(), use_container_width=True)
    st.caption("Kemiringan lebih curam → variabel lebih sensitif terhadap profit.")

# ─────────────────────────────────────────────
# ROW 3 — XAI (M16 NEW)
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="sec">🔍 Transparansi — Mengapa Hasilnya Demikian? (XAI · M16)</p>', unsafe_allow_html=True)

col_xai, col_xai_info = st.columns([1.2, 1])

with col_xai:
    st.markdown('<p style="font-size:.78rem;font-weight:600;color:#EEF0F7;margin-bottom:.8rem">Kontribusi Fitur terhadap Prediksi Profit</p>', unsafe_allow_html=True)

    total_kontrib = abs(xai_kontribusi).sum()
    bar_rows = ""
    for label, val in zip(xai_labels, xai_kontribusi):
        pct  = abs(val) / (total_kontrib + 1e-9) * 100
        cls  = "xai-pos" if val >= 0 else "xai-neg"
        fill = "#22C55E" if val >= 0 else "#EF4444"
        arah = f"+{val:.2f}" if val >= 0 else f"{val:.2f}"
        bar_rows += f"""
        <div class="xai-row">
          <div class="xai-lbl">{label}</div>
          <div class="xai-bar-wrap">
            <div class="xai-bar-inner" style="width:{min(pct*2, 100):.1f}%;background:{fill};"></div>
          </div>
          <div class="xai-val {cls}">{arah}</div>
        </div>"""

    intercept_str = f"+{xai_intercept:.2f}" if xai_intercept >= 0 else f"{xai_intercept:.2f}"
    bar_rows += f"""
    <div class="xai-row">
      <div class="xai-lbl" style="color:#8892B0">Intercept (bias)</div>
      <div class="xai-bar-wrap"><div class="xai-bar-inner" style="width:30%;background:#8892B0;"></div></div>
      <div class="xai-val" style="color:#8892B0">{intercept_str}</div>
    </div>"""

    st.markdown(f'<div style="background:#0E1E38;border:1px solid rgba(212,175,55,.18);border-radius:12px;padding:1rem 1.2rem;">{bar_rows}</div>', unsafe_allow_html=True)

with col_xai_info:
    dom_fitur = xai_labels[int(np.argmax(np.abs(xai_kontribusi)))]
    dom_val   = xai_kontribusi[int(np.argmax(np.abs(xai_kontribusi)))]
    st.markdown(f"""
<div class="sc" style="margin-top:0">
<div class="mc-lbl">Interpretasi XAI</div>
<br>
<div class="sc-txt">
Prediksi profit <b style="color:#D4AF37">Rp {pred:.2f} Jt</b> dihasilkan dari:<br><br>
• <b style="color:#E8CC6A">Anggaran Iklan</b> memberikan kontribusi <b style="color:{'#22C55E' if xai_kontribusi[0]>=0 else '#EF4444'}">{xai_kontribusi[0]:+.2f} Jt</b><br>
• <b style="color:#E8CC6A">Besaran Diskon</b> memberikan kontribusi <b style="color:{'#22C55E' if xai_kontribusi[1]>=0 else '#EF4444'}">{xai_kontribusi[1]:+.2f} Jt</b><br>
• <b style="color:#8892B0">Bias model (intercept)</b>: {xai_intercept:.2f} Jt<br><br>
🔑 <b style="color:#D4AF37">Fitur dominan: {dom_fitur}</b><br>
dengan kontribusi {dom_val:+.2f} Jt terhadap hasil akhir.
</div>
</div>""", unsafe_allow_html=True)

    if is_drift:
        st.warning("⚠️ **Peringatan:** Drift terdeteksi — interpretasi XAI di atas mungkin kurang akurat karena input di luar jangkauan data latih.")

# ─────────────────────────────────────────────
# ROW 4 — SAW / MCDM (M16 NEW)
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="sec">🏆 Rekomendasi Strategi — SPK Metode SAW (M16)</p>', unsafe_allow_html=True)

alt_labels = ["Strategi A (Iklan Agresif)", "Strategi B (Diskon Sedang)", "Strategi C (Konservatif)"]
badge_cls  = ["", "s2", "s3"]

col_saw1, col_saw2 = st.columns([1, 1.2])

with col_saw1:
    st.markdown('<p style="font-size:.78rem;font-weight:600;color:#EEF0F7;margin-bottom:.6rem">Skor SAW per Alternatif</p>', unsafe_allow_html=True)
    st.plotly_chart(saw_chart(skor_saw, ranking_saw), use_container_width=True)

with col_saw2:
    st.markdown('<p style="font-size:.78rem;font-weight:600;color:#EEF0F7;margin-bottom:.6rem">Ranking Keputusan Akhir</p>', unsafe_allow_html=True)

    rows_html = ""
    for rank_pos, alt_idx in enumerate(ranking_saw):
        badge = f'<span class="rank-badge {badge_cls[rank_pos]}">{rank_pos+1}</span>'
        is_best = "best" if rank_pos == 0 else ""
        crown = " 👑" if rank_pos == 0 else ""
        rows_html += f"""<tr class="{is_best}">
          <td>{badge}</td>
          <td><b>{alt_labels[alt_idx]}{crown}</b></td>
          <td style="color:#D4AF37;font-weight:700">{skor_saw[alt_idx]:.4f}</td>
          <td style="color:#8892B0">{matriks_saw[alt_idx,0]:.1f} Jt</td>
        </tr>"""

    st.markdown(f"""
<table class="rank-table">
  <thead><tr>
    <th>Rank</th><th>Alternatif</th><th>Skor SAW</th><th>Profit ML</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)

    best_idx   = ranking_saw[0]
    best_label = alt_labels[best_idx]
    best_skor  = skor_saw[best_idx]
    st.markdown(f"""
<div class="bl" style="margin-top:.8rem">
<b>📌 Rekomendasi Terbaik: {best_label}</b><br>
Skor SAW tertinggi = <b style="color:#D4AF37">{best_skor:.4f}</b>.<br>
Bobot AHP: Profit-ML 50% · Efisiensi 30% · Biaya 20%.<br>
<span style="color:#22C55E">Fitur dominan ML ({dominant}) konsisten dengan bobot tertinggi SPK.</span>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 5 — INTERPRETASI (dipertahankan dari M15)
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
with st.expander("📋 Interpretasi & Rekomendasi Kebijakan", expanded=False):
    st.markdown(f"""
**Hasil What-If:** Iklan Rp {iklan} Jt + Diskon {diskon}% → profit **Rp {pred:.2f} Jt** ({sign} Rp {abs(delta):.2f} Jt vs baseline)

**Sensitivitas:**
- Koef. Iklan `{ic}` → tiap +1 Jt iklan = +Rp {ic:.2f} Jt profit
- Koef. Diskon `{dc}` → tiap +1% diskon = Rp {dc:.2f} Jt profit
- **Variabel paling sensitif: {dominant}**

**Rekomendasi SPK (SAW):** {best_label} dengan skor {best_skor:.4f}

**Rekomendasi:** {narasi}
    """)

# ─────────────────────────────────────────────
# ROW 6 — MLOps: DRIFT + ANONYMIZATION (M15)
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="sec">MLOps — Monitoring & Etika Data (M15)</p>', unsafe_allow_html=True)

col_drift, col_anon = st.columns(2)

with col_drift:
    st.markdown("**🔍 Deteksi Drift Input**")
    st.markdown(f"""
| Parameter | Nilai |
|---|---|
| Rata-rata Data Latih | `{np.mean(X_train_ref):.2f}` |
| Rata-rata Input Aktif | `{np.mean(current_input):.2f}` |
| Selisih (Drift Score) | `{drift_val}` |
| Threshold | `15.0` |
| Status | {"⚠️ **DRIFT TERDETEKSI**" if is_drift else "✅ **Stabil**"} |
    """)
    if is_drift:
        st.warning("Model mungkin perlu dikalibrasi ulang dengan data terbaru.")
    else:
        st.info("Input dalam jangkauan normal data latih.")

with col_anon:
    st.markdown("**🔒 Demo Anonymization (Etika Data)**")
    raw_df = pd.DataFrame({
        'Nama_Operator': ['Budi Santoso'],
        'NIK_Petugas':   ['3273012345'],
        'Iklan_Jt':      [iklan],
        'Diskon_Pct':    [diskon],
    })
    st.markdown("*Data Mentah (sebelum filter):*")
    st.dataframe(raw_df, use_container_width=True, hide_index=True)
    clean_df = clean_sensitive_data(raw_df)
    st.markdown("*Data Bersih (setelah anonymization):*")
    st.dataframe(clean_df, use_container_width=True, hide_index=True)
    st.caption("Kolom PII (Nama, NIK) otomatis dihapus sebelum diproses model.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<div class='footer'>Mohamad Yusuf - NPM 2313020037 · Pemodelan &amp; Simulasi M16 · UAS Edition — ML · XAI · SAW · MLOps</div>",
    unsafe_allow_html=True
)
