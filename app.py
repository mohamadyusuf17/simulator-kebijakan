import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="SimKebijakan", page_icon="📊", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
:root {
    --bg:     #060D1F; --card:   #0E1E38; --deep:   #0A1628;
    --border: rgba(212,175,55,0.18);
    --gold:   #D4AF37; --gold2:  #E8CC6A;
    --slate:  #8892B0; --white:  #EEF0F7;
    --green:  #22C55E; --red:    #EF4444;
}
html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
.stApp { background: var(--bg); }
section[data-testid="stSidebar"]{ background: var(--deep)!important; border-right:1px solid var(--border); }

/* Header */
.hdr { display:flex; justify-content:space-between; align-items:flex-end;
        padding-bottom:1rem; border-bottom:1px solid var(--border); margin-bottom:1.4rem; }
.badge-tag { display:inline-block; background:rgba(212,175,55,.1); border:1px solid rgba(212,175,55,.3);
             border-radius:4px; padding:.12rem .6rem; font-size:.6rem; font-weight:700;
             letter-spacing:2px; text-transform:uppercase; color:var(--gold); margin-bottom:.4rem; }
.ttl { font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700;
       color:var(--white); line-height:1; margin:0; }
.ttl span { color:var(--gold); }
.sub { font-size:.74rem; color:var(--slate); margin-top:.35rem; }
.auth { text-align:right; }
.auth-name { font-size:.85rem; font-weight:600; color:var(--gold2); }
.auth-npm  { font-size:.68rem; color:var(--slate); margin-top:.1rem; }

/* Section label */
.sec { font-family:'Space Grotesk',sans-serif; font-size:.65rem; font-weight:700;
       letter-spacing:2px; text-transform:uppercase; color:var(--gold);
       display:flex; align-items:center; gap:.5rem; margin-bottom:.9rem; }
.sec::after { content:''; flex:1; height:1px; background:var(--border); }

/* Metric card — horizontal compact */
.mc {
    background:var(--card); border:1px solid var(--border); border-radius:12px;
    padding:.85rem 1.1rem; position:relative; overflow:hidden;
    display:flex; align-items:center; gap:1rem; margin-bottom:.7rem;
}
.mc::before { content:''; position:absolute; top:0;left:0;right:0; height:2px;
              background:linear-gradient(90deg,transparent,var(--gold),transparent); }
.mc-icon { font-size:1.4rem; flex-shrink:0; }
.mc-body { flex:1; min-width:0; }
.mc-lbl  { font-size:.58rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--slate); }
.mc-val  { font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700;
           color:var(--gold); line-height:1.1; }
.mc-val.green { color:var(--green); }
.mc-val.red   { color:var(--red); }
.mc-sub  { font-size:.62rem; color:var(--slate); margin-top:.15rem; }

/* Status card */
.sc { background:var(--card); border:1px solid var(--border); border-radius:12px;
      padding:.85rem 1.1rem; position:relative; overflow:hidden; }
.sc::before { content:''; position:absolute; top:0;left:0;right:0; height:2px;
              background:linear-gradient(90deg,transparent,var(--gold),transparent); }
.pill { display:inline-flex; align-items:center; gap:.3rem; padding:.22rem .8rem;
        border-radius:999px; font-size:.75rem; font-weight:700; margin-bottom:.35rem; }
.pill.g { background:rgba(34,197,94,.1);  border:1px solid rgba(34,197,94,.3);  color:var(--green); }
.pill.r { background:rgba(239,68,68,.1);  border:1px solid rgba(239,68,68,.3);  color:var(--red); }
.pill.y { background:rgba(212,175,55,.1); border:1px solid rgba(212,175,55,.3); color:var(--gold); }
.sc-txt { font-size:.68rem; color:var(--slate); line-height:1.5; }

/* Sidebar baseline box */
.bl { background:var(--card); border:1px solid var(--border); border-radius:10px;
      padding:.8rem 1rem; font-size:.73rem; color:var(--slate); line-height:1.9; }
.bl b { color:var(--gold2); }
.sb-lbl { font-size:.6rem; font-weight:700; letter-spacing:2px; text-transform:uppercase;
          color:var(--gold); margin-bottom:.35rem; }

.divider { border:none; border-top:1px solid var(--border); margin:1.2rem 0; }
.footer { text-align:center; font-size:.67rem; color:var(--slate);
          border-top:1px solid var(--border); padding:.9rem 0 .4rem; margin-top:.8rem; letter-spacing:.4px; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    X = np.array([[5,10],[10,20],[15,5],[20,25],[25,15]])
    y = np.array([50, 80, 110, 90, 150])
    return LinearRegression().fit(X, y)

model        = load_model()
B_IKLAN      = 10; B_DISKON = 10
baseline     = model.predict([[B_IKLAN, B_DISKON]])[0]
ic, dc       = round(model.coef_[0], 2), round(model.coef_[1], 2)

# ── CHARTS ────────────────────────────────────────────────────────────────────
DARK = "#0E1E38"; DARKER = "#0A1628"; GOLD = "#D4AF37"
FONT = {"color":"#EEF0F7","family":"Inter"}

def gauge(val):
    color = "#EF4444" if val/200<.4 else GOLD if val/200<.7 else "#22C55E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=val,
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
    fig.update_layout(paper_bgcolor=DARK, plot_bgcolor=DARK,
                      margin=dict(t=70,b=10,l=20,r=20), height=300, font=FONT)
    return fig

def bar_chart(base, pred):
    fig = go.Figure(go.Bar(
        x=["Baseline","Skenario Aktif"], y=[base, pred],
        marker_color=[DARKER, GOLD], marker_line_color=["#8892B0","#E8CC6A"], marker_line_width=1.2,
        text=[f"Rp {base:.1f}","Rp {:.1f}".format(pred)], textposition="outside",
        textfont={"color":"#EEF0F7","size":12}, width=0.38,
    ))
    fig.update_layout(paper_bgcolor=DARK, plot_bgcolor=DARKER, font=FONT,
                      yaxis={"gridcolor":"rgba(136,146,176,.1)","title":"Keuntungan (Jt Rp)",
                             "titlefont":{"color":"#8892B0","size":10},"tickfont":{"color":"#8892B0"},"zeroline":False},
                      xaxis={"tickfont":{"size":12,"color":"#EEF0F7"}},
                      margin=dict(t=15,b=10,l=45,r=10), height=260, showlegend=False, bargap=.4)
    return fig

def sens_chart():
    r = np.linspace(0, 50, 25)
    id_ = [model.predict([[v, B_DISKON]])[0] - baseline for v in r]
    dd_ = [model.predict([[B_IKLAN, v]])[0] - baseline for v in r]
    fig = go.Figure()
    for y, name, col, dash, fc in [
        (id_, "Iklan",  GOLD,     "solid", "rgba(212,175,55,.07)"),
        (dd_, "Diskon", "#38BDF8","dot",   "rgba(56,189,248,.06)"),
    ]:
        fig.add_trace(go.Scatter(x=r, y=y, name=name,
            line={"color":col,"width":2.2,"dash":dash}, fill="tozeroy", fillcolor=fc, mode="lines"))
    fig.add_hline(y=0, line_color="rgba(239,68,68,.45)", line_dash="dash", line_width=1)
    fig.update_layout(paper_bgcolor=DARK, plot_bgcolor=DARKER, font=FONT,
                      legend={"bgcolor":"rgba(10,22,40,.8)","bordercolor":"rgba(212,175,55,.2)",
                              "borderwidth":1,"font":{"size":10,"color":"#EEF0F7"}},
                      yaxis={"gridcolor":"rgba(136,146,176,.1)","title":"Δ Keuntungan (Jt)",
                             "titlefont":{"color":"#8892B0","size":10},"tickfont":{"color":"#8892B0"},"zeroline":False},
                      xaxis={"gridcolor":"rgba(136,146,176,.08)","title":"Nilai Variabel",
                             "titlefont":{"color":"#8892B0","size":10},"tickfont":{"color":"#8892B0"}},
                      margin=dict(t=15,b=40,l=50,r=10), height=260)
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
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

# ── COMPUTE ───────────────────────────────────────────────────────────────────
pred  = round(model.predict([[iklan, diskon]])[0], 2)
delta = round(pred - baseline, 2)
sign  = "▲" if delta >= 0 else "▼"
dcls  = "green" if delta >= 0 else "red"

if   delta >  5: pill_cls, pill_lbl, narasi = "g", "✅ Untung",  "Skenario menguntungkan. Pertimbangkan diterapkan."
elif delta < -5: pill_cls, pill_lbl, narasi = "r", "⚠️ Rugi",   "Skenario merugikan. Tinjau ulang anggaran."
else:            pill_cls, pill_lbl, narasi = "y", "➖ Netral",  "Dampak minimal. Coba variasikan slider lebih jauh."

dominant = "Anggaran Iklan" if abs(ic) > abs(dc) else "Besaran Diskon"

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <div>
    <div class="badge-tag">Pemodelan &amp; Simulasi · Minggu 14</div>
    <p class="ttl">📊 Sim<span>Kebijakan</span></p>
    <p class="sub">Simulator Interaktif What-If — Analisis Kebijakan Berbasis Data</p>
  </div>
  <div class="auth">
    <div class="auth-name">Mohamad Yusuf</div>
    <div class="auth-npm">NPM 2313020037</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── ROW 1 — Gauge (kiri) + 3 Kartu vertikal (kanan) ─────────────────────────
st.markdown('<p class="sec">Ringkasan Simulasi</p>', unsafe_allow_html=True)

col_g, col_cards = st.columns([1.6, 1])

with col_g:
    st.plotly_chart(gauge(pred), use_container_width=True)

with col_cards:
    # Kartu 1 — Profit
    st.markdown(f"""
    <div class="mc">
      <div class="mc-icon">💰</div>
      <div class="mc-body">
        <div class="mc-lbl">Prediksi Profit</div>
        <div class="mc-val">Rp {pred:.1f} <span style="font-size:.9rem">Jt</span></div>
        <div class="mc-sub">Juta Rupiah</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Kartu 2 — Delta
    st.markdown(f"""
    <div class="mc">
      <div class="mc-icon">📈</div>
      <div class="mc-body">
        <div class="mc-lbl">Delta (Δ) vs Baseline</div>
        <div class="mc-val {dcls}">{sign} {abs(delta):.1f} <span style="font-size:.9rem">Jt</span></div>
        <div class="mc-sub">Selisih dari nilai baseline</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Kartu 3 — Status
    st.markdown(f"""
    <div class="sc">
      <div class="mc-lbl">Status Skenario</div>
      <div><span class="pill {pill_cls}">{pill_lbl}</span></div>
      <div class="sc-txt">{narasi}</div>
    </div>""", unsafe_allow_html=True)

# ── ROW 2 — Charts ────────────────────────────────────────────────────────────
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

# ── INTERPRETASI ──────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
with st.expander("📋 Interpretasi & Rekomendasi Kebijakan", expanded=True):
    st.markdown(f"""
**Hasil What-If:** Iklan Rp {iklan} Jt + Diskon {diskon}% → profit **Rp {pred:.2f} Jt** ({sign} Rp {abs(delta):.2f} Jt vs baseline)

**Sensitivitas:**
- Koef. Iklan `{ic}` → tiap +1 Jt iklan = +Rp {ic:.2f} Jt profit
- Koef. Diskon `{dc}` → tiap +1% diskon = Rp {dc:.2f} Jt profit
- **Variabel paling sensitif: {dominant}**

**Rekomendasi:** {narasi}
    """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<div class='footer'>Mohamad Yusuf - NPM 2313020037</div>",
            unsafe_allow_html=True)