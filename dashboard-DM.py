import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropYield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

* { font-family: 'Syne', sans-serif; }
code, .mono { font-family: 'DM Mono', monospace !important; }

:root {
    --soil: #1a1208;
    --bark: #2d1f0f;
    --moss: #3d5a3e;
    --leaf: #6aaa64;
    --sun:  #f0c040;
    --sky:  #b8d4e8;
    --sand: #e8d5b0;
    --cream: #faf6ee;
}

.stApp { background: var(--cream); }

/* Hide default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] { display: none !important; }

/* ── HERO HEADER ── */
.hero-wrap {
    background: linear-gradient(135deg, var(--soil) 0%, var(--bark) 40%, var(--moss) 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: "🌿🌾🌱";
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 4rem;
    opacity: 0.25;
    letter-spacing: 1rem;
}
.hero-title {
    color: var(--sun);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-sub {
    color: var(--sand);
    font-size: 0.9rem;
    font-weight: 400;
    margin-top: 0.3rem;
    opacity: 0.85;
}
.hero-pills {
    display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,0.12);
    color: var(--sky);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ── NAV TABS ── */
div[data-testid="stHorizontalBlock"] > div { }
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 12px;
    padding: 0.3rem;
    gap: 0;
    border: 1.5px solid #e0d5c0;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.85rem;
    color: #666;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: var(--moss) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

/* ── METRIC CARDS ── */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1.5px solid #e8dfc8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-val {
    font-size: 2rem;
    font-weight: 800;
    color: var(--bark);
    line-height: 1;
    font-family: 'DM Mono', monospace;
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--moss);
    font-weight: 600;
    margin-top: 0.3rem;
}

/* ── SECTION HEADERS ── */
.sec-head {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--bark);
    letter-spacing: -0.01em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1.5rem;
}
.sec-line {
    flex: 1;
    height: 1.5px;
    background: linear-gradient(90deg, #d0c8b4, transparent);
}

/* ── PREDICT PANEL ── */
.predict-wrap {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1.5px solid #e8dfc8;
}
.predict-result {
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-HIGH { background: #e8f5e9; border: 2px solid #66bb6a; }
.result-MEDIUM { background: #fff8e1; border: 2px solid #ffca28; }
.result-LOW { background: #fce4ec; border: 2px solid #ef9a9a; }
.result-label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; color: #666; }
.result-value { font-size: 2.5rem; font-weight: 800; margin: 0.3rem 0; }
.result-HIGH .result-value { color: #2e7d32; }
.result-MEDIUM .result-value { color: #f57f17; }
.result-LOW .result-value { color: #c62828; }

/* ── MODEL BADGE ── */
.model-badge {
    display: inline-block;
    background: var(--moss);
    color: white;
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: var(--leaf) !important;
}
.stSlider label, .stSlider [data-testid="stWidgetLabel"] p {
    color: #2d1f0f !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #888 !important;
}
div[data-testid="stSliderThumbValue"] {
    color: #2d1f0f !important;
    font-weight: 700 !important;
    font-family: 'DM Mono', monospace !important;
}
/* Fix all widget labels */
.stSelectbox label p, .stSlider label p {
    color: #2d1f0f !important;
    font-weight: 600 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    border-color: #d0c8b4 !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button {
    background: var(--moss);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.02em;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--bark);
    transform: translateY(-1px);
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 700 !important;
    color: var(--bark) !important;
    font-size: 0.85rem !important;
}

/* Info/success box */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA & MODELS ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('crop_yield_tabular_xlsx_-_Sheet2.csv')
    return df

@st.cache_resource
def load_models():
    features_rfe = joblib.load('selected_features_rfe.joblib')
    scaler = joblib.load('scaler.joblib')
    nb = joblib.load('nb_smote.joblib')
    dt_rfe = joblib.load('dt_rfe_optimized.joblib')
    dt_all = joblib.load('dt_all_features.joblib')
    return features_rfe, scaler, nb, dt_rfe, dt_all

# Load with path handling
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')

try:
    df = load_data()
    features_rfe, scaler, nb, dt_rfe, dt_all = load_models()
    DATA_OK = True
except Exception as e:
    st.error(f"Error loading files: {e}")
    DATA_OK = False
    st.stop()

ALL_FEATURES = ['N', 'P', 'K', 'pH', 'temperature', 'humidity', 'rainfall',
                'Soil_Moisture', 'Wind_speed', 'Sunshine_hours', 'Organic_Carbon']
RFE_FEATURES = list(features_rfe)

FEATURE_META = {
    'N':              {'label': 'Nitrogen (N)',        'unit': 'mg/kg', 'emoji': '🧪'},
    'P':              {'label': 'Phosphorus (P)',      'unit': 'mg/kg', 'emoji': '🧪'},
    'K':              {'label': 'Potassium (K)',       'unit': 'mg/kg', 'emoji': '🧪'},
    'pH':             {'label': 'Soil pH',             'unit': '',      'emoji': '⚗️'},
    'temperature':    {'label': 'Temperature',         'unit': '°C',    'emoji': '🌡️'},
    'humidity':       {'label': 'Humidity',            'unit': '%',     'emoji': '💧'},
    'rainfall':       {'label': 'Rainfall',            'unit': 'mm',    'emoji': '🌧️'},
    'Soil_Moisture':  {'label': 'Soil Moisture',       'unit': '%',     'emoji': '🌱'},
    'Wind_speed':     {'label': 'Wind Speed',          'unit': 'km/h',  'emoji': '💨'},
    'Sunshine_hours': {'label': 'Sunshine Hours',      'unit': 'h/day', 'emoji': '☀️'},
    'Organic_Carbon': {'label': 'Organic Carbon',      'unit': '%',     'emoji': '🍂'},
}

COLORS = {
    'High':   '#1a7a1a',
    'Medium': '#e65c00',
    'Low':    '#b00020',
}

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">🌾 CropYield Intelligence</div>
  <div class="hero-sub">Data Mining Dashboard — Prediksi & Analisis Hasil Panen</div>
  <div class="hero-pills">
    <span class="pill">102.675 sampel</span>
    <span class="pill">11 fitur sensor</span>
    <span class="pill">3 model ML</span>
    <span class="pill">Klasifikasi High · Medium · Low</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  Overview Dataset", "🤖  Prediksi Yield", "🔬  Perbandingan Model", "📂  Batch Prediksi"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW (Tetap sama seperti kode asli)
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    stats = df.describe()
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Total Sampel", f"{len(df):,}", "baris data"),
        ("Fitur Input", "11", "variabel sensor"),
        ("Rata² Curah Hujan", f"{df['rainfall'].mean():.0f}", "mm/tahun"),
        ("Rentang Suhu", f"{df['temperature'].min():.0f}–{df['temperature'].max():.0f}", "°C"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-val">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">📈 Distribusi Fitur Utama <div class="sec-line"></div></div>', unsafe_allow_html=True)

    feat_groups = [['N', 'P', 'K'], ['temperature', 'humidity', 'rainfall']]
    for group in feat_groups:
        cols = st.columns(3)
        for col, feat in zip(cols, group):
            meta = FEATURE_META[feat]
            fig = px.histogram(df.sample(5000, random_state=42), x=feat, nbins=40, color_discrete_sequence=['#6aaa64'], template='simple_white')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), title=dict(text=f"{meta['emoji']} {meta['label']}", font_size=12, font_color='#2d1f0f'), xaxis_title=dict(text=meta['unit'], font=dict(color='#2d1f0f', size=11)), yaxis_title='', showlegend=False, plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Syne', color='#2d1f0f'))
            fig.update_xaxes(showgrid=False, color='#2d1f0f')
            fig.update_yaxes(showgrid=True, gridcolor='#f0ece0', color='#2d1f0f')
            col.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-head">🔗 Korelasi Antar Fitur <div class="sec-line"></div></div>', unsafe_allow_html=True)
    col_heat, col_scatter = st.columns([1.2, 1])
    with col_heat:
        corr = df[ALL_FEATURES].corr()
        fig_heat = px.imshow(corr, color_continuous_scale=['#b00020', '#f5f5f5', '#1a7a1a'], zmin=-1, zmax=1, text_auto='.2f', template='simple_white')
        fig_heat.update_layout(height=420, margin=dict(l=120, r=20, t=20, b=120), paper_bgcolor='white', plot_bgcolor='white', font=dict(family='DM Mono', size=9, color='#2d1f0f'), coloraxis_showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_scatter:
        sample = df.sample(3000, random_state=99).copy()
        sample['Yield_Cat'] = pd.qcut(sample['rainfall'], q=3, labels=['Low', 'Medium', 'High'])
        fig_sc = px.scatter(sample, x='temperature', y='humidity', color='Yield_Cat', color_discrete_map=COLORS, opacity=0.75, template='simple_white', labels={'temperature': 'Suhu (°C)', 'humidity': 'Kelembaban (%)'})
        fig_sc.update_layout(height=420, margin=dict(l=50, r=20, t=50, b=80), paper_bgcolor='white', plot_bgcolor='white', font=dict(family='Syne', color='#2d1f0f'), legend=dict(title='Estimasi Yield', orientation='h', y=-0.18, x=0.5, xanchor='center'))
        st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("📋 Statistik Deskriptif Lengkap"):
        styled = df[ALL_FEATURES].describe().T.round(2)
        styled.index = [f"{FEATURE_META[c]['emoji']} {FEATURE_META[c]['label']}" for c in styled.index]
        st.dataframe(styled, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI INDIVIDUAL (Tetap sama seperti kode asli)
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    col_form, col_result = st.columns([1.1, 1])
    with col_form:
        st.markdown('<div class="sec-head">⚙️ Input Parameter Lahan <div class="sec-line"></div></div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Pilih Model", ["Decision Tree (Semua Fitur)", "Decision Tree (RFE — 5 Fitur)", "Naive Bayes + SMOTE (RFE)"])
        active_features = ALL_FEATURES if "Semua Fitur" in model_choice or "Naive" in model_choice else RFE_FEATURES
        defaults = {f: float(df[f].mean()) for f in ALL_FEATURES}
        slider_ranges = {'N':(0,139,1), 'P':(5,99,1), 'K':(5,149,1), 'pH':(4.5,8.0,0.05), 'temperature':(15,40,0.5), 'humidity':(30,90,1.0), 'rainfall':(400,3000,10.0), 'Soil_Moisture':(14,70,1.0), 'Wind_speed':(2,25,0.5), 'Sunshine_hours':(3,12,0.25), 'Organic_Carbon':(0.2,2.0,0.05)}
        inputs = {}
        c_a, c_b = st.columns(2)
        for i, feat in enumerate(active_features):
            meta = FEATURE_META[feat]
            rng = slider_ranges[feat]
            col = c_a if i % 2 == 0 else c_b
            val = col.slider(f"{meta['emoji']} {meta['label']} ({meta['unit']})" if meta['unit'] else f"{meta['emoji']} {meta['label']}", min_value=float(rng[0]), max_value=float(rng[1]), value=float(defaults[feat]), step=float(rng[2]), key=f"sl_{feat}")
            inputs[feat] = val
        predict_btn = st.button("🔍 Prediksi Yield", use_container_width=True)

    with col_result:
        st.markdown('<div class="sec-head">🎯 Hasil Prediksi <div class="sec-line"></div></div>', unsafe_allow_html=True)
        if predict_btn:
            full_inputs = {f: defaults[f] for f in ALL_FEATURES}
            full_inputs.update(inputs)
            X_all = pd.DataFrame([full_inputs])
            if "Semua Fitur" in model_choice:
                pred, proba, classes, model_name = dt_all.predict(X_all)[0], dt_all.predict_proba(X_all)[0], dt_all.classes_, "Decision Tree (All Features)"
            elif "Naive" in model_choice:
                X_scaled = scaler.transform(X_all)
                pred, proba, classes, model_name = nb.predict(X_scaled)[0], nb.predict_proba(X_scaled)[0], nb.classes_, "Naive Bayes + SMOTE"
            else:
                X_rfe = pd.DataFrame([{f: full_inputs[f] for f in RFE_FEATURES}])
                pred, proba, classes, model_name = dt_rfe.predict(X_rfe)[0], dt_rfe.predict_proba(X_rfe)[0], dt_rfe.classes_, "Decision Tree (RFE Optimized)"
            st.markdown(f'<div class="predict-result result-{pred.upper()}"><div class="result-label">PREDIKSI CROP YIELD</div><div class="result-value">{pred.upper()}</div><div style="font-size:0.8rem;color:#666">via <b>{model_name}</b></div></div>', unsafe_allow_html=True)
            fig_prob = go.Figure(go.Bar(x=[str(c) for c in classes], y=[p*100 for p in proba], marker_color=[COLORS.get(c, '#888') for c in classes], text=[f"{p*100:.1f}%" for p in proba], textposition='outside'))
            fig_prob.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 115]), paper_bgcolor='white', plot_bgcolor='white')
            st.plotly_chart(fig_prob, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON (Tetap sama seperti kode asli)
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-head">🔬 Arsitektur Model <div class="sec-line"></div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    model_info = [{"name": "Decision Tree", "sub": "All Features", "emoji": "🌳", "features": "11 fitur", "classes": "High, Medium, Low", "notes": "Baseline model", "color": "#3d5a3e"}, {"name": "Decision Tree", "sub": "RFE Optimized", "emoji": "✂️", "features": "5 fitur", "classes": "High, Medium, Low", "notes": "Fitur RFE", "color": "#6aaa64"}, {"name": "Naive Bayes", "sub": "+ SMOTE", "emoji": "⚖️", "features": "5 fitur + Scaler", "classes": "High, Low", "notes": "SMOTE balanced", "color": "#f57f17"}]
    for col, m in zip([c1, c2, c3], model_info):
        col.markdown(f'<div style="background: white; border-radius: 14px; padding: 1.4rem; border: 1.5px solid #e8dfc8; height: 100%;"><div style="font-size:2rem">{m["emoji"]}</div><div style="font-size:1.1rem;font-weight:800;color:{m["color"]};margin-top:0.5rem">{m["name"]}</div><div style="font-size:0.75rem;font-weight:700;color:#888;">{m["sub"].upper()}</div><hr style="border-color:#f0ece0;margin:0.8rem 0"><div style="font-size:0.82rem;color:#555"><b>Fitur:</b> {m["features"]}<br><b>Kelas:</b> {m["classes"]}<br><br><i>{m["notes"]}</i></div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH PREDIKSI (DIREKAYASA ULANG DENGAN DIAGNOSTIK EKSTREM DETAIL)
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-head">📂 Batch Prediksi dari CSV & Analisis Diagnostik Model <div class="sec-line"></div></div>', unsafe_allow_html=True)

    col_up, col_cfg = st.columns([1.2, 1])

    with col_up:
        batch_model = st.selectbox(
            "Model untuk Batch Prediksi",
            ["Decision Tree (Semua Fitur)", "Decision Tree (RFE — 5 Fitur)", "Naive Bayes + SMOTE"],
            key="batch_model"
        )
        required_cols = RFE_FEATURES if "RFE" in batch_model else ALL_FEATURES
        st.info(f"Kolom yang dibutuhkan wajib ada di CSV: **{', '.join(required_cols)}**")
        
        template_df = pd.DataFrame([{f: round(float(df[f].mean()), 2) for f in required_cols}])
        st.download_button("⬇️ Download Template CSV", data=template_df.to_csv(index=False), file_name="template_batch.csv", mime="text/csv")
        uploaded = st.file_uploader("Upload file CSV untuk dievaluasi mendalam", type=["csv"], label_visibility="collapsed")

    with col_cfg:
        st.markdown("""
        <div style="background:white;border-radius:12px;padding:1rem 1.2rem;border:1.5px solid #e8dfc8;font-size:0.85rem;color:#444;">
        ⚙️ <b>Engine Pemrosesan Data Mining:</b><br>
        • Mengotomatisasi imputasi data kosong (mean-imputer).<br>
        • Mengukur <i>Confidence Score</i> dan <i>Ambiguity Margin</i>.<br>
        • Mendeteksi data yang melenceng dari distribusi historis training dataset.
        </div>
        """, unsafe_allow_html=True)

    if uploaded is not None:
        try:
            df_batch = pd.read_csv(uploaded)
            total_rows = len(df_batch)
            
            missing = [c for c in required_cols if c not in df_batch.columns]
            if missing:
                st.error(f"Gagal memproses! Kolom berikut hilang dari CSV: **{', '.join(missing)}**")
            else:
                # ── ENGINE PREDIKSI & EKSTRAKSI PROBABILITAS ──
                X_b = df_batch[required_cols].fillna(df[required_cols].mean())
                
                if "Semua Fitur" in batch_model:
                    preds = dt_all.predict(X_b)
                    probas = dt_all.predict_proba(X_b)
                    classes = dt_all.classes_
                elif "Naive" in batch_model:
                    X_b_full = df_batch[ALL_FEATURES].fillna(df[ALL_FEATURES].mean())
                    X_b_scaled = scaler.transform(X_b_full)
                    preds = nb.predict(X_b_scaled)
                    probas = nb.predict_proba(X_b_scaled)
                    classes = nb.classes_
                else:
                    preds = dt_rfe.predict(X_b)
                    probas = dt_rfe.predict_proba(X_b)
                    classes = dt_rfe.classes_

                # ── PERHITUNGAN INDEKS EVALUASI INTERNAL (DIAGNOSTIK) ──
                max_probs = np.max(probas, axis=1)
                sorted_probas = np.sort(probas, axis=1)
                
                # Margin = Selisih probabilitas peringkat 1 dan peringkat 2
                if sorted_probas.shape[1] > 1:
                    margins = sorted_probas[:, -1] - sorted_probas[:, -2]
                    # Cari kelas alternatif (runner-up)
                    idx_sort = np.argsort(probas, axis=1)
                    alt_classes = classes[idx_sort[:, -2]]
                else:
                    margins = np.ones(total_rows)
                    alt_classes = ["None"] * total_rows

                # Deteksi Baris Sulit/Ambigu (Treshold: Margin Kepercayaan < 15% atau Max Prob < 55%)
                is_hard = (margins < 0.15) | (max_probs < 0.55)
                total_hard = int(np.sum(is_hard))
                pct_hard = (total_hard / total_rows) * 100

                # Deteksi Data Outlier/Anomali (Nilai input di luar batas min/max data historis training)
                is_outlier = np.zeros(total_rows, dtype=bool)
                for col in required_cols:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    # Tandai jika ada fitur yang over-extreme
                    is_outlier |= (X_b[col] < min_val * 0.8) | (X_b[col] > max_val * 1.2)
                total_outliers = int(np.sum(is_outlier))

                # Inject hasil analisa ke dataframe utama
                df_eval = df_batch.copy()
                df_eval['Prediksi_Utama'] = preds
                df_eval['Kelas_Alternatif'] = alt_classes
                df_eval['Confidence_Score'] = max_probs
                df_eval['Margin_Ambiguitas'] = margins
                df_eval['Status_Baris'] = np.where(is_outlier, '🚨 ANOMALI OUTLIER', np.where(is_hard, '⚠️ SULIT / AMBIGU', '✅ AMAN (CONFIDENT)'))

                # ── DISPLAY VISUAL 1: RINGKASAN SUB-SISTEM EVALUASI ──
                st.markdown('<div class="sec-head">📈 Ringkasan Kesehatan Prediksi Batch <div class="sec-line"></div></div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                
                m1.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL DATA PROSES</div><div class="metric-val">{total_rows:,}</div><div class="metric-sub">Baris terdaftar</div></div>', unsafe_allow_html=True)
                
                c_hard = "#166534" if pct_hard < 10 else ("#92400e" if pct_hard < 25 else "#b00020")
                m2.markdown(f'<div class="metric-card"><div class="metric-label">BARIS SULIT / AMBIGU</div><div class="metric-val" style="color:{c_hard}">{total_hard:,}</div><div class="metric-sub">{pct_hard:.1f}% Ambiguitas Tinggi</div></div>', unsafe_allow_html=True)
                
                c_out = "#166534" if total_outliers == 0 else "#b00020"
                m3.markdown(f'<div class="metric-card"><div class="metric-label">DETEKSI OUTLIER LAHAN</div><div class="metric-val" style="color:{c_out}">{total_outliers:,}</div><div class="metric-sub">Data di luar tren historis</div></div>', unsafe_allow_html=True)
                
                avg_conf = np.mean(max_probs) * 100
                m4.markdown(f'<div class="metric-card"><div class="metric-label">RATA-RATA CONFIDENCE</div><div class="metric-val">{avg_conf:.1f}%</div><div class="metric-sub">Keandalan model rata-rata</div></div>', unsafe_allow_html=True)

                # ── VISUAL 2: CHART DISTRIBUSI STATUS DATA ──
                st.markdown("<br>", unsafe_allow_html=True)
                col_c1, col_c2 = st.columns([1, 1])
                
                with col_c1:
                    status_counts = df_eval['Status_Baris'].value_counts()
                    fig_status = go.Figure(go.Pie(
                        labels=status_counts.index, values=status_counts.values, hole=0.5,
                        marker_colors=['#6aaa64', '#e65c00', '#b00020'], textinfo='label+percent'
                    ))
                    fig_status.update_layout(height=260, margin=dict(l=10,r=10,t=30,b=10), title="Segmentasi Kualitas Prediksi Data", font_family="Syne")
                    st.plotly_chart(fig_status, use_container_width=True)
                    
                with col_c2:
                    # Klasifikasi hasil akhir untuk data Confident vs Ambigu
                    fig_compare = px.histogram(df_eval, x="Prediksi_Utama", color="Status_Baris", barmode="group",
                                               color_discrete_map={'✅ AMAN (CONFIDENT)': '#6aaa64', '⚠️ SULIT / AMBIGU': '#e65c00', '🚨 ANOMALI OUTLIER': '#b00020'},
                                               template="simple_white", title="Distribusi Prediksi Berdasarkan Status Kepercayaan")
                    fig_compare.update_layout(height=260, margin=dict(l=10,r=10,t=40,b=10), font_family="Syne", legend=dict(title=""))
                    st.plotly_chart(fig_compare, use_container_width=True)

                # ── SECTION 3: BEDAH DATA SULIT / AMBIGU ──
                st.markdown('<div class="sec-head">🔍 Investigasi Baris Paling Ambigu (Borderline Analysis) <div class="sec-line"></div></div>', unsafe_allow_html=True)
                
                if total_hard > 0:
                    st.warning(f"Model mendeteksi ada **{total_hard}** baris data yang berada di 'borderline' (batas keputusan). Ini terjadi karena probabilitas antar kelas sangat ketat.")
                    
                    df_hard_rows = df_eval[is_hard].sort_values(by='Margin_Ambiguitas', ascending=True).head(20)
                    
                    # Tampilkan tabel data tersulit dengan kolom khusus analisis probabilitas
                    display_cols = required_cols + ['Prediksi_Utama', 'Kelas_Alternatif', 'Confidence_Score', 'Margin_Ambiguitas']
                    st.markdown("**Top 20 Baris Data dengan Tingkat Kebingungan Tertinggi:**")
                    st.dataframe(
                        df_hard_rows[display_cols].style.format({
                            'Confidence_Score': '{:.1%}',
                            'Margin_Ambiguitas': '{:.1%}'
                        }), use_container_width=True
                    )
                    
                    # Analisis Karakteristik Fitur Penyebab Ambiguitas
                    st.markdown("💡 **Kenapa baris di atas sulit diprediksi oleh Machine Learning?**")
                    
                    # Hitung rata-rata fitur data sulit vs data yakin
                    mean_easy = df_eval[~is_hard][required_cols].mean()
                    mean_hard = df_eval[is_hard][required_cols].mean()
                    
                    df_profile = pd.DataFrame({
                        'Fitur Lahan': required_cols,
                        'Rata-rata (Data Aman/Yakin)': mean_easy.values,
                        'Rata-rata (Data Sulit/Ambigu)': mean_hard.values
                    })
                    
                    # Tampilkan deviasi nilai
                    with st.expander("📊 Lihat Deviasi Komparasi Nilai Fitur (Aman vs Ambigu)"):
                        st.write("Jika nilai rata-rata pada data sulit melenceng jauh dari data aman, fitur tersebut kemungkinan besar merupakan pemicu ketidakpastian model.")
                        st.dataframe(df_profile.style.highlight_max(axis=1, color="#ffebee"))
                        
                        # Buat grafik deviasi rasio untuk mempermudah pemahaman pengguna
                        df_profile['Rasio_Perubahan'] = (df_profile['Rata-rata (Data Sulit/Ambigu)'] / df_profile['Rata-rata (Data Aman/Yakin)']) - 1
                        fig_dev = px.bar(df_profile, x='Rasio_Perubahan', y='Fitur Lahan', orientation='h',
                                         title="Faktor Deviasi Fitur Lahan yang Membingungkan Model (Aman vs Ambigu)",
                                         color='Rasio_Perubahan', color_continuous_scale=px.colors.diverging.Curl)
                        fig_dev.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10))
                        st.plotly_chart(fig_dev, use_container_width=True)
                else:
                    st.success("Sempurna! Tidak ditemukan baris data yang ambigu. Model sangat percaya diri dengan seluruh baris data yang diunggah.")

                # ── SECTION 4: DETEKSI ERROR & OUTLIER DETECTOR ──
                st.markdown('<div class="sec-head">🚨 Deteksi Outlier Ekstrem & Anomali Sensor <div class="sec-line"></div></div>', unsafe_allow_html=True)
                
                if total_outliers > 0:
                    st.error(f"Bahaya! Ditemukan **{total_outliers}** baris data dengan indikasi **Anomali/Outlier**. Data ini memiliki nilai yang tidak logis atau berada di luar jangkauan historis training set asli.")
                    df_outliers = df_eval[is_outlier].head(20)
                    st.dataframe(df_outliers[required_cols + ['Status_Baris']], use_container_width=True)
                    st.info("ℹ️ **Rekomendasi Tindakan:** Periksa sensor IoT di lapangan atau pastikan tidak ada kesalahan ketik unit/satuan data pada baris-baris tersebut sebelum mengambil tindakan operasional.")
                else:
                    st.success("Aman! Seluruh sampel input berada dalam batas wajar distribusi variabel sensor tanaman (Tidak ada data anomali).")

                # ── SECTION 5: DOWNLOAD OUTPUT LENGKAP DENGAN METADATA DIAGNOSTIK ──
                st.markdown('<div class="sec-head">🗂️ Unduh Hasil Analisis Komprehensif <div class="sec-line"></div></div>', unsafe_allow_html=True)
                st.write("Preview hasil data akhir beserta metadata hasil audit data mining (50 baris pertama):")
                
                # Format output final tabel
                final_cols_order = ['Status_Baris', 'Prediksi_Utama', 'Confidence_Score', 'Kelas_Alternatif', 'Margin_Ambiguitas'] + required_cols
                st.dataframe(df_eval[final_cols_order].head(50), use_container_width=True, hide_index=True)
                
                csv_out = df_eval.to_csv(index=False)
                st.download_button(
                    "⬇️ Unduh Hasil Audit Diagnostik Lengkap (.CSV)",
                    data=csv_out,
                    file_name="crop_yield_batch_diagnostic_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error memproses file: {e}. Pastikan file format CSV valid.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem; padding: 1rem 1.5rem; background: #2d1f0f; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: #a09070;">
  <span>🌾 <b style="color:#f0c040">CropYield Intelligence</b> — Data Mining Project</span>
  <span>Dataset: 102.675 sampel · 3 Model · Scikit-learn</span>
</div>
""", unsafe_allow_html=True)