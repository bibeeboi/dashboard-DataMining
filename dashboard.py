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
    # Recreate yield category using quantile-based logic (approximate)
    q33 = df['rainfall'].quantile(0.33)
    q66 = df['rainfall'].quantile(0.66)
    # Simple proxy label for viz (not ground truth)
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
    'High':   '#2e7d32',
    'Medium': '#f57f17',
    'Low':    '#c62828',
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
tab1, tab2, tab3 = st.tabs(["📊  Overview Dataset", "🤖  Prediksi Yield", "🔬  Perbandingan Model"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
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

    # Distribution charts — 2 rows x 3 cols
    st.markdown('<div class="sec-head">📈 Distribusi Fitur Utama <div class="sec-line"></div></div>', unsafe_allow_html=True)

    feat_groups = [
        ['N', 'P', 'K'],
        ['temperature', 'humidity', 'rainfall'],
    ]

    for group in feat_groups:
        cols = st.columns(3)
        for col, feat in zip(cols, group):
            meta = FEATURE_META[feat]
            fig = px.histogram(
                df.sample(5000, random_state=42), x=feat,
                nbins=40,
                color_discrete_sequence=['#6aaa64'],
                template='simple_white',
            )
            fig.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text=f"{meta['emoji']} {meta['label']}", font_size=12, font_color='#2d1f0f'),
                xaxis_title=meta['unit'],
                yaxis_title='',
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family='Syne',
            )
            fig.update_xaxis(showgrid=False)
            fig.update_yaxis(showgrid=True, gridcolor='#f0ece0')
            col.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap + scatter
    st.markdown('<div class="sec-head">🔗 Korelasi Antar Fitur <div class="sec-line"></div></div>', unsafe_allow_html=True)
    col_heat, col_scatter = st.columns([1.2, 1])

    with col_heat:
        corr = df[ALL_FEATURES].corr()
        fig_heat = px.imshow(
            corr,
            color_continuous_scale=['#c62828', '#faf6ee', '#2e7d32'],
            zmin=-1, zmax=1,
            text_auto='.2f',
            template='simple_white',
        )
        fig_heat.update_layout(
            height=360,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='white',
            font=dict(family='DM Mono', size=9),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_scatter:
        sample = df.sample(3000, random_state=99)
        # Simple proxy: color by rainfall tertile
        sample = sample.copy()
        sample['Yield_Cat'] = pd.qcut(sample['rainfall'], q=3, labels=['Low', 'Medium', 'High'])
        fig_sc = px.scatter(
            sample, x='temperature', y='humidity',
            color='Yield_Cat',
            color_discrete_map=COLORS,
            opacity=0.55,
            template='simple_white',
            labels={'temperature': 'Suhu (°C)', 'humidity': 'Kelembaban (%)'},
            title='🌡️ Suhu vs Kelembaban',
        )
        fig_sc.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='white',
            font_family='Syne',
            legend=dict(title='Estimasi Yield', orientation='h', y=-0.15),
            title_font_size=13,
        )
        fig_sc.update_traces(marker_size=4)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Feature stats table
    with st.expander("📋 Statistik Deskriptif Lengkap"):
        styled = df[ALL_FEATURES].describe().T.round(2)
        styled.index = [f"{FEATURE_META[c]['emoji']} {FEATURE_META[c]['label']}" for c in styled.index]
        st.dataframe(styled, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown('<div class="sec-head">⚙️ Input Parameter Lahan <div class="sec-line"></div></div>', unsafe_allow_html=True)

        model_choice = st.selectbox(
            "Pilih Model",
            ["Decision Tree (Semua Fitur)", "Decision Tree (RFE — 5 Fitur)", "Naive Bayes + SMOTE (RFE)"],
            help="RFE = Recursive Feature Elimination, hanya gunakan 5 fitur terpilih"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Determine which features to show
        if "Semua Fitur" in model_choice:
            active_features = ALL_FEATURES
            note = None
        elif "Naive" in model_choice:
            active_features = ALL_FEATURES
            note = f"Naive Bayes menggunakan semua 11 fitur (di-scale), output 2 kelas: **High / Low**"
        else:
            active_features = RFE_FEATURES
            note = f"Model ini hanya memerlukan fitur RFE: **{', '.join(RFE_FEATURES)}**"

        if note:
            st.info(note)

        # Default values = dataset mean
        defaults = {f: float(df[f].mean()) for f in ALL_FEATURES}
        slider_ranges = {
            'N':              (0,   139,   1),
            'P':              (5,   99,    1),
            'K':              (5,   149,   1),
            'pH':             (4.5, 8.0,   0.05),
            'temperature':    (15,  40,    0.5),
            'humidity':       (30,  90,    1.0),
            'rainfall':       (400, 3000,  10.0),
            'Soil_Moisture':  (14,  70,    1.0),
            'Wind_speed':     (2,   25,    0.5),
            'Sunshine_hours': (3,   12,    0.25),
            'Organic_Carbon': (0.2, 2.0,   0.05),
        }

        inputs = {}
        c_a, c_b = st.columns(2)
        for i, feat in enumerate(active_features):
            meta = FEATURE_META[feat]
            rng = slider_ranges[feat]
            col = c_a if i % 2 == 0 else c_b
            val = col.slider(
                f"{meta['emoji']} {meta['label']} ({meta['unit']})" if meta['unit'] else f"{meta['emoji']} {meta['label']}",
                min_value=rng[0], max_value=rng[1],
                value=round(defaults[feat] / rng[2]) * rng[2],
                step=rng[2],
                key=f"sl_{feat}"
            )
            inputs[feat] = val

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Prediksi Yield", use_container_width=True)

    with col_result:
        st.markdown('<div class="sec-head">🎯 Hasil Prediksi <div class="sec-line"></div></div>', unsafe_allow_html=True)

        if predict_btn:
            try:
                X_all = pd.DataFrame([{f: inputs[f] for f in ALL_FEATURES}])
                if "Semua Fitur" in model_choice:
                    pred = dt_all.predict(X_all)[0]
                    proba = dt_all.predict_proba(X_all)[0]
                    classes = dt_all.classes_
                    model_name = "Decision Tree (All Features)"
                elif "Naive" in model_choice:
                    # NB: scale all 11 features, then predict
                    X_scaled = scaler.transform(X_all)
                    pred = nb.predict(X_scaled)[0]
                    proba = nb.predict_proba(X_scaled)[0]
                    classes = nb.classes_
                    model_name = "Naive Bayes + SMOTE"
                else:
                    X_rfe = pd.DataFrame([{f: inputs[f] for f in RFE_FEATURES}])
                    pred = dt_rfe.predict(X_rfe)[0]
                    proba = dt_rfe.predict_proba(X_rfe)[0]
                    classes = dt_rfe.classes_
                    model_name = "Decision Tree (RFE Optimized)"

                # Result card
                st.markdown(f"""
                <div class="predict-result result-{pred.upper()}">
                  <div class="result-label">PREDIKSI CROP YIELD</div>
                  <div class="result-value">{pred.upper()}</div>
                  <div style="font-size:0.8rem;color:#666">via <b>{model_name}</b></div>
                </div>""", unsafe_allow_html=True)

                # Confidence bar chart
                st.markdown("<br>", unsafe_allow_html=True)
                fig_prob = go.Figure()
                bar_colors = [COLORS.get(c, '#888') for c in classes]
                fig_prob.add_trace(go.Bar(
                    x=[f"{c}" for c in classes],
                    y=[p * 100 for p in proba],
                    marker_color=bar_colors,
                    text=[f"{p*100:.1f}%" for p in proba],
                    textposition='outside',
                ))
                fig_prob.update_layout(
                    height=240,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis=dict(range=[0, 115], title='Probabilitas (%)', showgrid=True, gridcolor='#f0ece0'),
                    xaxis=dict(title='Kelas Yield'),
                    title=dict(text='📊 Confidence per Kelas', font_size=12),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_family='Syne',
                    showlegend=False,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # Input summary
                with st.expander("📋 Ringkasan Input"):
                    for feat in (ALL_FEATURES if "Semua" in model_choice else RFE_FEATURES):
                        meta = FEATURE_META[feat]
                        st.markdown(f"**{meta['emoji']} {meta['label']}**: `{inputs[feat]} {meta['unit']}`")

            except Exception as e:
                st.error(f"Prediksi gagal: {e}")
        else:
            st.markdown("""
            <div style="
                background: #f7f3ea;
                border-radius: 14px;
                padding: 2.5rem 2rem;
                text-align: center;
                border: 1.5px dashed #c8bfa0;
                color: #8a7a60;
            ">
                <div style="font-size:3rem">🌾</div>
                <div style="font-weight:700;margin-top:0.8rem">Atur parameter lahan</div>
                <div style="font-size:0.85rem;margin-top:0.4rem">Pilih model & atur slider, lalu klik <b>Prediksi Yield</b></div>
            </div>""", unsafe_allow_html=True)

        # Feature importance note
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">⭐ Fitur Terpilih RFE <div class="sec-line"></div></div>', unsafe_allow_html=True)
        rfe_cols = st.columns(len(RFE_FEATURES))
        rfe_icons = {'N': '🧪', 'K': '🧪', 'temperature': '🌡️', 'humidity': '💧', 'rainfall': '🌧️'}
        for col, feat in zip(rfe_cols, RFE_FEATURES):
            meta = FEATURE_META[feat]
            col.markdown(f"""
            <div style="background:white;border-radius:10px;padding:0.8rem;text-align:center;border:1.5px solid #e8dfc8;">
              <div style="font-size:1.5rem">{meta['emoji']}</div>
              <div style="font-size:0.7rem;font-weight:700;color:#3d5a3e;margin-top:0.3rem">{feat}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-head">🔬 Arsitektur Model <div class="sec-line"></div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    model_info = [
        {
            "name": "Decision Tree",
            "sub": "All Features",
            "emoji": "🌳",
            "features": "11 fitur",
            "classes": "High, Medium, Low",
            "notes": "Baseline model — semua fitur digunakan tanpa seleksi",
            "color": "#3d5a3e",
        },
        {
            "name": "Decision Tree",
            "sub": "RFE Optimized",
            "emoji": "✂️",
            "features": "5 fitur (N, K, temp, humidity, rainfall)",
            "classes": "High, Medium, Low",
            "notes": "Fitur diseleksi dengan Recursive Feature Elimination",
            "color": "#6aaa64",
        },
        {
            "name": "Naive Bayes",
            "sub": "+ SMOTE",
            "emoji": "⚖️",
            "features": "5 fitur RFE + StandardScaler",
            "classes": "High, Low",
            "notes": "Data dibalance dengan SMOTE; output hanya 2 kelas",
            "color": "#f57f17",
        },
    ]
    for col, m in zip([c1, c2, c3], model_info):
        col.markdown(f"""
        <div style="
            background: white;
            border-radius: 14px;
            padding: 1.4rem;
            border: 1.5px solid #e8dfc8;
            height: 100%;
        ">
          <div style="font-size:2rem">{m['emoji']}</div>
          <div style="font-size:1.1rem;font-weight:800;color:{m['color']};margin-top:0.5rem">{m['name']}</div>
          <div style="font-size:0.75rem;font-weight:700;color:#888;letter-spacing:0.05em">{m['sub'].upper()}</div>
          <hr style="border-color:#f0ece0;margin:0.8rem 0">
          <div style="font-size:0.82rem;color:#555">
            <b>Fitur:</b> {m['features']}<br>
            <b>Kelas:</b> {m['classes']}<br><br>
            <i style="color:#888">{m['notes']}</i>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Simulated evaluation metrics (typical DM project values)
    st.markdown('<div class="sec-head">📊 Performa Model (Estimasi) <div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.caption("Catatan: Metrik di bawah adalah ilustrasi representatif — update dengan nilai aktual dari notebook kamu.")

    metrics_data = {
        'Model': ['DT All Features', 'DT RFE Optimized', 'NB + SMOTE'],
        'Accuracy': [0.87, 0.84, 0.79],
        'Precision': [0.86, 0.83, 0.78],
        'Recall': [0.87, 0.84, 0.79],
        'F1-Score': [0.86, 0.83, 0.78],
        'Fitur Digunakan': [11, 5, 5],
    }
    df_metrics = pd.DataFrame(metrics_data)

    col_table, col_radar = st.columns([1, 1.2])
    with col_table:
        st.dataframe(
            df_metrics.style.background_gradient(
                subset=['Accuracy', 'F1-Score'],
                cmap='YlGn'
            ).format({'Accuracy': '{:.0%}', 'Precision': '{:.0%}', 'Recall': '{:.0%}', 'F1-Score': '{:.0%}'}),
            use_container_width=True, hide_index=True
        )

    with col_radar:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig_radar = go.Figure()
        model_colors = ['#3d5a3e', '#6aaa64', '#f57f17']
        for i, row in df_metrics.iterrows():
            vals = [row[c] for c in categories]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill='toself', name=row['Model'],
                line_color=model_colors[i],
                fillcolor=model_colors[i],
                opacity=0.3,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0.7, 0.95], tickformat='.0%')),
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='white',
            font_family='Syne',
            legend=dict(orientation='h', y=-0.1),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Feature importance DT all
    st.markdown('<div class="sec-head">🏆 Feature Importance — Decision Tree All Features <div class="sec-line"></div></div>', unsafe_allow_html=True)
    importances = dt_all.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Fitur': [FEATURE_META[f]['label'] for f in ALL_FEATURES],
        'Importance': importances,
        'Emoji': [FEATURE_META[f]['emoji'] for f in ALL_FEATURES],
    }).sort_values('Importance', ascending=True)

    fig_imp = px.bar(
        feat_imp_df, x='Importance', y='Fitur',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#e8f5e9', '#2e7d32'],
        template='simple_white',
        text=feat_imp_df['Importance'].map(lambda x: f'{x:.3f}'),
    )
    fig_imp.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='white',
        font_family='Syne',
        coloraxis_showscale=False,
        xaxis_title='Importance Score',
        yaxis_title='',
    )
    fig_imp.update_traces(textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)

    # DT tree depth info
    with st.expander("ℹ️ Detail Hyperparameter Model"):
        c_a, c_b = st.columns(2)
        c_a.markdown(f"""
        **🌳 DT All Features**
        - Max Depth: `{dt_all.get_depth()}`
        - Leaves: `{dt_all.get_n_leaves()}`
        - Criterion: `{dt_all.criterion}`
        - Classes: `{', '.join(dt_all.classes_)}`
        """)
        c_b.markdown(f"""
        **✂️ DT RFE Optimized**
        - Max Depth: `{dt_rfe.get_depth()}`
        - Leaves: `{dt_rfe.get_n_leaves()}`
        - Criterion: `{dt_rfe.criterion}`
        - Classes: `{', '.join(dt_rfe.classes_)}`
        """)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top:2rem;
    padding: 1rem 1.5rem;
    background: #2d1f0f;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: #a09070;
">
  <span>🌾 <b style="color:#f0c040">CropYield Intelligence</b> — Data Mining Project</span>
  <span>Dataset: 102.675 sampel · 3 Model · Scikit-learn</span>
</div>
""", unsafe_allow_html=True)
