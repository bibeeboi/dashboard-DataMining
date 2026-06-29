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
                xaxis_title=dict(text=meta['unit'], font=dict(color='#2d1f0f', size=11)),
                yaxis_title='',
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Syne', color='#2d1f0f'),
            )
            fig.update_xaxes(showgrid=False, color='#2d1f0f', tickfont=dict(color='#555'))
            fig.update_yaxes(showgrid=True, gridcolor='#f0ece0', color='#2d1f0f', tickfont=dict(color='#555'))
            col.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap + scatter
    st.markdown('<div class="sec-head">🔗 Korelasi Antar Fitur <div class="sec-line"></div></div>', unsafe_allow_html=True)
    col_heat, col_scatter = st.columns([1.2, 1])

    with col_heat:
        corr = df[ALL_FEATURES].corr()
        fig_heat = px.imshow(
            corr,
            color_continuous_scale=['#b00020', '#f5f5f5', '#1a7a1a'],
            zmin=-1, zmax=1,
            text_auto='.2f',
            template='simple_white',
        )
        fig_heat.update_layout(
            height=420,
            margin=dict(l=120, r=20, t=20, b=120),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='DM Mono', size=9, color='#2d1f0f'),
            coloraxis_showscale=False,
        )
        fig_heat.update_xaxes(tickangle=45, tickfont=dict(size=9, color='#2d1f0f'))
        fig_heat.update_yaxes(tickfont=dict(size=9, color='#2d1f0f'))
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
            opacity=0.75,
            template='simple_white',
            labels={'temperature': 'Suhu (°C)', 'humidity': 'Kelembaban (%)', 'Yield_Cat': 'Estimasi Yield'},
            title='🌡️ Suhu vs Kelembaban',
        )
        fig_sc.update_layout(
            height=420,
            margin=dict(l=50, r=20, t=50, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Syne', color='#2d1f0f'),
            legend=dict(
                title='Estimasi Yield',
                orientation='h',
                y=-0.18, x=0.5, xanchor='center',
                font=dict(color='#2d1f0f'),
            ),
            title_font=dict(size=13, color='#2d1f0f'),
            xaxis=dict(color='#2d1f0f', gridcolor='#f0ece0'),
            yaxis=dict(color='#2d1f0f', gridcolor='#f0ece0'),
        )
        fig_sc.update_traces(marker_size=5)
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
            use_float = isinstance(rng[2], float)
            if use_float:
                mn, mx, stp = float(rng[0]), float(rng[1]), float(rng[2])
                dv = float(round(defaults[feat] / stp) * stp)
                dv = max(mn, min(mx, dv))
            else:
                mn, mx, stp = int(rng[0]), int(rng[1]), int(rng[2])
                dv = max(int(rng[0]), min(int(rng[1]), int(round(defaults[feat]))))
            val = col.slider(
                f"{meta['emoji']} {meta['label']} ({meta['unit']})" if meta['unit'] else f"{meta['emoji']} {meta['label']}",
                min_value=mn, max_value=mx,
                value=dv,
                step=stp,
                key=f"sl_{feat}"
            )
            inputs[feat] = val

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Prediksi Yield", use_container_width=True)

    with col_result:
        st.markdown('<div class="sec-head">🎯 Hasil Prediksi <div class="sec-line"></div></div>', unsafe_allow_html=True)

        if predict_btn:
            try:
                full_inputs = {f: defaults[f] for f in ALL_FEATURES}
                full_inputs.update(inputs) 
                X_all = pd.DataFrame([{f: full_inputs[f] for f in ALL_FEATURES}])
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
                    X_rfe = pd.DataFrame([{f: full_inputs[f] for f in RFE_FEATURES}])
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
        df_display = df_metrics.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            df_display[col] = df_display[col].map('{:.0%}'.format)
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={c: st.column_config.TextColumn(c) for c in df_display.columns}
        )

    with col_radar:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig_radar = go.Figure()
        model_colors = ['#166534', '#92400e', '#881337']
        for i, row in df_metrics.iterrows():
            vals = [row[c] for c in categories]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill='toself', name=row['Model'],
                line_color=model_colors[i],
                fillcolor=model_colors[i],
                opacity=0.75,
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0.7, 0.95], tickformat='.0%', color='#2d1f0f'),
                angularaxis=dict(color='#2d1f0f'),
            ),
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Syne', color='#2d1f0f'),
            legend=dict(orientation='h', y=-0.1, font=dict(color='#2d1f0f', size=12)),
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
            plot_bgcolor='white',
            font=dict(family='Syne', color='#2d1f0f'),
            coloraxis_showscale=False,
            xaxis_title='Importance Score',
            yaxis_title='',
        )
    fig_imp.update_xaxes(color='#2d1f0f', tickfont=dict(color='#555'), gridcolor='#f0ece0')
    fig_imp.update_yaxes(color='#2d1f0f', tickfont=dict(color='#2d1f0f'))
    
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
# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH PREDIKSI
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-head">📂 Batch Prediksi dari CSV <div class="sec-line"></div></div>', unsafe_allow_html=True)

    col_up, col_cfg = st.columns([1.2, 1])

    with col_up:
        st.markdown("**Format CSV yang dibutuhkan:**")
        
        batch_model = st.selectbox(
            "Model untuk Batch Prediksi",
            ["Decision Tree (Semua Fitur)", "Decision Tree (RFE — 5 Fitur)", "Naive Bayes + SMOTE"],
            key="batch_model"
        )
        
        if "RFE" in batch_model:
            required_cols = RFE_FEATURES
        else:
            required_cols = ALL_FEATURES
            
        st.info(f"Kolom yang dibutuhkan: **{', '.join(required_cols)}**")
        
        # Template download
        template_df = pd.DataFrame([{f: round(float(df[f].mean()), 2) for f in required_cols}])
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Template CSV",
            data=csv_template,
            file_name="template_batch.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload file CSV", type=["csv"], label_visibility="collapsed")

    with col_cfg:
        st.markdown("**Panduan:**")
        st.markdown("""
        <div style="background:white;border-radius:12px;padding:1.2rem;border:1.5px solid #e8dfc8;font-size:0.85rem;color:#444;">
        1️⃣ Download template CSV dulu<br><br>
        2️⃣ Isi data sesuai kolom yang tersedia<br><br>
        3️⃣ Upload file CSV kamu<br><br>
        4️⃣ Hasil prediksi langsung muncul & bisa di-download
        </div>
        """, unsafe_allow_html=True)

    if uploaded is not None:
        try:
            df_batch = pd.read_csv(uploaded)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="sec-head">🔍 Hasil Prediksi — {len(df_batch)} baris <div class="sec-line"></div></div>', unsafe_allow_html=True)

            # Validasi kolom
            missing = [c for c in required_cols if c not in df_batch.columns]
            if missing:
                st.error(f"Kolom tidak lengkap: **{', '.join(missing)}**")
            else:
                # Prediksi
                if "Semua Fitur" in batch_model:
                    X_b = df_batch[ALL_FEATURES].fillna(df_batch[ALL_FEATURES].mean())
                    preds = dt_all.predict(X_b)
                    probas = dt_all.predict_proba(X_b)
                    classes = dt_all.classes_
                elif "Naive" in batch_model:
                    X_b = df_batch[ALL_FEATURES].fillna(df_batch[ALL_FEATURES].mean())
                    X_b_scaled = scaler.transform(X_b)
                    preds = nb.predict(X_b_scaled)
                    probas = nb.predict_proba(X_b_scaled)
                    classes = nb.classes_
                else:
                    X_b = df_batch[RFE_FEATURES].fillna(df_batch[RFE_FEATURES].mean())
                    preds = dt_rfe.predict(X_b)
                    probas = dt_rfe.predict_proba(X_b)
                    classes = dt_rfe.classes_

                # Tambah hasil ke dataframe
                df_result = df_batch.copy()
                df_result['Prediksi'] = preds
                for i, cls in enumerate(classes):
                    df_result[f'Prob_{cls}'] = [round(p[i]*100, 1) for p in probas]

                # Summary donut
               # ── RINGKASAN EKSEKUTIF ──
                col_sum, col_stats = st.columns([1, 1])

                with col_sum:
                    counts = pd.Series(preds).value_counts()
                    fig_donut = go.Figure(go.Pie(
                        labels=counts.index,
                        values=counts.values,
                        hole=0.55,
                        marker_colors=[COLORS.get(c, '#888') for c in counts.index],
                        textinfo='label+percent',
                        textfont=dict(color='#2d1f0f', size=12),
                    ))
                    fig_donut.update_layout(
                        height=260,
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor='white',
                        showlegend=False,
                        title=dict(text='Distribusi Kelas Yield', font=dict(color='#2d1f0f', size=13)),
                        font=dict(family='Syne', color='#2d1f0f'),
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                with col_stats:
                    total = len(preds)
                    for cls in classes:
                        n = (preds == cls).sum()
                        pct = n / total * 100
                        color_map = {'High': '#e8f5e9', 'Medium': '#fff8e1', 'Low': '#fce4ec'}
                        border_map = {'High': '#1a7a1a', 'Medium': '#e65c00', 'Low': '#b00020'}
                        text_map = {'High': '#1a7a1a', 'Medium': '#e65c00', 'Low': '#b00020'}
                        st.markdown(f"""
                        <div style="background:{color_map.get(cls,'#f5f5f5')};border-left:4px solid {border_map.get(cls,'#888')};
                            border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.6rem;">
                          <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#666;text-transform:uppercase">{cls} Yield</div>
                          <div style="font-size:1.6rem;font-weight:800;color:{text_map.get(cls,'#333')};font-family:'DM Mono',monospace">{n:,} <span style="font-size:0.9rem">baris</span></div>
                          <div style="font-size:0.8rem;color:#555">{pct:.1f}% dari total dataset</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── ANALISIS PROBABILITAS ──
                st.markdown('<div class="sec-head">📈 Analisis Confidence Prediksi <div class="sec-line"></div></div>', unsafe_allow_html=True)

                col_conf1, col_conf2 = st.columns(2)

                # Distribusi confidence per kelas
                prob_df = pd.DataFrame(probas, columns=[f'Prob_{c}' for c in classes])
                prob_df['Prediksi'] = preds

                with col_conf1:
                    # Rata-rata confidence per kelas prediksi
                    avg_conf = []
                    for cls in classes:
                        mask = prob_df['Prediksi'] == cls
                        if mask.sum() > 0:
                            avg = prob_df.loc[mask, f'Prob_{cls}'].mean() * 100
                            avg_conf.append({'Kelas': cls, 'Avg Confidence (%)': round(avg, 1)})
                    
                    df_conf = pd.DataFrame(avg_conf)
                    fig_conf = px.bar(
                        df_conf, x='Kelas', y='Avg Confidence (%)',
                        color='Kelas',
                        color_discrete_map=COLORS,
                        template='simple_white',
                        title='Rata-rata Confidence per Kelas',
                        text='Avg Confidence (%)',
                    )
                    fig_conf.update_layout(
                        height=250, margin=dict(l=10,r=10,t=40,b=10),
                        paper_bgcolor='white', plot_bgcolor='white',
                        font=dict(family='Syne', color='#2d1f0f'),
                        showlegend=False,
                        yaxis=dict(range=[0,110]),
                        title_font=dict(size=12, color='#2d1f0f'),
                    )
                    fig_conf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig_conf, use_container_width=True)

                with col_conf2:
                    # Confidence tinggi vs rendah
                    max_prob = np.max(probas, axis=1)
                    high_conf = (max_prob >= 0.8).sum()
                    mid_conf = ((max_prob >= 0.6) & (max_prob < 0.8)).sum()
                    low_conf = (max_prob < 0.6).sum()

                    fig_cert = go.Figure(go.Bar(
                        x=['≥80% (Tinggi)', '60–80% (Sedang)', '<60% (Rendah)'],
                        y=[high_conf, mid_conf, low_conf],
                        marker_color=['#1a7a1a', '#e65c00', '#b00020'],
                        text=[f'{high_conf:,}', f'{mid_conf:,}', f'{low_conf:,}'],
                        textposition='outside',
                    ))
                    fig_cert.update_layout(
                        height=250, margin=dict(l=10,r=10,t=40,b=10),
                        paper_bgcolor='white', plot_bgcolor='white',
                        font=dict(family='Syne', color='#2d1f0f'),
                        title=dict(text='Tingkat Kepercayaan Prediksi', font=dict(size=12, color='#2d1f0f')),
                        yaxis=dict(gridcolor='#f0ece0'),
                        showlegend=False,
                    )
                    fig_cert.update_xaxes(color='#2d1f0f')
                    fig_cert.update_yaxes(color='#2d1f0f')
                    st.plotly_chart(fig_cert, use_container_width=True)

                # ── INSIGHT OTOMATIS ──
                st.markdown('<div class="sec-head">💡 Insight Otomatis <div class="sec-line"></div></div>', unsafe_allow_html=True)

                dominant = counts.idxmax()
                dominant_pct = counts.max() / total * 100
                high_conf_pct = high_conf / total * 100
                
                insights = []
                insights.append(f"📌 Kelas **{dominant}** mendominasi hasil prediksi dengan **{dominant_pct:.1f}%** dari total {total:,} sampel.")
                insights.append(f"✅ **{high_conf_pct:.1f}%** prediksi memiliki confidence ≥80% — model cukup yakin pada sebagian besar data.")
                
                if 'Low' in counts and counts.get('Low', 0) / total > 0.3:
                    insights.append(f"⚠️ Proporsi kelas **Low** cukup tinggi ({counts.get('Low',0)/total*100:.1f}%) — perlu perhatian khusus pada kondisi lahan.")
                if 'High' in counts and counts.get('High', 0) / total > 0.4:
                    insights.append(f"🌾 Mayoritas lahan diprediksi **High yield** — kondisi dataset secara umum mendukung hasil panen baik.")
                if low_conf / total > 0.2:
                    insights.append(f"🔍 **{low_conf/total*100:.1f}%** prediksi memiliki confidence rendah (<60%) — data ini perlu diverifikasi lebih lanjut.")

                for ins in insights:
                    st.markdown(f"""
                    <div style="background:white;border-radius:10px;padding:0.8rem 1.2rem;
                        margin-bottom:0.5rem;border:1.5px solid #e8dfc8;font-size:0.87rem;color:#2d1f0f;">
                        {ins}
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── TABEL SAMPEL ──
                st.markdown('<div class="sec-head">🗂️ Sampel Hasil (50 baris pertama) <div class="sec-line"></div></div>', unsafe_allow_html=True)
                df_result = df_batch.copy()
                df_result['Prediksi'] = preds
                for i, cls in enumerate(classes):
                    df_result[f'Prob_{cls} (%)'] = [round(p[i]*100, 1) for p in probas]
                st.dataframe(df_result.head(50), use_container_width=True, hide_index=True, height=280)

                # Download
                csv_out = df_result.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Hasil Lengkap CSV",
                    data=csv_out,
                    file_name="hasil_prediksi_batch.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error memproses file: {e}")
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
