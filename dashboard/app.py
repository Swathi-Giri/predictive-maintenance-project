"""
app.py — Streamlit Dashboard: Engine Health Monitor

A live, interactive dashboard that:
  - Monitors engine sensor data in real-time
  - Predicts Remaining Useful Life (RUL)
  - Shows health status with color-coded gauges
  - Simulates live engine operation
  - Displays sensor degradation trends

Run:
  streamlit run dashboard/app.py
  Opens at http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_train_data, cap_rul
from src.features import build_features, get_feature_columns, USEFUL_SENSORS
from src.predict import RULPredictor

# ── Page Config ──
st.set_page_config(
    page_title="🔧 Engine Health Monitor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache data and model loading ──
@st.cache_data(show_spinner="Loading engine data...")
def load_data():
    df = load_train_data()
    df = cap_rul(df)
    return df


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    return RULPredictor(
        model_path="models/best_model.pkl",
        features_path="models/feature_columns.json"
    )


def create_health_gauge(rul_value, health_info):
    """Create a gauge chart showing engine health."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rul_value,
        number={'suffix': ' cycles', 'font': {'size': 36}},
        title={'text': 'Remaining Useful Life', 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 125], 'tickwidth': 2},
            'bar': {'color': health_info['color'], 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 2,
            'steps': [
                {'range': [0, 15], 'color': '#ffcdd2'},    # Danger zone
                {'range': [15, 40], 'color': '#fff9c4'},    # Critical zone
                {'range': [40, 80], 'color': '#ffe0b2'},    # Warning zone
                {'range': [80, 125], 'color': '#c8e6c9'},   # Healthy zone
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': rul_value
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_sensor_chart(data, sensors, current_cycle):
    """Create sensor trend chart."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    for i, sensor in enumerate(sensors):
        fig.add_trace(go.Scatter(
            x=data['cycle'],
            y=data[sensor],
            mode='lines',
            name=sensor,
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.85
        ))

    fig.add_vline(
        x=current_cycle,
        line_dash="dash",
        line_color="red",
        annotation_text="📍 Now",
        annotation_font_color="red"
    )

    fig.update_layout(
        xaxis_title="Operating Cycle",
        yaxis_title="Sensor Value",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    return fig


def create_rul_comparison_chart(featured_data, feature_cols, predictor):
    """Show actual vs predicted RUL over the engine's life."""
    # Sample points to avoid slow rendering
    n_points = min(60, len(featured_data))
    indices = np.linspace(0, len(featured_data) - 1, n_points, dtype=int)

    records = []
    for idx in indices:
        row = featured_data.iloc[idx]
        pred = predictor.predict(featured_data[feature_cols].iloc[[idx]])[0]
        records.append({
            'cycle': int(row['cycle']),
            'Actual RUL': row['RUL'],
            'Predicted RUL': pred
        })

    rul_df = pd.DataFrame(records)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rul_df['cycle'], y=rul_df['Actual RUL'],
        mode='lines', name='Actual RUL',
        line=dict(color='#2196F3', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=rul_df['cycle'], y=rul_df['Predicted RUL'],
        mode='lines+markers', name='Predicted RUL',
        line=dict(color='#FF5722', width=2, dash='dot'),
        marker=dict(size=4)
    ))

    fig.add_hline(y=40, line_dash="dot", line_color="orange",
                  annotation_text="⚠️ Warning", annotation_position="top left")
    fig.add_hline(y=15, line_dash="dot", line_color="red",
                  annotation_text="🚨 Critical", annotation_position="top left")

    fig.update_layout(
        xaxis_title="Operating Cycle",
        yaxis_title="Remaining Useful Life (cycles)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    return fig


# ═══════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════

def main():
    # ── Header ──
    st.markdown('<div class="main-header">🔧 Turbofan Engine Health Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Real-time predictive maintenance dashboard — '
        'powered by ML on NASA C-MAPSS sensor data'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Load data ──
    try:
        train_df = load_data()
        predictor = load_model()
    except Exception as e:
        st.error(f"❌ Failed to load data or model: {e}")
        st.info("Run `python src/train.py` from the project root first!")
        return

    # ── Sidebar Controls ──
    st.sidebar.image("https://img.icons8.com/fluency/96/engine.png", width=60)
    st.sidebar.title("Controls")

    engine_ids = sorted(train_df['engine_id'].unique())
    engine_id = st.sidebar.selectbox("🔩 Select Engine", engine_ids, index=0)

    engine_data = train_df[train_df['engine_id'] == engine_id].copy()
    max_cycle = int(engine_data['cycle'].max())

    current_cycle = st.sidebar.slider(
        "⏱️ Current Cycle",
        min_value=1,
        max_value=max_cycle,
        value=int(max_cycle * 0.7),
        help="Drag to simulate different points in the engine's life"
    )

    st.sidebar.markdown("---")
    simulate = st.sidebar.button("▶️ Run Live Simulation", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.markdown(f"- **Engines:** {len(engine_ids)}")
    st.sidebar.markdown(f"- **This engine's life:** {max_cycle} cycles")
    st.sidebar.markdown(f"- **Sensors monitored:** {len(USEFUL_SENSORS)}")

    # ── Get data up to current cycle ──
    current_data = engine_data[engine_data['cycle'] <= current_cycle].copy()
    latest = current_data.iloc[-1]
    actual_rul = latest['RUL']

    # ── Build features and predict ──
    featured_data = build_features(current_data)
    feature_cols = get_feature_columns(featured_data)
    predicted_rul = predictor.predict(featured_data[feature_cols].iloc[[-1]])[0]
    health = predictor.get_health_status(predicted_rul)

    # ═══════════════════════════════════════
    # ROW 1: Key Metrics
    # ═══════════════════════════════════════

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔩 Engine", f"#{engine_id}")
    with col2:
        st.metric("⏱️ Cycle", f"{current_cycle} / {max_cycle}")
    with col3:
        delta_val = round(predicted_rul - actual_rul, 1)
        st.metric(
            "🎯 Predicted RUL",
            f"{predicted_rul:.0f} cycles",
            delta=f"Error: {delta_val:+.0f}",
            delta_color="inverse"
        )
    with col4:
        st.metric("📋 Actual RUL", f"{actual_rul:.0f} cycles")

    # Status banner
    status_colors = {
        "HEALTHY": ("🟢", "#d4edda", "#155724"),
        "WARNING": ("🟡", "#fff3cd", "#856404"),
        "CRITICAL": ("🔴", "#f8d7da", "#721c24"),
        "DANGER": ("🚨", "#f5c6cb", "#721c24"),
    }
    emoji, bg, fg = status_colors.get(health['status'], ("❓", "#e2e3e5", "#383d41"))

    st.markdown(
        f'<div style="background:{bg}; color:{fg}; padding:12px 20px; '
        f'border-radius:8px; font-size:1.1rem; margin:10px 0 20px 0;">'
        f'{emoji} <strong>Status: {health["status"]}</strong> — {health["action"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    # ═══════════════════════════════════════
    # ROW 2: Charts
    # ═══════════════════════════════════════

    tab1, tab2, tab3 = st.tabs(["📈 Sensor Trends", "🎯 RUL Tracking", "🏥 Health Gauge"])

    with tab1:
        selected_sensors = st.multiselect(
            "Choose sensors to monitor",
            USEFUL_SENSORS,
            default=['sensor_2', 'sensor_11', 'sensor_15', 'sensor_21'],
            key="sensor_select"
        )
        if selected_sensors:
            st.plotly_chart(
                create_sensor_chart(current_data, selected_sensors, current_cycle),
                use_container_width=True
            )
        else:
            st.info("Select at least one sensor above to see trends.")

    with tab2:
        st.plotly_chart(
            create_rul_comparison_chart(featured_data, feature_cols, predictor),
            use_container_width=True
        )

    with tab3:
        g_col1, g_col2 = st.columns([1, 1])
        with g_col1:
            st.plotly_chart(
                create_health_gauge(predicted_rul, health),
                use_container_width=True
            )
        with g_col2:
            st.markdown("### Health Zones")
            st.markdown("""
            | Zone | RUL | Action |
            |------|-----|--------|
            | 🟢 Healthy | > 80 cycles | Normal operation |
            | 🟡 Warning | 40–80 cycles | Schedule maintenance |
            | 🔴 Critical | 15–40 cycles | Maintain immediately |
            | 🚨 Danger | < 15 cycles | **STOP ENGINE** |
            """)

    # ═══════════════════════════════════════
    # ROW 3: Sensor Snapshot Table
    # ═══════════════════════════════════════

    with st.expander("📋 Current Sensor Values", expanded=False):
        snapshot = current_data.iloc[-1][USEFUL_SENSORS]
        snap_df = pd.DataFrame({
            'Sensor': snapshot.index,
            'Current Value': snapshot.values.round(4),
        })
        st.dataframe(snap_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════
    # LIVE SIMULATION
    # ═══════════════════════════════════════

    if simulate:
        st.markdown("---")
        st.subheader("🔴 Live Engine Simulation")
        st.caption("Watching engine degrade in real-time...")

        progress = st.progress(0)
        status_text = st.empty()
        live_chart = st.empty()

        # Start from 60% of engine life
        start = max(20, int(max_cycle * 0.6))
        sim_records = []

        for i, cyc in enumerate(range(start, max_cycle + 1)):
            cyc_data = engine_data[engine_data['cycle'] <= cyc]
            feat = build_features(cyc_data)
            fc = get_feature_columns(feat)
            pred = predictor.predict(feat[fc].iloc[[-1]])[0]
            actual = cyc_data.iloc[-1]['RUL']
            h = predictor.get_health_status(pred)

            sim_records.append({
                'cycle': cyc,
                'Predicted RUL': pred,
                'Actual RUL': actual
            })

            pct = (i + 1) / (max_cycle - start + 1)
            progress.progress(min(pct, 1.0))
            status_text.markdown(
                f"**Cycle {cyc}/{max_cycle}** | "
                f"Predicted: **{pred:.0f}** | "
                f"Actual: **{actual:.0f}** | "
                f"{h['emoji']} **{h['status']}**"
            )

            if len(sim_records) > 3:
                sim_df = pd.DataFrame(sim_records)
                fig = px.line(
                    sim_df, x='cycle', y=['Predicted RUL', 'Actual RUL'],
                    color_discrete_map={'Predicted RUL': '#FF5722', 'Actual RUL': '#2196F3'}
                )
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                live_chart.plotly_chart(fig, use_container_width=True)

            time.sleep(0.08)

        st.success("✅ Simulation complete — engine reached end of life!")

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#999; font-size:0.85rem;">'
        'Built with Python, Scikit-learn, FastAPI & Streamlit | '
        'Data: NASA C-MAPSS Turbofan Engine Degradation Dataset | '
        '<a href="https://github.com/YOUR_USERNAME/predictive-maintenance">GitHub</a>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # Change to project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(project_root)
    main()
