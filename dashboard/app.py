import streamlit as st
import requests
import numpy as np
import pandas as pd
import altair as alt

API_URL = "https://deadneurons.onrender.com"

st.set_page_config(
    page_title="DeadNeurons | Neural Decoder MLOps",
    layout="wide"
)

# ── Global Monochrome Theme ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap');

    :root {
        --bg: #050505;
        --panel: #0b0b0b;
        --panel-2: #101010;
        --panel-3: #141414;
        --border: #1f1f1f;
        --border-strong: #2a2a2a;
        --text: #f5f5f5;
        --text-soft: #bdbdbd;
        --text-muted: #6a6a6a;
        --text-faint: #444444;
        --white: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background: var(--bg);
        color: var(--text);
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1340px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syne', sans-serif !important;
        color: var(--white) !important;
        letter-spacing: -0.02em;
    }

    p, li, div, span, label {
        color: var(--text-soft);
    }

    a {
        color: #d8d8d8 !important;
        text-decoration: none !important;
    }

    a:hover {
        color: var(--white) !important;
    }

    .mono-kicker {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-faint);
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .section-divider {
        border-top: 1px solid var(--border);
        margin: 34px 0;
    }

    .soft-panel {
        background: linear-gradient(180deg, #0b0b0b 0%, #080808 100%);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 18px 20px;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #0a0a0a 100%);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 18px 20px;
        box-shadow: none;
    }

    [data-testid="metric-container"] label {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.68rem !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: var(--white) !important;
    }

    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 10px;
    }

    [data-testid="stTabs"] [role="tab"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.74rem !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: transparent !important;
        border: none !important;
        padding: 14px 20px !important;
        border-radius: 0 !important;
    }

    [data-testid="stTabs"] [role="tab"]:hover {
        color: #cfcfcf !important;
        background: transparent !important;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--white) !important;
        border-bottom: 2px solid var(--white) !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] th {
        background: #0d0d0d !important;
        color: var(--text-muted) !important;
        border-bottom: 1px solid var(--border) !important;
        font-family: 'DM Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.7rem !important;
    }

    [data-testid="stDataFrame"] td {
        background: #090909 !important;
        color: #d0d0d0 !important;
        border-bottom: 1px solid #121212 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: #0a0a0a !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--white) !important;
        border-radius: 2px !important;
        color: #d7d7d7 !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        font-family: 'DM Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.74rem;
        background: #111111 !important;
        color: var(--white) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 2px !important;
        padding: 0.7rem 1.1rem !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--white) !important;
        color: #050505 !important;
        border-color: var(--white) !important;
    }

    .stLinkButton a {
        font-family: 'DM Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.74rem;
        background: transparent !important;
        color: var(--white) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 2px !important;
        padding: 0.8rem 1rem !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .stLinkButton a:hover {
        background: var(--white) !important;
        color: #050505 !important;
        border-color: var(--white) !important;
    }

    /* Images */
    [data-testid="stImage"] img {
        border: 1px solid var(--border);
        border-radius: 2px;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.04em;
    }

    /* Markdown paragraphs */
    .stMarkdown p {
        color: var(--text-soft);
        line-height: 1.75;
    }

    /* Code */
    code, pre {
        background: #0b0b0b !important;
        border: 1px solid var(--border) !important;
        color: #dddddd !important;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: #0b0b0b !important;
        color: #e5e5e5 !important;
        border: 1px solid var(--border) !important;
    }

    /* Hero chips */
    .hero-chip {
        border: 1px solid var(--border-strong);
        color: #9f9f9f;
        padding: 6px 18px;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        background: rgba(255,255,255,0.01);
    }
</style>
""", unsafe_allow_html=True)


# ── API Status ──────────────────────────────────────────────────────────────
api_live = False
model_loaded = False
model_ver = None
model_info_data = None

try:
    health = requests.get(f"{API_URL}/health", timeout=10).json()
    api_live = True
    model_loaded = health.get("model_loaded", False)
    model_ver = health.get("model_version", None)
except Exception:
    pass

try:
    model_info_data = requests.get(f"{API_URL}/model/info", timeout=10).json()
except Exception:
    pass


# ── Hero Section ────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 58px 0 22px 0; border-bottom: 1px solid #1a1a1a;">
    <p style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #4a4a4a;
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 16px;">
    Neural Decoder · MLOps · Neuropixels
    </p>

    <h1 style="font-family: 'Syne', sans-serif; font-size: 5rem; font-weight: 800;
    margin: 0; color: #ffffff; letter-spacing: -0.05em; line-height: 0.95;">
    DeadNeurons
    </h1>

    <p style="font-family: 'DM Mono', monospace; font-size: 0.9rem; color: #666666;
    margin-top: 18px; line-height: 1.8;">
    Self-improving neural decoder with full MLOps lifecycle<br>
    Built in pure NumPy · Trained on real Neuropixels brain recordings · Deployed end to end
    </p>

    <div style="margin-top: 26px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
        <span class="hero-chip">84.3% Mean Accuracy</span>
        <span class="hero-chip">26 Sessions</span>
        <span class="hero-chip">4,869 Trials</span>
        <span class="hero-chip">$0 Infrastructure</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; font-family: 'DM Mono', monospace; font-size: 0.72rem;
color: #3f3f3f; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 18px;">
Built by <a href="https://github.com/Rekhii" style="color: #9e9e9e;">Rekhi</a>
&nbsp;·&nbsp;
<a href="https://github.com/Rekhii/DeadNeurons" style="color: #9e9e9e;">GitHub</a>
&nbsp;·&nbsp;
<a href="https://huggingface.co/datasets/rekhi/deadneurons-registry" style="color: #9e9e9e;">HF Hub</a>
</p>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Static Results ──────────────────────────────────────────────────────────
results = {
    "mean_accuracy": 0.843,
    "best_accuracy": 0.980,
    "worst_accuracy": 0.677,
    "above_chance": 25,
    "total_sessions": 26,
    "total_trials": 4869,
    "per_session": [
        {"session": 0, "mouse": "Cori", "neurons": 734, "acc": 0.929},
        {"session": 1, "mouse": "Cori", "neurons": 1070, "acc": 0.781},
        {"session": 2, "mouse": "Cori", "neurons": 619, "acc": 0.677},
        {"session": 3, "mouse": "Forssmann", "neurons": 1769, "acc": 0.760},
        {"session": 4, "mouse": "Forssmann", "neurons": 1077, "acc": 0.767},
        {"session": 5, "mouse": "Forssmann", "neurons": 1169, "acc": 0.825},
        {"session": 6, "mouse": "Forssmann", "neurons": 584, "acc": 0.789},
        {"session": 7, "mouse": "Hench", "neurons": 1156, "acc": 0.875},
        {"session": 8, "mouse": "Hench", "neurons": 788, "acc": 0.875},
        {"session": 9, "mouse": "Hench", "neurons": 1172, "acc": 0.864},
        {"session": 10, "mouse": "Hench", "neurons": 857, "acc": 0.980},
        {"session": 11, "mouse": "Lederberg", "neurons": 698, "acc": 0.893},
        {"session": 12, "mouse": "Lederberg", "neurons": 983, "acc": 0.939},
        {"session": 13, "mouse": "Lederberg", "neurons": 756, "acc": 0.905},
        {"session": 14, "mouse": "Lederberg", "neurons": 743, "acc": 0.919},
        {"session": 15, "mouse": "Lederberg", "neurons": 474, "acc": 0.929},
        {"session": 16, "mouse": "Lederberg", "neurons": 565, "acc": 0.943},
        {"session": 17, "mouse": "Lederberg", "neurons": 1089, "acc": 0.875},
        {"session": 18, "mouse": "Moniz", "neurons": 606, "acc": 0.704},
        {"session": 19, "mouse": "Moniz", "neurons": 899, "acc": 0.828},
        {"session": 20, "mouse": "Moniz", "neurons": 578, "acc": 0.714},
        {"session": 21, "mouse": "Muller", "neurons": 646, "acc": 0.833},
        {"session": 22, "mouse": "Muller", "neurons": 1268, "acc": 0.778},
        {"session": 23, "mouse": "Muller", "neurons": 1337, "acc": 0.762},
        {"session": 24, "mouse": "Radnitz", "neurons": 885, "acc": 0.892},
        {"session": 25, "mouse": "Radnitz", "neurons": 1056, "acc": 0.885},
    ]
}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean Accuracy", f"{results['mean_accuracy']:.1%}")
c2.metric("Best Session", f"{results['best_accuracy']:.1%}")
c3.metric("Sessions > Chance", f"{results['above_chance']}/{results['total_sessions']}")
c4.metric("Total Trials", f"{results['total_trials']:,}")
c5.metric("API Status", "Live" if api_live else "Offline")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab_neuro, tab2, tab3, tab4 = st.tabs([
    "Performance",
    "How Neural Activity Was Captured",
    "Live Prediction",
    "Architecture",
    "Model Registry"
])


# ── Tab 1: Performance ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Decoding Accuracy Across 26 Recording Sessions")
    st.caption("Each session is a separate Neuropixels recording from a mouse performing a visual decision task.")

    df = pd.DataFrame(results["per_session"])

    chart = (
        alt.Chart(df)
        .mark_bar(color="#f2f2f2", size=18)
        .encode(
            x=alt.X("session:O", title="Session", axis=alt.Axis(labelColor="#8c8c8c", titleColor="#bcbcbc")),
            y=alt.Y("acc:Q", title="Accuracy", axis=alt.Axis(format="%", labelColor="#8c8c8c", titleColor="#bcbcbc")),
            tooltip=[
                alt.Tooltip("session:O", title="Session"),
                alt.Tooltip("mouse:N", title="Mouse"),
                alt.Tooltip("neurons:Q", title="Neurons"),
                alt.Tooltip("acc:Q", title="Accuracy", format=".1%")
            ]
        )
        .properties(height=360)
        .configure_view(stroke="#1f1f1f", fill="#0a0a0a")
        .configure_axis(gridColor="#171717", domainColor="#2a2a2a", tickColor="#2a2a2a")
        .configure(background="#0a0a0a")
    )
    st.altair_chart(chart, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Average Accuracy by Mouse")
        mouse_df = df.groupby("mouse").agg(
            mean_acc=("acc", "mean"),
            sessions=("session", "count"),
            neurons=("neurons", "mean")
        ).sort_values("mean_acc", ascending=False)

        display_mouse_df = mouse_df.copy()
        display_mouse_df["mean_acc"] = display_mouse_df["mean_acc"].apply(lambda x: f"{x:.1%}")
        display_mouse_df["neurons"] = display_mouse_df["neurons"].astype(int)
        display_mouse_df.columns = ["Accuracy", "Sessions", "Avg Neurons"]
        st.dataframe(display_mouse_df, use_container_width=True)

    with col_b:
        st.subheader("All Sessions")
        table_df = df[["session", "mouse", "neurons", "acc"]].copy()
        table_df["acc"] = table_df["acc"].apply(lambda x: f"{x:.1%}")
        table_df.columns = ["Session", "Mouse", "Neurons", "Accuracy"]
        st.dataframe(table_df, use_container_width=True, height=360)

    st.info(
        "Session 10 (Hench, 857 neurons) achieves 98.0% accuracy. "
        "Session 2 (Cori, 619 neurons) is the hardest at 67.7%. "
        "This spread across sessions is exactly what production drift monitoring should surface."
    )


# ── Tab 2: Neural Activity ──────────────────────────────────────────────────
with tab_neuro:
    st.subheader("Neural Activity — How the Data Was Captured")
    st.markdown(
        "The Steinmetz 2019 dataset recorded spiking activity from hundreds of neurons "
        "simultaneously across multiple brain regions using **Neuropixels probes**. "
        "The figures below walk through the experimental pipeline — from surgical access "
        "to stable recording and imaging systems."
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    neural_panels = [
        {
            "path": "figures/1.png",
            "title": "Step 1 — Craniotomy & Probe Insertion",
            "caption": (
                "The skull is opened above the target region, exposing the craniotomy site. "
                "A Neuropixels probe is then inserted along carefully controlled mediolateral "
                "and dorsoventral trajectories so deep structures can be targeted with precision. "
                "Insertion speed and angle matter because they directly affect tissue damage, "
                "recording stability, and the final number of cleanly isolated neurons."
            ),
        },
        {
            "path": "figures/2.png",
            "title": "Step 2 — Head-Fixed Recording Setup",
            "caption": (
                "A metal plate is cemented to the skull and rigidly clamped during the experiment. "
                "This eliminates motion between the brain and the probe, improving signal stability "
                "while the animal still performs the visual decision task with its paws. "
                "The main benefit is precise synchronization between behavior, sensory stimuli, and neural activity."
            ),
        },
        {
            "path": "figures/3.png",
            "title": "Step 3 — Cranial Window Over Time",
            "caption": (
                "A chronic cranial window replaces a patch of skull with a sealed glass coverslip, "
                "allowing repeated optical access to the same brain region across weeks or months. "
                "This is essential for longitudinal neuroscience because the same neurons and microcircuits "
                "can be revisited over time. The challenge is maintaining optical clarity despite inflammation or tissue regrowth."
            ),
        },
        {
            "path": "figures/4.png",
            "title": "Step 4 — Two-Photon Imaging of Neural Activity",
            "caption": (
                "Two-photon microscopy uses a focused infrared laser to excite fluorescent calcium indicators "
                "within a tiny focal volume, enabling cellular-resolution imaging deep in tissue. "
                "The fluorescence traces are an indirect proxy for spiking because they reflect calcium dynamics, "
                "not membrane voltage directly. The major strength is spatial resolution; the limitation is slower temporal precision."
            ),
        },
        {
            "path": "figures/5.png",
            "title": "Step 5 — In Vivo Microendoscopy & Fiber Photometry",
            "caption": (
                "These techniques extend optical access to deeper structures. "
                "Microendoscopy preserves some spatial detail through a GRIN lens and miniature microscope, "
                "while fiber photometry provides a stable bulk fluorescence signal from a larger population. "
                "The tradeoff is richer cellular detail versus simpler, more robust population-level readout."
            ),
        },
    ]

    for panel in neural_panels:
        st.markdown(f"### {panel['title']}")
        img_col, txt_col = st.columns([3, 2])

        with img_col:
            st.image(panel["path"], use_container_width=True)

        with txt_col:
            st.markdown(
                f"""
                <div class="soft-panel">
                    <div class="mono-kicker" style="margin-bottom: 10px;">Experimental Pipeline</div>
                    <p style="margin: 0; color: #c7c7c7;">{panel['caption']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Tab 3: Live Prediction ──────────────────────────────────────────────────
with tab2:
    st.subheader("Live Prediction")
    st.markdown(
        "This section can be used to connect the frontend to your deployed API for real-time inference, "
        "show prediction probabilities, and surface model metadata from production."
    )

    live_c1, live_c2 = st.columns([1, 1])

    with live_c1:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("#### Runtime Status")
        st.write(f"**API:** {'Live' if api_live else 'Offline'}")
        st.write(f"**Model Loaded:** {'Yes' if model_loaded else 'No'}")
        st.write(f"**Version:** {model_ver if model_ver else 'Unavailable'}")
        st.markdown("</div>", unsafe_allow_html=True)

    with live_c2:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("#### Notes")
        st.markdown(
            "Keep this block minimal and operational. "
            "It should show exactly what is deployed, whether the service is reachable, "
            "and what model version is currently serving predictions."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 4: Architecture ─────────────────────────────────────────────────────
with tab3:
    st.subheader("System Architecture")
    st.image("figures/System_Arch.png", use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Self-Improvement Cycle")
    si1, si2, si3 = st.columns(3)

    with si1:
        st.markdown(
            """
            <div class="soft-panel">
                <div class="mono-kicker" style="margin-bottom: 8px;">1. Observe</div>
                <p style="margin: 0;">After each training epoch, record mean activation and standard deviation of every hidden neuron across the batch.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with si2:
        st.markdown(
            """
            <div class="soft-panel">
                <div class="mono-kicker" style="margin-bottom: 8px;">2. Diagnose</div>
                <p style="margin: 0;">Dead neurons show near-zero mean and variance. Saturated neurons show high mean but almost no variance.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with si3:
        st.markdown(
            """
            <div class="soft-panel">
                <div class="mono-kicker" style="margin-bottom: 8px;">3. Correct</div>
                <p style="margin: 0;">Dead neurons are reinitialized with fresh He weights. Saturated neurons have their weights scaled down.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.subheader("Tech Stack")
    tech_df = pd.DataFrame([
        ["Core Model", "Pure NumPy", "No deep learning frameworks"],
        ["API", "FastAPI + Uvicorn", "REST endpoints for predictions"],
        ["Deployment", "Docker on Render.com", "Free tier, auto-deploy"],
        ["Dashboard", "Streamlit Cloud", "Live monitoring"],
        ["Model Registry", "Hugging Face Hub", "Versioned artifact storage"],
        ["Experiment Tracking", "SQLite", "Custom built, no MLflow"],
        ["CI/CD", "GitHub Actions", "Tests on push, weekly retrain"],
        ["Drift Detection", "PSI (NumPy/SciPy)", "Population Stability Index"],
        ["Total Cost", "$0", "Entire stack is free"],
    ], columns=["Component", "Tool", "Notes"])

    st.dataframe(tech_df, use_container_width=True, hide_index=True)


# ── Tab 5: Model Registry ───────────────────────────────────────────────────
with tab4:
    st.subheader("Model Registry")

    if model_info_data:
        reg1, reg2, reg3 = st.columns(3)
        reg1.metric("Production Model", model_ver or "None")
        reg2.metric("Input Features", model_info_data.get("n_features", "N/A"))
        reg3.metric("Hidden Neurons", model_info_data.get("n_hidden", "N/A"))

        if model_info_data.get("config"):
            st.subheader("Production Model Configuration")
            config = model_info_data["config"]

            cfg_df = pd.DataFrame([
                ["Hidden Layer Size", config.get("hidden", "N/A")],
                ["Learning Rate", config.get("lr", "N/A")],
                ["L2 Regularization", config.get("reg", "N/A")],
                ["PCA Components", config.get("pca_components", "N/A")],
                ["Training Epochs", config.get("epochs", "N/A")],
                ["Random Seed", config.get("seed", "N/A")],
            ], columns=["Parameter", "Value"])

            st.dataframe(cfg_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not connect to API to fetch model info.")

    st.subheader("How Promotion Works")
    st.markdown(
        "1. A new model is trained and registered as **candidate**.\n"
        "2. Its accuracy is compared against the current **production** model.\n"
        "3. If the candidate wins, it gets promoted. The old model is **retired**.\n"
        "4. If the candidate loses, it stays candidate. Production remains unchanged.\n"
        "5. Weights are stored on Hugging Face Hub with full version history."
    )

    st.link_button(
        "View Model Registry on HF Hub",
        "https://huggingface.co/datasets/rekhi/deadneurons-registry",
        use_container_width=True
    )


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 6px 0 20px 0;">
    <p style="font-family:'DM Mono', monospace; font-size:0.72rem; color:#4a4a4a;
    letter-spacing:0.12em; text-transform:uppercase; margin-bottom:10px;">
    DeadNeurons · Neural Decoder · Production Dashboard
    </p>

    <p style="font-family:'DM Mono', monospace; font-size:0.76rem; color:#666666; margin:0;">
    Built by <a href="https://github.com/Rekhii" style="color:#a6a6a6;">Rekhi</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/Rekhii/DeadNeurons" style="color:#a6a6a6;">GitHub</a>
    &nbsp;·&nbsp;
    <a href="https://huggingface.co/datasets/rekhi/deadneurons-registry" style="color:#a6a6a6;">HF Hub</a>
    </p>
</div>
""", unsafe_allow_html=True)