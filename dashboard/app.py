import streamlit as st
import requests
import numpy as np
import pandas as pd

API_URL = 'https://deadneurons.onrender.com'

st.set_page_config(
    page_title='DeadNeurons | Neural Decoder MLOps',
    layout='wide'
)

st.markdown("""
<style>
    .section-divider {
        border-top: 1px solid #2d2d44;
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div style="text-align: center; padding: 40px 0 20px 0;">
    <h1 style="font-size: 4rem; font-weight: 800; margin: 0; 
    background: linear-gradient(90deg, #4ade80, #22d3ee); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    DeadNeurons</h1>
    <p style="font-size: 1.3rem; color: #888888; margin-top: 8px;">
    Self-improving neural decoder with full MLOps lifecycle<br>
    Built in pure NumPy. Trained on real Neuropixels brain recordings. Deployed end to end.
    </p>
    <p style="margin-top: 16px;">
    <span style="background: #065f46; color: #4ade80; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin: 0 4px;">84.3% Mean Accuracy</span>
    <span style="background: #1e1b4b; color: #818cf8; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin: 0 4px;">26 Sessions</span>
    <span style="background: #312e81; color: #c4b5fd; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin: 0 4px;">4,869 Trials</span>
    <span style="background: #064e3b; color: #6ee7b7; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin: 0 4px;">$0 Infrastructure</span>
    </p>
</div>
""", unsafe_allow_html=True)

# API Status Check
api_live = False
model_loaded = False
model_ver = None
model_info_data = None

try:
    health = requests.get(f'{API_URL}/health', timeout=10).json()
    api_live = True
    model_loaded = health.get('model_loaded', False)
    model_ver = health.get('model_version', None)
except:
    pass

try:
    model_info_data = requests.get(f'{API_URL}/model/info', timeout=10).json()
except:
    pass

# Top metrics row
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

results = {
    'mean_accuracy': 0.843,
    'best_accuracy': 0.980,
    'worst_accuracy': 0.677,
    'above_chance': 25,
    'total_sessions': 26,
    'total_trials': 4869,
    'per_session': [
        {'session': 0, 'mouse': 'Cori', 'neurons': 734, 'acc': 0.929},
        {'session': 1, 'mouse': 'Cori', 'neurons': 1070, 'acc': 0.781},
        {'session': 2, 'mouse': 'Cori', 'neurons': 619, 'acc': 0.677},
        {'session': 3, 'mouse': 'Forssmann', 'neurons': 1769, 'acc': 0.760},
        {'session': 4, 'mouse': 'Forssmann', 'neurons': 1077, 'acc': 0.767},
        {'session': 5, 'mouse': 'Forssmann', 'neurons': 1169, 'acc': 0.825},
        {'session': 6, 'mouse': 'Forssmann', 'neurons': 584, 'acc': 0.789},
        {'session': 7, 'mouse': 'Hench', 'neurons': 1156, 'acc': 0.875},
        {'session': 8, 'mouse': 'Hench', 'neurons': 788, 'acc': 0.875},
        {'session': 9, 'mouse': 'Hench', 'neurons': 1172, 'acc': 0.864},
        {'session': 10, 'mouse': 'Hench', 'neurons': 857, 'acc': 0.980},
        {'session': 11, 'mouse': 'Lederberg', 'neurons': 698, 'acc': 0.893},
        {'session': 12, 'mouse': 'Lederberg', 'neurons': 983, 'acc': 0.939},
        {'session': 13, 'mouse': 'Lederberg', 'neurons': 756, 'acc': 0.905},
        {'session': 14, 'mouse': 'Lederberg', 'neurons': 743, 'acc': 0.919},
        {'session': 15, 'mouse': 'Lederberg', 'neurons': 474, 'acc': 0.929},
        {'session': 16, 'mouse': 'Lederberg', 'neurons': 565, 'acc': 0.943},
        {'session': 17, 'mouse': 'Lederberg', 'neurons': 1089, 'acc': 0.875},
        {'session': 18, 'mouse': 'Moniz', 'neurons': 606, 'acc': 0.704},
        {'session': 19, 'mouse': 'Moniz', 'neurons': 899, 'acc': 0.828},
        {'session': 20, 'mouse': 'Moniz', 'neurons': 578, 'acc': 0.714},
        {'session': 21, 'mouse': 'Muller', 'neurons': 646, 'acc': 0.833},
        {'session': 22, 'mouse': 'Muller', 'neurons': 1268, 'acc': 0.778},
        {'session': 23, 'mouse': 'Muller', 'neurons': 1337, 'acc': 0.762},
        {'session': 24, 'mouse': 'Radnitz', 'neurons': 885, 'acc': 0.892},
        {'session': 25, 'mouse': 'Radnitz', 'neurons': 1056, 'acc': 0.885},
    ]
}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Mean Accuracy', f"{results['mean_accuracy']:.1%}")
c2.metric('Best Session', f"{results['best_accuracy']:.1%}")
c3.metric('Sessions > Chance', f"{results['above_chance']}/{results['total_sessions']}")
c4.metric('Total Trials', f"{results['total_trials']:,}")
c5.metric('API Status', 'Live' if api_live else 'Offline')

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    'Performance',
    'Live Prediction',
    'Architecture',
    'Model Registry'
])

# Tab 1: Performance
with tab1:
    st.subheader('Decoding Accuracy Across 26 Recording Sessions')
    st.caption('Each session is a separate Neuropixels recording from a mouse performing a visual decision task.')

    df = pd.DataFrame(results['per_session'])

    chart_df = df[['session', 'acc']].copy()
    chart_df = chart_df.set_index('session')

    st.bar_chart(chart_df['acc'], height=350, color='#4ade80')

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader('Average Accuracy by Mouse')
        mouse_df = df.groupby('mouse').agg(
            mean_acc=('acc', 'mean'),
            sessions=('session', 'count'),
            neurons=('neurons', 'mean')
        ).sort_values('mean_acc', ascending=False)
        mouse_df['mean_acc'] = mouse_df['mean_acc'].apply(lambda x: f"{x:.1%}")
        mouse_df['neurons'] = mouse_df['neurons'].astype(int)
        mouse_df.columns = ['Accuracy', 'Sessions', 'Avg Neurons']
        st.dataframe(mouse_df, use_container_width=True)

    with col_b:
        st.subheader('All Sessions')
        table_df = df[['session', 'mouse', 'neurons', 'acc']].copy()
        table_df['acc'] = table_df['acc'].apply(lambda x: f"{x:.1%}")
        table_df.columns = ['Session', 'Mouse', 'Neurons', 'Accuracy']
        st.dataframe(table_df, use_container_width=True, height=350)

    st.info(
        'Session 10 (Hench, 857 neurons) achieves 98.0% accuracy. '
        'Session 2 (Cori, 619 neurons) is the hardest at 67.7%. '
        'This variance across sessions is exactly what drift detection monitors in production.'
    )

# Tab 2: Live Prediction
with tab2:
    st.subheader('Send Neural Features to the Deployed Model')

    if not api_live:
        st.warning(
            'The API is on Render free tier and may be sleeping. '
            'Click Predict to wake it up. First request takes about 50 seconds.'
        )

    pred_col1, pred_col2 = st.columns([2, 1])

    with pred_col1:
        if st.button('Generate Random Trial', use_container_width=True):
            rng = np.random.default_rng()
            st.session_state['demo_features'] = rng.normal(0, 1, 50).tolist()

        features_text = st.text_area(
            'Features (50 PCA components)',
            value=', '.join([f'{x:.3f}' for x in st.session_state.get('demo_features', [0.0]*50)]),
            height=120
        )

    with pred_col2:
        st.markdown('**How it works:**')
        st.markdown(
            '1. Raw spike data from 734+ neurons\n'
            '2. Mean firing rates in 4 time windows\n'
            '3. PCA reduces to 50 components\n'
            '4. Self-improving classifier predicts\n'
            '5. Output: LEFT or RIGHT decision'
        )

    if st.button('Predict', type='primary', use_container_width=True):
        with st.spinner('Sending to API...'):
            try:
                features = [float(x.strip()) for x in features_text.split(',')]
                response = requests.post(
                    f'{API_URL}/predict',
                    json={'features': features},
                    timeout=120
                )
                result = response.json()

                if 'prediction' in result:
                    r1, r2, r3 = st.columns(3)
                    r1.metric('Decision', result['label'].upper())
                    r2.metric('Confidence', f"{result['confidence']:.1%}")
                    r3.metric('Raw Prediction', result['prediction'])
                else:
                    st.error(f"API Error: {result}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# Tab 3: Architecture
with tab3:
    st.subheader('System Architecture')
    st.image('figures/System_Arch.png', use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('**Self-Improvement Cycle**')
    si1, si2, si3 = st.columns(3)
    with si1:
        st.markdown('**1. Observe**')
        st.markdown(
            'After each training epoch, record mean activation and '
            'standard deviation of every hidden neuron across the training batch.'
        )
    with si2:
        st.markdown('**2. Diagnose**')
        st.markdown(
            'Dead neuron: mean and std near zero. Outputs nothing for any input. '
            'Saturated neuron: high mean, near-zero std. Same output for everything.'
        )
    with si3:
        st.markdown('**3. Correct**')
        st.markdown(
            'Dead neurons get reinitialized with fresh He weights. '
            'Saturated neurons get their weights scaled down by 50%.'
        )

    st.subheader('Tech Stack')
    tech_df = pd.DataFrame([
        ['Core Model', 'Pure NumPy', 'No deep learning frameworks'],
        ['API', 'FastAPI + Uvicorn', 'REST endpoints for predictions'],
        ['Deployment', 'Docker on Render.com', 'Free tier, auto-deploy'],
        ['Dashboard', 'Streamlit Cloud', 'Live monitoring'],
        ['Model Registry', 'Hugging Face Hub', 'Versioned artifact storage'],
        ['Experiment Tracking', 'SQLite', 'Custom built, no MLflow'],
        ['CI/CD', 'GitHub Actions', 'Tests on push, weekly retrain'],
        ['Drift Detection', 'PSI (NumPy/SciPy)', 'Population Stability Index'],
        ['Total Cost', '$0', 'Entire stack is free'],
    ], columns=['Component', 'Tool', 'Notes'])
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

# Tab 4: Model Registry
with tab4:
    st.subheader('Model Registry')

    if model_info_data:
        reg1, reg2, reg3 = st.columns(3)
        reg1.metric('Production Model', model_ver or 'None')
        reg2.metric('Input Features', model_info_data.get('n_features', 'N/A'))
        reg3.metric('Hidden Neurons', model_info_data.get('n_hidden', 'N/A'))

        if model_info_data.get('config'):
            st.subheader('Production Model Configuration')
            config = model_info_data['config']
            cfg_df = pd.DataFrame([
                ['Hidden Layer Size', config.get('hidden', 'N/A')],
                ['Learning Rate', config.get('lr', 'N/A')],
                ['L2 Regularization', config.get('reg', 'N/A')],
                ['PCA Components', config.get('pca_components', 'N/A')],
                ['Training Epochs', config.get('epochs', 'N/A')],
                ['Random Seed', config.get('seed', 'N/A')],
            ], columns=['Parameter', 'Value'])
            st.dataframe(cfg_df, use_container_width=True, hide_index=True)
    else:
        st.warning('Could not connect to API to fetch model info.')

    st.subheader('How Promotion Works')
    st.markdown(
        '1. A new model is trained and registered as **candidate**.\n'
        '2. Its accuracy is compared against the current **production** model.\n'
        '3. If the candidate wins, it gets promoted. The old model is **retired**.\n'
        '4. If the candidate loses, it stays candidate. Production model unchanged.\n'
        '5. Weights are stored on Hugging Face Hub with full version history.'
    )

    st.link_button(
        'View Model Registry on HF Hub',
        'https://huggingface.co/datasets/rekhi/deadneurons-registry',
        use_container_width=True
    )

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666666;">'
    'Built by <a href="https://github.com/Rekhii" style="color: #4ade80;">Rekhi</a> '
    ' <a href="https://github.com/Rekhii/DeadNeurons" style="color: #4ade80;">GitHub</a> '
    ' <a href="https://huggingface.co/datasets/rekhi/deadneurons-registry" style="color: #4ade80;">HF Hub</a>'
    '</p>',
    unsafe_allow_html=True
)