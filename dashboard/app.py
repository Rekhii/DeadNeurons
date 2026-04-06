import streamlit as st
import requests
import json
import numpy as np

# Your live API URL
API_URL = 'https://deadneurons.onrender.com'

st.set_page_config(
    page_title='DeadNeurons Dashboard',
    page_icon='🧠',
    layout='wide'
)

st.title('DeadNeurons')
st.caption('A self-improving neural decoder with full MLOps lifecycle')


# Sidebar with model info
st.sidebar.header('Model Status')

try:
    health = requests.get(f'{API_URL}/health', timeout=10).json()
    if health['model_loaded']:
        st.sidebar.success(f"Model: {health['model_version']}")
    else:
        st.sidebar.error('No model loaded')
except Exception as e:
    st.sidebar.error(f'API unreachable: {e}')

try:
    info = requests.get(f'{API_URL}/model/info', timeout=10).json()
    st.sidebar.write(f"Features: {info['n_features']}")
    st.sidebar.write(f"Hidden neurons: {info['n_hidden']}")
    if info.get('config'):
        st.sidebar.write(f"Learning rate: {info['config'].get('lr', 'N/A')}")
        st.sidebar.write(f"Regularization: {info['config'].get('reg', 'N/A')}")
        st.sidebar.write(f"PCA components: {info['config'].get('pca_components', 'N/A')}")
except:
    pass


# Main content - two columns
col1, col2 = st.columns(2)

# Column 1: Live Prediction Demo
with col1:
    st.header('Live Prediction')
    st.write('Send neural features to the deployed model and get a prediction.')

    # Generate random features for demo
    if st.button('Generate Random Trial'):
        rng = np.random.default_rng()
        random_features = rng.normal(0, 1, 50).tolist()
        st.session_state['demo_features'] = random_features

    features_text = st.text_area(
        'Features (50 comma-separated values)',
        value=', '.join([f'{x:.3f}' for x in st.session_state.get('demo_features', [0.0]*50)]),
        height=100
    )

    if st.button('Predict'):
        try:
            features = [float(x.strip()) for x in features_text.split(',')]
            response = requests.post(
                f'{API_URL}/predict',
                json={'features': features},
                timeout=60
            )
            result = response.json()

            if 'prediction' in result:
                label = result['label'].upper()
                conf = result['confidence']
                color = 'green' if conf > 0.7 else 'orange' if conf > 0.5 else 'red'
                st.markdown(f"### Decision: **{label}**")
                st.markdown(f"Confidence: **{conf:.1%}**")
            else:
                st.error(f"Error: {result}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# Column 2: Training Results
with col2:
    st.header('Training Results')

    try:
        with open('weights/training_summary.json', 'r') as f:
            summary = json.load(f)

        st.metric('Mean Accuracy', f"{summary['mean_accuracy']:.1%}")
        st.metric('Best Session', f"{summary['best_accuracy']:.1%}")
        st.metric('Worst Session', f"{summary['worst_accuracy']:.1%}")
        st.metric('Sessions Above Chance', f"{summary['above_chance']}/26")
        st.metric('Dead Neurons Fixed', summary['total_reinitialized'])
    except FileNotFoundError:
        st.info('No training summary found. Run train.py first.')


# Project info section
st.divider()
st.header('About')
st.write(
    'DeadNeurons is a self-improving neural decoder trained on Steinmetz 2019 '
    'Neuropixels data. The model monitors its own hidden neurons, detects dead '
    'and saturated units, and reinitializes them automatically during training.'
)

st.subheader('Architecture')
st.code(
    'Steinmetz Spike Data (neurons x trials x time_bins)\n'
    '    |\n'
    '    v\n'
    'Feature Extraction (mean firing rates per time window)\n'
    '    |\n'
    '    v\n'
    'PCA (reduce to 50 components)\n'
    '    |\n'
    '    v\n'
    'Self-Improving Neural Network (50 -> 32 hidden -> 1 output)\n'
    '    |\n'
    '    v\n'
    'Decision Prediction (left or right)',
    language=None
)

st.subheader('Links')
st.write('[GitHub Repository](https://github.com/Rekhii/DeadNeurons)')
st.write('[Live API Docs](https://deadneurons.onrender.com/docs)')
st.write('[Model Registry (HF Hub)](https://huggingface.co/datasets/rekhi/deadneurons-registry)')