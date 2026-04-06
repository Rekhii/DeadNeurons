from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json
import os
import sys

from src.model.classifier import SelfImprovingClassifier



# Add project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


app = FastAPI(
    title='DeadNeurons API',
    description='Neural decoder serving predictions from self-improving classifier',
    version='1.0.0'
)


# Define what the request data looks like
class PredictRequest(BaseModel):
    features: list  # list of 50 floats (one trial after PCA)

class BatchPredictRequest(BaseModel):
    features: list  # list of lists (multiple trials)

# Global variables to hold the loaded model
model = None
model_config = None
model_version = None


def load_model():
    """Load the production model from HF Hub at startup."""
    global model, model_config, model_version
    print("  [API] Starting model load...")
    try:
        from huggingface_hub import hf_hub_download

        repo_id = 'rekhi/deadneurons-registry'
        token = os.environ.get('HF_TOKEN')

        registry_path = hf_hub_download(
            repo_id=repo_id,
            filename='registry.json',
            repo_type='dataset',
            token=token
        )

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        version = registry.get('current_production')
        if version is None:
            print("WARNING: No production model in registry")
            return

        # Download weights and config for the production version
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f'models/{version}/weights.npz',
            repo_type='dataset',
            token=token
        )
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f'models/{version}/config.json',
            repo_type='dataset',
            token=token
        )

        weights = dict(np.load(weights_path))

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        n_features = weights['W1'].shape[0]
        n_hidden = weights['W1'].shape[1]

        model = SelfImprovingClassifier(
            n_features=n_features,
            n_hidden=n_hidden
        )
        model.set_weights(weights)
        model_version = version

        print(f"  [API] Loaded {version} from HF Hub ({n_features} features, {n_hidden} hidden)")


    except Exception as e:
        import traceback
        print(f"  [API] Failed to load model: {e}")
        traceback.print_exc()

@app.get('/health')
def health():
    """Check if the API is alive and model is loaded."""
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_version': model_version
    }


@app.get('/model/info')
def model_info():
    """Return info about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail='No model loaded')

    return {
        'version': model_version,
        'config': model_config,
        'n_features': model.n_features,
        'n_hidden': model.n_hidden
    }


@app.post('/predict')
def predict(request: PredictRequest):
    """Predict decision from one trial's features."""
    if model is None:
        raise HTTPException(status_code=503, detail='No model loaded')

    # Convert input to numpy array
    X = np.array(request.features).reshape(1, -1)

    # Check feature count matches
    if X.shape[1] != model.n_features:
        raise HTTPException(
            status_code=400,
            detail=f'Expected {model.n_features} features, got {X.shape[1]}'
        )

    labels, probs = model.predict(X)

    return {
        'prediction': int(labels[0]),
        'label': 'right' if labels[0] == 1 else 'left',
        'confidence': round(float(probs[0]), 4)
    }


@app.post('/predict/batch')
def predict_batch(request: BatchPredictRequest):
    """Predict decisions from multiple trials."""
    if model is None:
        raise HTTPException(status_code=503, detail='No model loaded')

    X = np.array(request.features)

    if X.ndim != 2 or X.shape[1] != model.n_features:
        raise HTTPException(
            status_code=400,
            detail=f'Expected shape (n_trials, {model.n_features}), got {X.shape}'
        )

    labels, probs = model.predict(X)

    results = []
    for i in range(len(labels)):
        results.append({
            'prediction': int(labels[i]),
            'label': 'right' if labels[i] == 1 else 'left',
            'confidence': round(float(probs[i]), 4)
        })

    return {'predictions': results, 'count': len(results)}

load_model()