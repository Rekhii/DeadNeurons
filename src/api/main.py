from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json
import os
import sys

from src.model.classifier import SelfImprovingClassifier
from src.registry.registry import ModelRegistry



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
    """Load the production model from registry at startup."""
    global model, model_config, model_version

    registry = ModelRegistry('rekhi/deadneurons-registry')
    weights, config = registry.get_production_model()

    if weights is None:
        print("WARNING: No production model found in registry")
        return

    # Create classifier with the right dimensions
    n_features = weights['W1'].shape[0]
    n_hidden = weights['W1'].shape[1]

    model = SelfImprovingClassifier(
        n_features=n_features,
        n_hidden=n_hidden
    )
    model.set_weights(weights)
    model_config = config
    model_version = registry.registry['current_production']

    print(f"  [API] Loaded model {model_version} "
          f"({n_features} features, {n_hidden} hidden)")


# Load model when the app starts
load_model()


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