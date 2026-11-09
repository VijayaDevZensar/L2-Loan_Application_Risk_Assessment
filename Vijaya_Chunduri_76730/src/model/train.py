"""Load bundled synthetic baseline artifacts; do not train on customer data."""
from pathlib import Path
import pickle
MODELS_DIR = Path(__file__).resolve().parents[2] / 'models'
def load_artifacts():
    # Only load the trusted artifacts shipped with this project.
    with (MODELS_DIR / 'risk_model_v1.pkl').open('rb') as f: model = pickle.load(f)
    with (MODELS_DIR / 'scaler_v1.pkl').open('rb') as f: scaler = pickle.load(f)
    with (MODELS_DIR / 'pipeline_v1.pkl').open('rb') as f: pipeline = pickle.load(f)
    return model, scaler, pipeline
def risk_category(probability):
    if not 0 <= probability <= 1: raise ValueError('Probability outside [0,1]')
    return 'Low' if probability < 0.35 else ('Medium' if probability < 0.7 else 'High')
