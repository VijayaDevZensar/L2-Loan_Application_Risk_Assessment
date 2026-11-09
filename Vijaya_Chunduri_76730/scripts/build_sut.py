"""Verify the bundled synthetic SUT artifacts without overwriting baselines."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model.train import load_artifacts
required = ['data/clean/loan_data_clean_v1.0.csv','models/risk_model_v1.pkl','models/scaler_v1.pkl','models/pipeline_v1.pkl']
missing = [p for p in required if not (ROOT/p).is_file()]
if missing: raise SystemExit('Missing artifacts: '+', '.join(missing))
model, scaler, pipeline = load_artifacts()
print('Bundled synthetic model artifacts validated. No retraining performed.')
