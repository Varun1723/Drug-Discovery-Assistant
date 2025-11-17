#!/usr/bin/env python3
"""
Training script for Toxicity Classifier (Binary Classification)
"""
import json
import os
import sys
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_fingerprint(smiles: str):
    """Convert SMILES to Morgan Fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    except:
        return None
    return None

def train_toxicity_model(data_path: Path, output_dir: Path):
    logger.info(f"Loading training data from {data_path}")

    try:
        df = pd.read_csv(data_path, compression='gzip')
        df = df[['smiles', 'NR-AR']].dropna()
        df = df.rename(columns={'NR-AR': 'is_toxic'})
        df['is_toxic'] = df['is_toxic'].astype(int)
    except Exception as e:
        logger.error(f"Failed to load real Tox21 data: {e}. Exiting.")
        return

    logger.info(f"Loaded {len(df)} valid samples from Tox21 (NR-AR task).")

    logger.info("Generating fingerprints...")
    df['fp'] = df['smiles'].apply(get_fingerprint)
    df = df.dropna()

    X = np.array(df['fp'].tolist())
    y = df['is_toxic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- THIS IS THE GOOD FIX (scale_pos_weight) ---
    try:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Dataset is imbalanced. Using scale_pos_weight: {scale_pos_weight:.2f}")
    except ZeroDivisionError:
        logger.warning("No toxic samples in training set. Using default weight.")
        scale_pos_weight = 1
    # --- END OF FIX ---

    logger.info(f"Training XGBoost Classifier on {len(X_train)} samples...")

    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        device='cuda',
        scale_pos_weight=scale_pos_weight  # <-- We are using the correct fix
    )
    
    model.fit(X_train, y_train) 

    # --- Test model ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report_str = classification_report(y_test, preds)
    report_dict = classification_report(y_test, preds, output_dict=True)

    logger.info(f"Model Accuracy on test set: {acc:.2%}")
    logger.info("\nClassification Report:\n" + report_str)

    # --- Save model ---
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "toxicity_xgb_model.pkl"
    joblib.dump(model, save_path)
    logger.info(f"✓ Toxicity Model saved to {save_path}")

    # --- Save metrics to JSON ---
    metrics = {
        "accuracy": acc,
        "classification_report": report_dict,
        "n_samples_train": len(X_train), # Use original train size
        "n_samples_test": len(X_test)
    }
    metrics_path = output_dir / "toxicity_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✓ Toxicity metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train property predictor models")
    parser.add_argument('--data_file', type=Path, default=project_root / "data" / "raw" / "tox21.csv.gz")
    parser.add_argument('--output_dir', type=Path, default=project_root / "data" / "models" / "predictor_toxicity")
    args = parser.parse_args()

    train_toxicity_model(args.data_file, args.output_dir)