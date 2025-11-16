#!/usr/bin/env python3
"""
Training script for Toxicity Classifier (Binary Classification)
"""

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
        # Load the gzipped CSV file
        df = pd.read_csv(data_path, compression='gzip')
        # We will train on the "NR-AR" assay (a common toxicity marker)
        # We drop any rows where this task wasn't measured
        df = df[['smiles', 'NR-AR']].dropna()
        df = df.rename(columns={'NR-AR': 'is_toxic'})
        # Convert float (0.0, 1.0) to integer (0, 1)
        df['is_toxic'] = df['is_toxic'].astype(int)
    except Exception as e:
        logger.error(f"Failed to load real Tox21 data: {e}. Exiting.")
        return

    logger.info(f"Loaded {len(df)} valid samples from Tox21 (NR-AR task).")

    logger.info("Generating fingerprints...")
    df['fp'] = df['smiles'].apply(get_fingerprint)
    df = df.dropna() # Remove any SMILES that failed fingerprinting

    # Prepare data
    X = np.array(df['fp'].tolist())
    y = df['is_toxic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Training XGBoost Classifier on {len(X_train)} samples...")

    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6,
        use_label_encoder=False, # This is deprecated anyway
        eval_metric='logloss',
        device='cuda' # <-- USE THE GPU!
    )
    model.fit(X_train, y_train)

    # Test model
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Model Accuracy on test set: {acc:.2%}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, preds))

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "toxicity_xgb_model.pkl"
    joblib.dump(model, save_path)

    logger.info(f"✓ Toxicity Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Update the default path to point to the new file
    parser.add_argument('--data_file', type=Path, default=project_root / "data" / "raw" / "tox21.csv.gz")
    parser.add_argument('--output_dir', type=Path, default=project_root / "data" / "models" / "predictor_toxicity")
    args = parser.parse_args()

    train_toxicity_model(args.data_file, args.output_dir)