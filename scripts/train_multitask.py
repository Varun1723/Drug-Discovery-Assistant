#!/usr/bin/env python3
"""
Training script for a Multi-Task Toxicity Classifier (Full Tox21)
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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
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
    except Exception as e:
        logger.warning(f"Failed to parse SMILES: {smiles} - {e}")
        return None
    return None

def train_multitask_model(data_path: Path, output_dir: Path):
    logger.info(f"Loading FULL Tox21 training data from {data_path}")

    try:
        df = pd.read_csv(data_path, compression='gzip')
        # Define the 12 toxicity tasks (the targets)
        tasks = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        # Keep only the SMILES and the 12 task columns
        df = df[['smiles'] + tasks]
        
        # Drop any rows where *all* 12 tasks are missing
        df = df.dropna(subset=tasks, how='all')
        
        # Replace missing values (NaN) with 0 (assuming non-toxic if not measured)
        df = df.fillna(0)
        
        # Convert all task columns to integers
        df[tasks] = df[tasks].astype(int)
        
    except Exception as e:
        logger.error(f"Failed to load real Tox21 data: {e}. Exiting.")
        return

    logger.info(f"Loaded {len(df)} valid samples from Full Tox21.")

    logger.info("Generating fingerprints... (This may take a moment)")
    df['fp'] = df['smiles'].apply(get_fingerprint)
    df = df.dropna(subset=['fp']) # Drop SMILES that failed fingerprinting

    # Prepare data
    X = np.array(df['fp'].tolist())
    y = df[tasks].values # y is now an array of [0, 1, 0, 0, 1, ..., 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Training Multi-Task XGBoost Classifier on {len(X_train)} samples...")

    # Create a base XGBoost model
    base_xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        device='cuda' # Use the GPU!
    )

    # Create the Multi-Task wrapper
    # This will train one separate XGBoost model for each of the 12 tasks
    model = MultiOutputClassifier(base_xgb, n_jobs=-1) # Use all CPU cores to parallelize
    
    model.fit(X_train, y_train)

    # --- Test model ---
    preds = model.predict(X_test)
    
    logger.info("Multi-Task Model Test Results:")
    avg_accuracy = accuracy_score(y_test, preds)
    logger.info(f"Average Accuracy (over all tasks): {avg_accuracy:.2%}")
    
    # Calculate average AUC (a better metric for imbalanced data)
    try:
        preds_proba = model.predict_proba(X_test)
        # Need to re-format proba for multi-output
        preds_proba_per_task = [p[:, 1] for p in preds_proba]
        avg_auc = roc_auc_score(y_test, np.array(preds_proba_per_task).T, average='weighted')
        logger.info(f"Weighted Average AUC: {avg_auc:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        avg_auc = 0

    # --- Save model ---
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "toxicity_multitask_model.pkl"
    joblib.dump(model, save_path)
    logger.info(f"✓ Multi-Task Model saved to {save_path}")

    # --- Save metrics to JSON ---
    metrics = {
        "accuracy": avg_accuracy,
        "weighted_auc": avg_auc,
        "tasks": tasks,
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test)
    }
    metrics_path = output_dir / "toxicity_multitask_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✓ Multi-Task metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Task Toxicity Classifier")
    parser.add_argument('--data_file', type=Path, default=project_root / "data" / "raw" / "tox21.csv.gz")
    parser.add_argument('--output_dir', type=Path, default=project_root / "data" / "models" / "predictor_multitask_toxicity")
    args = parser.parse_args()
    
    train_multitask_model(args.data_file, args.output_dir)