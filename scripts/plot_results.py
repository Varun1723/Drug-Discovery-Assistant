#!/usr/bin/env python3
"""
Generates and saves result plots for the research paper.
Reads from the saved .json metrics files.
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
GENERATOR_HISTORY_PATH = project_root / "data" / "models" / "generator" / "generator_lstm_training_history.json"
TOXICITY_METRICS_PATH = project_root / "data" / "models" / "predictor_toxicity" / "toxicity_model_metrics.json"
OUTPUT_DIR = project_root / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_generator_history():
    """
    Plots the training and validation loss for the generator model.
    """
    logger.info(f"Loading generator history from {GENERATOR_HISTORY_PATH}...")
    try:
        with open(GENERATOR_HISTORY_PATH, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        logger.error("Generator history file not found. Run 'train_generator.py' first.")
        return

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss or not val_loss:
        logger.error("History file is empty or missing data.")
        return

    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
    
    plt.title('Generator Model: Training vs. Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Cross-Entropy)', fontsize=12)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0, top=max(max(train_loss), max(val_loss)) * 1.1) # Set y-axis to start at 0
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "generator_loss_curve.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"✓ Generator loss curve saved to {save_path}")
    plt.close()

def plot_toxicity_metrics():
    """
    Plots the high-level metrics for the new multi-task toxicity model.
    """
    logger.info(f"Loading multi-task toxicity metrics from {TOXICITY_METRICS_PATH}...")
    try:
        # --- UPDATE FILE PATH ---
        metrics_path = project_root / "data" / "models" / "predictor_multitask_toxicity" / "toxicity_multitask_metrics.json"
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        logger.error("Toxicity metrics file not found. Run 'scripts/train_multitask.py' first.")
        return

    # --- NEW PLOTTING LOGIC ---
    acc = metrics.get('accuracy', 0)
    auc = metrics.get('weighted_auc', 0)

    if acc == 0 or auc == 0:
        logger.error("Metrics file is missing 'accuracy' or 'weighted_auc' data.")
        return

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Metric': ['Weighted Avg. AUC', 'Average Accuracy'],
        'Score': [auc, acc]
    })

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(data=df, x='Metric', y='Score', palette='viridis')

    plt.title('Multi-Task Toxicity Classifier Performance (Tox21)', fontsize=16, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.0)

    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    plt.tight_layout()

    save_path = OUTPUT_DIR / "toxicity_multitask_metrics_report.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"✓ New toxicity metrics plot saved to {save_path}")
    plt.close()

def main():
    logger.info("Starting result plotting...")
    plot_generator_history()
    plot_toxicity_metrics()
    logger.info("All plots saved to /outputs/ directory.")

if __name__ == "__main__":
    main()