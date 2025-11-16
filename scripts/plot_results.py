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
    Plots the precision, recall, and f1-score for the toxicity model.
    """
    logger.info(f"Loading toxicity metrics from {TOXICITY_METRICS_PATH}...")
    try:
        with open(TOXICITY_METRICS_PATH, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        logger.error("Toxicity metrics file not found. Run 'train_toxicity.py' first.")
        return

    report = metrics.get('classification_report', {})
    
    # We only care about the '0' (Safe) and '1' (Toxic) classes
    data_to_plot = {
        '0 (Safe)': report.get('0', {}),
        '1 (Toxic)': report.get('1', {})
    }
    
    if not data_to_plot['0 (Safe)'] or not data_to_plot['1 (Toxic)']:
        logger.error("Metrics file is missing classification report data.")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(data_to_plot).T.loc[:, ['precision', 'recall', 'f1-score']]
    df = df.reset_index().melt('index', var_name='Metric', value_name='Score')
    df = df.rename(columns={'index': 'Class'})

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=df, x='Metric', y='Score', hue='Class', palette='deep')
    
    plt.title('Toxicity Classifier Performance (Tox21)', fontsize=16, fontweight='bold')
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title='Class', fontsize=12)
    
    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "toxicity_metrics_report.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"✓ Toxicity metrics plot saved to {save_path}")
    plt.close()

def main():
    logger.info("Starting result plotting...")
    plot_generator_history()
    plot_toxicity_metrics()
    logger.info("All plots saved to /outputs/ directory.")

if __name__ == "__main__":
    main()