#!/usr/bin/env python3
"""
Analyzes the output of the trained generator model.
Generates:
1. Property Distribution plots (MW, LogP)
2. t-SNE Chemical Space plot
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.manifold import TSNE

import torch

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from c_tokenizers.selfies_tokenizer import SELFIESTokenizer
from models.generator.lightweight_generator import create_lightweight_generator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
GENERATOR_MODEL_PATH = project_root / "data" / "models" / "generator" / "generator_lstm_best.pt"
GENERATOR_CONFIG_PATH = project_root / "data" / "models" / "generator" / "config.json"
TOKENIZER_PATH = project_root / "data" / "models" / "tokenizer"
TRAINING_DATA_PATH = project_root / "data" / "raw" / "training_molecules.txt"
OUTPUT_DIR = project_root / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_TO_GENERATE = 1000 # Number of new molecules to generate for analysis
N_BITS_FP = 2048 # Fingerprint size

# --- HELPER FUNCTIONS ---

def load_generator():
    """Loads the trained generator model and tokenizer."""
    logger.info("Loading tokenizer...")
    tokenizer = SELFIESTokenizer.load(TOKENIZER_PATH)
    
    logger.info("Loading generator model...")
    with open(GENERATOR_CONFIG_PATH, 'r') as f:
        model_config = json.load(f)
    
    model = create_lightweight_generator(
        vocab_size=len(tokenizer),
        profile=model_config.get("profile", "light")
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(GENERATOR_MODEL_PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])      
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device.upper()}.")
    return model, tokenizer, device

def load_training_smiles():
    """Loads the original training SMILES from the text file."""
    logger.info(f"Loading training data from {TRAINING_DATA_PATH}...")
    with open(TRAINING_DATA_PATH, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list

def generate_molecules(model, tokenizer, device, num_samples):
    """Generates N new molecules and returns a list of valid SMILES."""
    logger.info(f"Generating {num_samples} new molecules...")
    selifies_list = model.generate(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=150,
        temperature=1.0, # Use high temperature for more diversity
        device=device
    )
    
    smiles_list = []
    for s in selifies_list:
        smiles = tokenizer.selfies_to_smiles(s)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol: # Final validity check
                smiles_list.append(smiles)
    
    logger.info(f"Generated {len(smiles_list)} valid SMILES (out of {num_samples} attempts).")
    return smiles_list

def get_properties(smiles_list):
    """Calculate MW and LogP for a list of SMILES."""
    props = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            props.append({
                'smiles': smiles,
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol)
            })
    return pd.DataFrame(props)

def get_fingerprints(smiles_list):
    """Calculate Morgan Fingerprints for a list of SMILES."""
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS_FP)
            fps.append(np.array(fp))
    return np.array(fps)

# --- PLOTTING FUNCTIONS ---

def plot_property_distributions(df_generated, df_train):
    """Plots and saves the MW and LogP distributions."""
    logger.info("Plotting property distributions...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot Molecular Weight
    sns.kdeplot(df_train['MW'], label='Training Data', ax=ax1, fill=True, clip=(0, 1000))
    sns.kdeplot(df_generated['MW'], label='Generated Molecules', ax=ax1, fill=True, clip=(0, 1000))
    ax1.set_title('Molecular Weight (MW) Distribution', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Molecular Weight (g/mol)', fontsize=12)
    ax1.legend()
    
    # Plot LogP
    sns.kdeplot(df_train['LogP'], label='Training Data', ax=ax2, fill=True, clip=(-5, 10))
    sns.kdeplot(df_generated['LogP'], label='Generated Molecules', ax=ax2, fill=True, clip=(-5, 10))
    ax2.set_title('LogP Distribution', fontsize=16, fontweight='bold')
    ax2.set_xlabel('LogP', fontsize=12)
    ax2.legend()
    
    fig.suptitle('Generator Property Validation', fontsize=20, y=1.05)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "property_distribution.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"✓ Property distribution plot saved to {save_path}")
    plt.close()

def plot_tsne_chemical_space(fp_generated, fp_train):
    """Plots and saves the 2D t-SNE chemical space."""
    logger.info("Calculating t-SNE... (This can take a few minutes)")
    
    # Combine fingerprints
    fp_all = np.concatenate([fp_train, fp_generated], axis=0)
    
    # Create labels
    labels = ['Training Data'] * len(fp_train) + ['Generated'] * len(fp_generated)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(fp_all)
    
    # Create DataFrame for plotting
    df_tsne = pd.DataFrame(
        data=tsne_results,
        columns=['t-SNE 1', 't-SNE 2']
    )
    df_tsne['Source'] = labels
    
    logger.info("Plotting t-SNE...")
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(
        data=df_tsne,
        x='t-SNE 1',
        y='t-SNE 2',
        hue='Source',
        style='Source',
        s=50,
        alpha=0.6
    )
    
    plt.title('t-SNE Visualization of Chemical Space', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "chemical_space_tsne.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"✓ t-SNE plot saved to {save_path}")
    plt.close()

# --- MAIN EXECUTION ---

def main():
    logger.info("Starting generator analysis...")
    
    # 1. Load models and data
    model, tokenizer, device = load_generator()
    train_smiles = load_training_smiles()
    
    # 2. Generate new molecules
    generated_smiles = generate_molecules(model, tokenizer, device, NUM_TO_GENERATE)
    
    # 3. Calculate properties
    df_generated_props = get_properties(generated_smiles)
    df_train_props = get_properties(train_smiles)

    # 4. Plot property distributions
    plot_property_distributions(df_generated_props, df_train_props)
    
    # 5. Calculate fingerprints
    fp_generated = get_fingerprints(generated_smiles)
    fp_train = get_fingerprints(train_smiles)
    
    # 6. Plot t-SNE
    plot_tsne_chemical_space(fp_generated, fp_train)
    
    # --- 7. Print stats AFTER plotting ---
    print("\n" + "="*30)
    print("--- Training Data Stats ---")
    print(df_train_props.describe())
    print("\n" + "="*30)
    print("--- Generated Data Stats ---")
    print(df_generated_props.describe())
    print("="*30 + "\n")
    logger.info("Analysis complete. All graphs saved to /outputs/ directory.")

if __name__ == "__main__":
    main()