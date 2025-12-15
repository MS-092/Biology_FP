# Protein Secondary Structure Prediction - Computational Biology Final Project

## Members: Matthew, Ryan, Jonathan

This project implements a machine learning pipeline to predict the secondary structure (Alpha Helix, Beta Sheet, Coil) of protein sequences. It features a training script that utilizes multiple machine learning algorithms (including PyTorch Neural Networks) and a rich Streamlit dashboard for visualization and analysis.

## Features

*   **Multi-Model Architecture**:
    *   **Rule-Based:** Uses physicochemical limits (Hydrophobicity/Charge).
    *   **Decision Tree:** Baseline ML interpretation.
    *   **Random Forest:** Robust ensemble tree method.
    *   **HistGradientBoosting:** High-performance boosting algorithm.
    *   **PyTorch Sliding MLP:** Deep Learning model using sliding window context.

*   **Advanced Feature Engineering**: Uses **BLOSUM62** evolutionary substitution matrices + Physicochemical properties.

*   **Interactive Dashboard**:
    *   **Prediction Maps**: Visual comparison of all models.
    *   **Residue Explorer**: Detailed per-residue confidence scores and agreement analysis.
    *   **3D Visualization**: Interactive PDB viewer (using py3Dmol).
    *   **Feature Analysis**: Deep dive into the input vectors seen by the ML models.

## Main File Structure

### Dataset source: https://www.kaggle.com/code/kirkdco/secondary-structure-data-eda/input

The main file structure of the project is as follows:
```text
├── train_all.py          # Script to train and save ML models
├── app.py                # Streamlit dashboard application
├── requirements.txt      # Installing all of the necessary python dependencies
├── dataset2022.csv       # (Required) PISCES protein dataset
├── dataset2018.csv       # (Required) PISCES protein dataset
├── models22.pkl          # (Generated) The trained models file from 2022 dataset
└── models2018.pkl        # (Generated) The trained models file from 2018 dataset
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Models
```bash
# Training all models using the dataset2022.csv file
python train_all.py

# To train specific size for let's say just testing how the application would work
python train_all.py --train_size 2000 --test_size 500

# Arguments:
"""
1. --train_size: Number of proteins to use for training (default: 3000).
2. --test_size: Number of proteins to use for testing (default: 600).
3. --min_len / --max_len: Filter proteins by sequence length.
"""
```

### Running the Dashboard
```bash
# After the trained models file are generated, run the streamlit dashboard
streamlit run app.py
```

## Dashboard Overview

### Input Sidebar:
1. Select an Example protein (e.g., Myoglobin).
2. Paste a raw sequence.
3. Upload a FASTA or TXT file.

### Prediction Maps:
* View colorful "genome-browser" style visualizations of predictions from all models simultaneously.

### Comparison:
1. See a Bar Chart comparison of Helices/Sheets/Coils across models.
2. View an Agreement Heatmap (how closely models match each other).

### Residue Explorer:
1. A detailed table showing every residue, the predictions from every model, and the confidence scores.
2. Filter to see only where models disagree.

### 3D View:
* Visualize the actual 3D structure (fetched from RCSB PDB) if a PDB ID is detected or known.