import pandas as pd
import numpy as np
import pickle
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random

# 1. Configuration
DATASET_FILE = "dataset2018.csv"
OUTPUT_MODEL_FILE = "optimized_models2018.pkl"
WINDOW_SIZE = 17

# 2. PyTorch Model Definition
AMINO_ACIDS_TORCH = {
    'A': {'hydro': 1.8,  'mw': 89.1,   'charge': 0},
    'R': {'hydro': -4.5, 'mw': 174.2,  'charge': 1},
    'N': {'hydro': -3.5, 'mw': 132.1,  'charge': 0},
    'D': {'hydro': -3.5, 'mw': 133.1,  'charge': -1},
    'C': {'hydro': 2.5,  'mw': 121.2,  'charge': 0},
    'Q': {'hydro': -3.5, 'mw': 146.2,  'charge': 0},
    'E': {'hydro': -3.5, 'mw': 147.1,  'charge': -1},
    'G': {'hydro': -0.4, 'mw': 75.1,   'charge': 0},
    'H': {'hydro': -3.2, 'mw': 155.2,  'charge': 1},
    'I': {'hydro': 4.5,  'mw': 131.2,  'charge': 0},
    'L': {'hydro': 3.8,  'mw': 131.2,  'charge': 0},
    'K': {'hydro': -3.9, 'mw': 146.2,  'charge': 1},
    'M': {'hydro': 1.9,  'mw': 149.2,  'charge': 0},
    'F': {'hydro': 2.8,  'mw': 165.2,  'charge': 0},
    'P': {'hydro': -1.6, 'mw': 115.1,  'charge': 0},
    'S': {'hydro': -0.8, 'mw': 105.1,  'charge': 0},
    'T': {'hydro': -0.7, 'mw': 119.1,  'charge': 0},
    'W': {'hydro': -0.9, 'mw': 204.2,  'charge': 0},
    'Y': {'hydro': -1.3, 'mw': 181.2,  'charge': 0},
    'V': {'hydro': 4.2,  'mw': 117.1,  'charge': 0},
}
AA_LIST = sorted(AMINO_ACIDS_TORCH.keys())
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

class SlidingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=3, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def encode_window_torch(sequence, center_idx, window_size):
    half = window_size // 2
    vecs = []
    n = len(sequence)
    for i in range(center_idx - half, center_idx + half + 1):
        if i < 0 or i >= n:
            vecs.append(np.zeros(24, dtype=np.float32))
        else:
            aa = sequence[i]
            if aa not in AMINO_ACIDS_TORCH:
                vecs.append(np.zeros(24, dtype=np.float32))
                continue
            prop = AMINO_ACIDS_TORCH[aa]
            pos_norm = i / max(1, n)
            base = [pos_norm, prop['hydro'], prop['mw'], prop['charge']]
            one_hot = [0.0] * 20
            one_hot[AA_TO_IDX[aa]] = 1.0
            vecs.append(np.array(base + one_hot, dtype=np.float32))
    return np.concatenate(vecs)

class SlidingWindowDataset(Dataset):
    def __init__(self, sequences, ss_labels, window_size=17):
        self.samples = []
        all_labels = []
        for seq, ss in zip(sequences, ss_labels):
            if len(seq) != len(ss): continue
            for i in range(len(seq)):
                self.samples.append((encode_window_torch(seq, i, window_size), ss[i]))
                all_labels.append(ss[i])
        
        self.le = LabelEncoder()
        self.le.fit(all_labels)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.le.transform([y])[0], dtype=torch.long)

# 3. Sklearn Constants
BLOSUM62 = {
    'A': [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
    'R': [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
    'N': [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
    'D': [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
    'C': [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    'Q': [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
    'E': [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
    'G': [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
    'H': [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
    'I': [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
    'L': [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
    'K': [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
    'M': [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
    'F': [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
    'P': [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
    'S': [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
    'T': [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
    'W': [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
    'Y': [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
    'V': [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
}
HYDROPHOBICITY = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}

def extract_features_sklearn(sequence, window_size=WINDOW_SIZE):
    n = len(sequence)
    half_win = window_size // 2
    n_feats = (21 * window_size) + 1
    X = np.zeros((n, n_feats), dtype=np.float32)
    padded_seq = 'X' * half_win + sequence + 'X' * half_win
    
    for i in range(n):
        window = padded_seq[i : i + window_size]
        current_feat = []
        for aa in window:
            if aa in BLOSUM62:
                current_feat.extend(BLOSUM62[aa])
                current_feat.append(HYDROPHOBICITY[aa])
            else:
                current_feat.extend([0]*20)
                current_feat.append(0)
        current_feat.append(i / n)
        X[i] = np.array(current_feat, dtype=np.float32)
    return X

def load_data_and_split(train_size, test_size):
    if not os.path.exists(DATASET_FILE):
        print("❌ CSV not found."); return None, None, None, None
    
    print("📂 Loading CSV...")
    df = pd.read_csv(DATASET_FILE)
    df['len'] = df['seq'].apply(lambda x: len(str(x)))
    df = df[(df['len'] >= 30) & (df['len'] <= 700)]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    limit = train_size + test_size
    if len(df) < limit:
        print(f"⚠️ Data limit reached. Using available {len(df)}")
        limit = len(df)
        train_size = int(limit * 0.8)
    
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:limit]
    
    # Extract Raw sequences for PyTorch
    train_seqs = [str(x).upper() for x in df_train['seq']]
    train_lbls = [str(x).upper() for x in df_train['sst3']]
    test_seqs = [str(x).upper() for x in df_test['seq']]
    test_lbls = [str(x).upper() for x in df_test['sst3']]
    
    # Extract Features for Sklearn
    print("   Extracting Sklearn features...")
    X_sk_train, y_sk_train = [], []
    for s, l in zip(train_seqs, train_lbls):
        X_sk_train.append(extract_features_sklearn(s))
        y_sk_train.extend(list(l))
    X_sk_train = np.vstack(X_sk_train)
    y_sk_train = np.array(y_sk_train)
    
    X_sk_test, y_sk_test = [], []
    for s, l in zip(test_seqs, test_lbls):
        X_sk_test.append(extract_features_sklearn(s))
        y_sk_test.extend(list(l))
    X_sk_test = np.vstack(X_sk_test)
    y_sk_test = np.array(y_sk_test)
    
    return (X_sk_train, y_sk_train, X_sk_test, y_sk_test), (train_seqs, train_lbls, test_seqs, test_lbls)

# 4. Main Training Routine
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--test_size", type=int, default=600)
    args = parser.parse_args()
    
    # 1. Load Data
    sklearn_data, torch_data = load_data_and_split(args.train_size, args.test_size)
    if sklearn_data is None: return
    
    X_train_sk, y_train_sk, X_test_sk, y_test_sk = sklearn_data
    train_seqs, train_lbls, test_seqs, test_lbls = torch_data
    
    # 2. Prepare Labels for Sklearn
    le_sk = LabelEncoder()
    y_train_sk_enc = le_sk.fit_transform(y_train_sk)
    y_test_sk_enc = le_sk.transform(y_test_sk)
    
    scaler = StandardScaler()
    X_train_sk_sc = scaler.fit_transform(X_train_sk)
    X_test_sk_sc = scaler.transform(X_test_sk)
    
    # 3. Train Sklearn Models
    print("\n🚀 Training Sklearn Models...")
    models = {}
    metrics = {}
    
    sk_models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=20, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1),
        "HistGradient": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1)
    }
    
    for name, clf in sk_models.items():
        print(f"   Fitting {name}...")
        clf.fit(X_train_sk_sc, y_train_sk_enc)
        acc = clf.score(X_test_sk_sc, y_test_sk_enc)
        models[name] = clf
        metrics[name] = {'accuracy': acc}
        print(f"   ✅ {name} Acc: {acc:.4f}")
        
    # 4. Train PyTorch Model
    print("\n🔥 Training PyTorch MLP (Sliding Window)...")
    train_ds = SlidingWindowDataset(train_seqs, train_lbls, WINDOW_SIZE)
    input_dim = train_ds[0][0].shape[0]
    num_classes = len(train_ds.le.classes_)
    
    torch_model = SlidingMLP(input_dim, [256, 128], num_classes)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop - Epoch
    torch_model.train()
    for epoch in range(15):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = torch_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
        
    # Evaluate PyTorch
    torch_model.eval()
    test_ds = SlidingWindowDataset(test_seqs, test_lbls, WINDOW_SIZE)
    # Important: Force test dataset to use Training LabelEncoder
    test_ds.le = train_ds.le 
    
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = torch_model(xb)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    
    torch_acc = correct / total
    print(f"   ✅ PyTorch MLP Acc: {torch_acc:.4f}")
    
    metrics["PyTorch MLP"] = {'accuracy': torch_acc}
    
    # 5. Save Everything
    # We save the PyTorch model state_dict, not the object
    save_data = {
        'sk_models': models,
        'torch_state': torch_model.state_dict(),
        'torch_classes': train_ds.le.classes_.tolist(),
        'scaler': scaler,
        'le': le_sk,
        'metrics': metrics,
        'input_dim': input_dim
    }
    
    with open(OUTPUT_MODEL_FILE, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"\n💾 Saved all models to {OUTPUT_MODEL_FILE}")

if __name__ == "__main__":
    main()