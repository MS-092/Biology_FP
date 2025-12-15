import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import py3Dmol
from stmol import showmol
import requests
import re
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
import torch
import torch.nn as nn
from io import StringIO
from collections import Counter

# ========================
# 1. Configuration & Constants
# ========================
MODEL_FILE = "models22.pkl"
WINDOW_SIZE = 17

# --- PyTorch Model Class ---
class SlidingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], num_classes=3, dropout=0.3):
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

# --- Feature Data ---
AMINO_ACIDS = {
    'A': {'name': 'Alanine', 'hydro': 1.8, 'mw': 89.1, 'charge': 0, 'type': 'Hydrophobic'},
    'R': {'name': 'Arginine', 'hydro': -4.5, 'mw': 174.2, 'charge': 1, 'type': 'Polar'},
    'N': {'name': 'Asparagine', 'hydro': -3.5, 'mw': 132.1, 'charge': 0, 'type': 'Polar'},
    'D': {'name': 'Aspartic', 'hydro': -3.5, 'mw': 133.1, 'charge': -1, 'type': 'Polar'},
    'C': {'name': 'Cysteine', 'hydro': 2.5, 'mw': 121.2, 'charge': 0, 'type': 'Polar'},
    'Q': {'name': 'Glutamine', 'hydro': -3.5, 'mw': 146.2, 'charge': 0, 'type': 'Polar'},
    'E': {'name': 'Glutamic', 'hydro': -3.5, 'mw': 147.1, 'charge': -1, 'type': 'Polar'},
    'G': {'name': 'Glycine', 'hydro': -0.4, 'mw': 75.1, 'charge': 0, 'type': 'Special'},
    'H': {'name': 'Histidine', 'hydro': -3.2, 'mw': 155.2, 'charge': 1, 'type': 'Polar'},
    'I': {'name': 'Isoleucine', 'hydro': 4.5, 'mw': 131.2, 'charge': 0, 'type': 'Hydrophobic'},
    'L': {'name': 'Leucine', 'hydro': 3.8, 'mw': 131.2, 'charge': 0, 'type': 'Hydrophobic'},
    'K': {'name': 'Lysine', 'hydro': -3.9, 'mw': 146.2, 'charge': 1, 'type': 'Polar'},
    'M': {'name': 'Methionine', 'hydro': 1.9, 'mw': 149.2, 'charge': 0, 'type': 'Hydrophobic'},
    'F': {'name': 'Phenylalanine', 'hydro': 2.8, 'mw': 165.2, 'charge': 0, 'type': 'Hydrophobic'},
    'P': {'name': 'Proline', 'hydro': -1.6, 'mw': 115.1, 'charge': 0, 'type': 'Special'},
    'S': {'name': 'Serine', 'hydro': -0.8, 'mw': 105.1, 'charge': 0, 'type': 'Polar'},
    'T': {'name': 'Threonine', 'hydro': -0.7, 'mw': 119.1, 'charge': 0, 'type': 'Polar'},
    'W': {'name': 'Tryptophan', 'hydro': -0.9, 'mw': 204.2, 'charge': 0, 'type': 'Hydrophobic'},
    'Y': {'name': 'Tyrosine', 'hydro': -1.3, 'mw': 181.2, 'charge': 0, 'type': 'Polar'},
    'V': {'name': 'Valine', 'hydro': 4.2, 'mw': 117.1, 'charge': 0, 'type': 'Hydrophobic'},
}

# --- Sklearn Feature Helpers ---
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

KNOWN_PROTEINS = {
    "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG": "1LYZ",
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH": "1MBN",
    "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR": "1HBA",
}

# ========================
# 2. Logic & Loading
# ========================

@st.cache_resource
def load_data():
    if not os.path.exists(MODEL_FILE): return None
    with open(MODEL_FILE, 'rb') as f: data = pickle.load(f)
    
    # Reconstruct PyTorch Model
    if 'torch_state' in data:
        model = SlidingMLP(data['input_dim'], [256, 128], len(data['torch_classes']))
        model.load_state_dict(data['torch_state'])
        model.eval()
        data['torch_model'] = model
    
    return data

def encode_torch_window(sequence, center_idx, window_size):
    half = window_size // 2
    vecs = []
    n = len(sequence)
    sorted_aa = sorted(AMINO_ACIDS.keys())
    aa_to_idx = {aa: i for i, aa in enumerate(sorted_aa)}
    
    for i in range(center_idx - half, center_idx + half + 1):
        if i < 0 or i >= n:
            vecs.append(np.zeros(24, dtype=np.float32))
        else:
            aa = sequence[i]
            if aa not in AMINO_ACIDS:
                vecs.append(np.zeros(24, dtype=np.float32))
                continue
            prop = AMINO_ACIDS[aa]
            pos_norm = i / max(1, n)
            base = [pos_norm, prop['hydro'], prop['mw'], prop['charge']]
            one_hot = [0.0] * 20
            one_hot[aa_to_idx[aa]] = 1.0
            vecs.append(np.array(base + one_hot, dtype=np.float32))
    return np.concatenate(vecs)

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

def get_residue_profile(sequence, index, window_size=WINDOW_SIZE):
    half_win = window_size // 2
    padded_seq = 'X' * half_win + sequence + 'X' * half_win
    window_seq = padded_seq[index : index + window_size]
    
    return {
        'aa': sequence[index],
        'pos': index + 1, 
        'window': window_seq,
        'hydro': HYDROPHOBICITY.get(sequence[index], 0),
        'mw': AMINO_ACIDS.get(sequence[index], {}).get('mw', 0)
    }

def smooth_predictions(preds):
    cleaned = list(preds)
    n = len(preds)
    for i in range(1, n-1):
        prev_s, curr_s, next_s = cleaned[i-1], cleaned[i], cleaned[i+1]
        if prev_s == next_s and curr_s != prev_s:
            cleaned[i] = prev_s
    final = "".join(cleaned)
    final = re.sub(r'(?<!H)H(?!H)', 'C', final) 
    final = re.sub(r'(?<!E)E(?!E)', 'C', final)
    return final

def predict_rule_based(sequence):
    preds, confs = [], []
    for aa in sequence:
        if aa in ['P', 'G']:
            preds.append('C')
            confs.append(0.85)
        elif aa in AMINO_ACIDS and AMINO_ACIDS[aa]['hydro'] > 1.5:
            preds.append('H')
            confs.append(0.75)
        else:
            preds.append('E')
            confs.append(0.55)
    return "".join(preds), confs

def predict_sklearn(sequence, model, scaler, le):
    X = extract_features_sklearn(sequence)
    X_sc = scaler.transform(X)
    y_idx = model.predict(X_sc)
    raw_pred = "".join(le.inverse_transform(y_idx))
    smoothed_pred = smooth_predictions(raw_pred)
    try:
        proba = model.predict_proba(X_sc)
        conf = [max(p) for p in proba]
    except:
        conf = [0.5] * len(sequence)
    return smoothed_pred, conf

def predict_torch(sequence, model, classes):
    n = len(sequence)
    X_list = [encode_torch_window(sequence, i, WINDOW_SIZE) for i in range(n)]
    X_tensor = torch.tensor(np.stack(X_list), dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    
    preds_idx = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1).tolist()
    
    raw_pred = "".join([classes[i] for i in preds_idx])
    return smooth_predictions(raw_pred), confs

def choose_pdb_id(sequence, header=None):
    if header:
        match = re.search(r'([1-9][A-Z0-9]{3})', header.upper())
        if match: return match.group(1), 'header'
    seq_clean = sequence.replace(" ", "").upper()
    if seq_clean in KNOWN_PROTEINS: return KNOWN_PROTEINS[seq_clean], 'known'
    return "1CRN", 'fallback'

def extract_features_display(sequence):
    features = []
    for aa in sequence:
        prop = AMINO_ACIDS.get(aa, {})
        features.append({
            'AA': aa,
            'Name': prop.get('name', 'Unknown'),
            'Hydrophobicity': prop.get('hydro', 0),
            'MW': prop.get('mw', 0),
            'Charge': prop.get('charge', 0),
            'Type': prop.get('type', 'Unknown')
        })
    return pd.DataFrame(features)

def render_interactive_sequence(sequence, prediction, model_name):
    data = []
    for i, (aa, struct) in enumerate(zip(sequence, prediction)):
        data.append({
            'Index': i + 1,
            'Residue': aa,
            'Structure': struct,
            'Name': AMINO_ACIDS.get(aa, {}).get('name', 'Unknown'),
            'Row': i // 50,
            'Col': i % 50
        })
    df = pd.DataFrame(data)
    
    domain = ['H', 'E', 'C']
    range_ = ['#FF9999', '#9999FF', '#EEEEEE']
    
    base = alt.Chart(df).encode(
        x=alt.X('Col:O', axis=None),
        y=alt.Y('Row:O', axis=None),
        tooltip=['Index', 'Residue', 'Name', 'Structure']
    ).properties(width=700)
    
    rects = base.mark_rect().encode(
        color=alt.Color('Structure', scale=alt.Scale(domain=domain, range=range_), legend=None)
    )
    text = base.mark_text().encode(text='Residue', color=alt.value('black'))
    
    st.altair_chart((rects + text).configure_axis(grid=False).configure_view(strokeWidth=0), use_container_width=True)

# ========================
# 3. Streamlit App
# ========================
def main():
    st.set_page_config(page_title="PSSP Multi-Model Suite", layout="wide", page_icon="🧬")
    st.title("🧬 Protein Secondary Structure Predictor")
    
    # 1. Load Models
    data = load_data()
    if not data:
        st.error(f"❌ `{MODEL_FILE}` not found.")
        st.stop()
        
    sk_models = data['sk_models']
    torch_model = data.get('torch_model')
    torch_classes = data.get('torch_classes')
    le = data['le']
    scaler = data['scaler']
    metrics = data.get('metrics', {})

    # 2. Sidebar Input
    st.sidebar.header("Choices of Input Sequence")
    input_mode = st.sidebar.radio("Source", ["Example", "Paste", "Upload"])
    seq_input, header_input = "", None

    if input_mode == "Example":
        ex_name = st.sidebar.selectbox("Protein", list(KNOWN_PROTEINS.keys()), format_func=lambda x: KNOWN_PROTEINS[x])
        seq_input = ex_name
        header_input = f">PDB:{KNOWN_PROTEINS[ex_name]}"
    elif input_mode == "Paste":
        raw = st.sidebar.text_area("Sequence", height=150)
        if raw:
            if ">" in raw:
                header_input = raw.splitlines()[0]
                seq_input = "".join(raw.splitlines()[1:])
            else:
                seq_input = raw
    elif input_mode == "Upload":
        uploaded = st.sidebar.file_uploader("File", type=["fasta", "txt"])
        if uploaded:
            raw = StringIO(uploaded.getvalue().decode("utf-8")).read()
            if ">" in raw:
                header_input = raw.splitlines()[0]
                seq_input = "".join(raw.splitlines()[1:])
            else:
                seq_input = raw

    seq_input = "".join([c for c in seq_input.upper() if c.isalpha()])
    
    if not seq_input:
        st.info("👈 Please input a sequence.")
        st.stop()

    st.sidebar.success(f"Loaded {len(seq_input)} residues.")
    
    # 3. Run Predictions (All Models)
    with st.spinner("Analyzing..."):
        all_results = {}
        
        # Rule Based
        rp, rc = predict_rule_based(seq_input)
        all_results['Rule-Based'] = {'pred': rp, 'conf': rc, 'acc': "N/A"}
        
        # Sklearn
        for name, clf in sk_models.items():
            mp, mc = predict_sklearn(seq_input, clf, scaler, le)
            # Safe Metric Access
            raw_metric = metrics.get(name, 0)
            acc_val = raw_metric if isinstance(raw_metric, float) else raw_metric.get('accuracy', 0)
            all_results[name] = {'pred': mp, 'conf': mc, 'acc': f"{acc_val:.1%}"}
            
        # PyTorch
        if torch_model:
            tp, tc = predict_torch(seq_input, torch_model, torch_classes)
            raw_metric = metrics.get("PyTorch MLP", 0)
            acc_val = raw_metric if isinstance(raw_metric, float) else raw_metric.get('accuracy', 0)
            all_results["PyTorch MLP"] = {'pred': tp, 'conf': tc, 'acc': f"{acc_val:.1%}"}

    pdb_id, pdb_source = choose_pdb_id(seq_input, header_input)

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Prediction Maps", "📊 Comparison & Metrics", "🔬 Features", "🧬 3D View"])

    # --- TAB 1: PREDICTION MAPS (ALL MODELS) ---
    with tab1:
        st.subheader("Interactive Prediction Maps")
        st.info("Hover over residues for details. Models are stacked for comparison.")
        
        for name, res in all_results.items():
            st.markdown(f"**{name}** (Test Acc: {res['acc']})")
            render_interactive_sequence(seq_input, res['pred'], name)
            st.markdown("---")

    # --- TAB 2: COMPARISON & METRICS ---
    with tab2:
        st.subheader("Structure Composition Comparison")
        
        # Composition Bar Chart
        comp_data = []
        for m, res in all_results.items():
            p = res['pred']
            comp_data.append({'Model': m, 'Type': 'Helix', 'Count': p.count('H'), 'Pct': p.count('H')/len(p)})
            comp_data.append({'Model': m, 'Type': 'Sheet', 'Count': p.count('E'), 'Pct': p.count('E')/len(p)})
            comp_data.append({'Model': m, 'Type': 'Coil', 'Count': p.count('C'), 'Pct': p.count('C')/len(p)})
            
        df_comp = pd.DataFrame(comp_data)
        fig = px.bar(df_comp, x="Model", y="Pct", color="Type", 
                     color_discrete_map={'Helix': '#FF9999', 'Sheet': '#9999FF', 'Coil': '#EEEEEE'},
                     hover_data=['Count'], title="Secondary Structure Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("Interactive Residue Comparison")
        
        # Selector for comparison
        models_to_compare = st.multiselect("Select Models to Compare", list(all_results.keys()), default=list(all_results.keys())[:2])
        
        if models_to_compare:
            # Heatmap
            if len(models_to_compare) > 1:
                matrix = []
                for m1 in models_to_compare:
                    r = []
                    for m2 in models_to_compare:
                        p1 = all_results[m1]['pred']
                        p2 = all_results[m2]['pred']
                        agree = sum(1 for a, b in zip(p1, p2) if a==b) / len(p1)
                        r.append(agree)
                    matrix.append(r)
                
                fig_heat = px.imshow(matrix, x=models_to_compare, y=models_to_compare, text_auto=".2f", 
                                     color_continuous_scale="Viridis", title="Agreement Matrix (1.0 = Identical)")
                st.plotly_chart(fig_heat, use_container_width=True)

            # Table
            data_rows = []
            for i, aa in enumerate(seq_input):
                row = {'Pos': i+1, 'AA': aa}
                preds = [all_results[m]['pred'][i] for m in models_to_compare]
                
                row['Agreement'] = "✅" if len(set(preds)) == 1 else "⚠️"
                
                for m in models_to_compare:
                    row[f"{m}"] = all_results[m]['pred'][i]
                    row[f"{m}_Conf"] = f"{all_results[m]['conf'][i]:.2f}"
                
                data_rows.append(row)
            
            df_res = pd.DataFrame(data_rows)
            
            if st.checkbox("Show Disagreements Only"):
                df_res = df_res[df_res['Agreement'] == "⚠️"]
                
            st.dataframe(df_res, use_container_width=True, height=500)

    # --- TAB 3: FEATURE ANALYSIS ---
    with tab3:
        st.subheader("Biochemical Profile")
        
        # Stats Cards
        feats = [AMINO_ACIDS.get(aa, {}) for aa in seq_input]
        avg_hydro = np.mean([f.get('hydro', 0) for f in feats])
        avg_mw = np.mean([f.get('mw', 0) for f in feats])
        net_charge = sum([f.get('charge', 0) for f in feats])
        hydro_pct = sum(1 for f in feats if f.get('type') == 'Hydrophobic') / len(seq_input)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Hydrophobicity", f"{avg_hydro:.2f}")
        c2.metric("Avg MW", f"{avg_mw:.1f} Da")
        c3.metric("Net Charge", f"{net_charge}")
        c4.metric("Hydrophobic %", f"{hydro_pct:.1%}")
        
        st.divider()
        
        # Profile Viewer
        st.subheader("🔬 Feature Profile Viewer")
        sel_pos = st.number_input("Select Residue Position", 1, len(seq_input), 1)
        idx = sel_pos - 1
        
        profile = get_residue_profile(seq_input, idx)
        
        c_left, c_right = st.columns([1, 2])
        with c_left:
            st.markdown(f"**Residue:** {profile['aa']} ({idx+1})")
            st.markdown(f"**Hydrophobicity:** {profile['hydro']}")
            st.markdown(f"**Molecular Weight:** {profile['mw']}")
            
        with c_right:
            st.markdown("**Window Context (Input to ML):**")
            w = profile['window']
            mid = len(w)//2
            st.code(f"{w[:mid]}[{w[mid]}]{w[mid+1:]}")
            
        st.markdown("### Full Feature Table")
        df_feats = extract_features_display(seq_input)
        df_feats.index = df_feats.index + 1
        st.dataframe(df_feats, use_container_width=True)

    # --- TAB 4: 3D VIEW ---
    with tab4:
        st.subheader("3D Structure")
        
        if pdb_id:
            try:
                # Robust PDB Fetching
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                response = requests.get(url)
                
                if response.status_code == 200:
                    pdb_data = response.text
                    
                    col_ctrl, col_view = st.columns([1, 3])
                    with col_ctrl:
                        st.info(f"Loaded PDB: {pdb_id}")
                        style = st.selectbox("Style", ["Cartoon", "Stick", "Surface"])
                        
                    with col_view:
                        view = py3Dmol.view(width=700, height=500)
                        view.addModel(pdb_data, "pdb")
                        
                        if style == "Cartoon":
                            view.setStyle({'cartoon': {'color': 'spectrum'}})
                        elif style == "Stick":
                            view.setStyle({'stick': {}})
                        else:
                            view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'})
                            view.setStyle({'cartoon': {'color': 'spectrum'}})
                            
                        view.zoomTo()
                        showmol(view, height=500, width=700)
                else:
                    st.error(f"PDB ID '{pdb_id}' not found in RCSB database.")
                    st.warning("Try using a known PDB ID from the example list.")
            except Exception as e:
                st.error(f"Error loading 3D view: {e}")
        else:
            st.warning("No PDB ID identified for this sequence.")

if __name__ == "__main__":
    main()
