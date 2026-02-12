"""
Sensitivity analysis: evaluate ANN and Physics-Aware ANN under demand perturbations.
Simulates population change by scaling demand, pressure, and flow in the test set.
Run this script to populate Table 1 (Sensitivity Analysis) in the paper.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import tqdm

# Paths
current = os.getcwd()
workspace = os.path.join(current, "..")
files_dir = os.path.join(workspace, "files")
log_dir = os.path.join(workspace, "logs")

WINDOW_SIZE = 12
BATCH_SIZE = 256
BETA = 0.7  # pressure-demand coupling (paper: 0.7)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SCENARIOS = [3, 5, 11, 30, 49, 61, 64, 71, 77, 78, 80, 89, 93, 108, 109, 122, 123, 127, 140, 143, 156, 160, 164, 183, 190, 192, 231, 239, 248, 257, 262, 272, 286, 295, 300, 306, 311, 320, 325, 331, 332, 333, 351, 370, 430, 436, 447, 448, 466, 501, 533, 534, 539, 543, 557, 563, 573, 574, 580, 610, 612, 618, 626, 642, 644, 651, 668, 673, 677, 683, 686, 690, 691, 692, 694, 699, 722, 725, 731, 732, 754, 755, 780, 787, 812, 814, 815, 835, 854, 862, 866, 867, 873, 875, 887, 910, 914, 930, 951, 952, 963, 965, 981, 994, 995]

# Feature layout: 32 demand, 32 pressure, 34 flow, sin_hour, cos_hour
N_DEMAND, N_PRESSURE, N_FLOW = 32, 32, 34


def load_scenario_data(scenario_id):
    path = os.path.join(workspace, f"Scenario-{scenario_id}.csv")
    if not os.path.exists(path):
        return None
    data = pd.read_csv(path)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    return data


def add_temporal_columns(df):
    step_of_day = np.array([i % 48 for i in range(len(df))])
    df["sin_hour"] = np.sin(2 * np.pi * step_of_day / 48)
    df["cos_hour"] = np.cos(2 * np.pi * step_of_day / 48)
    return df


def perturb_features(arr, alpha, beta=BETA):
    """Perturb demand, pressure, flow. arr: (T, F) with F=100."""
    arr = arr.copy()
    # Demand: columns 0:32
    arr[:, 0:32] = arr[:, 0:32] * (1 + alpha)
    # Pressure: columns 32:64
    arr[:, 32:64] = arr[:, 32:64] * (1 - beta * alpha)
    # Flow: columns 64:98
    arr[:, 64:98] = arr[:, 64:98] * (1 + alpha)
    # sin_hour, cos_hour unchanged
    return arr


class WindowedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    # Load training mean/std (ANN and Physics-Aware ANN may use same or different scalers)
    mean_path = os.path.join(files_dir, "mean_ann.txt")
    std_path = os.path.join(files_dir, "std_ann.txt")
    mean_phys_path = os.path.join(files_dir, "mean_ann_physics.txt")
    std_phys_path = os.path.join(files_dir, "std_ann_physics.txt")
    if not os.path.exists(mean_path):
        print("Run ann.py first to generate mean_ann.txt and std_ann.txt")
        return
    mean = np.loadtxt(mean_path).astype(np.float32)
    std = np.loadtxt(std_path).astype(np.float32)
    mean_phys = np.loadtxt(mean_phys_path).astype(np.float32) if os.path.exists(mean_phys_path) else mean
    std_phys = np.loadtxt(std_phys_path).astype(np.float32) if os.path.exists(std_phys_path) else std

    # Load test data
    test_dfs = []
    for sid in tqdm.tqdm(TEST_SCENARIOS):
        df = load_scenario_data(sid)
        if df is None:
            continue
        if "Timestamps" in df.columns:
            df = df.drop(columns=["Timestamps"])
        if "Leaks" not in df.columns:
            continue
        df = add_temporal_columns(df)
        test_dfs.append((df, sid))

    if len(test_dfs) == 0:
        print("No test data found.")
        return

    # Build base windows (alpha=0)
    X_list, y_list = [], []
    for df, _ in test_dfs:
        arr = df.drop(columns=["Leaks"]).values.astype(np.float32)
        labs = df["Leaks"].values.astype(np.int64)
        T, F = arr.shape
        for start in range(0, T - WINDOW_SIZE + 1, 1):
            window = arr[start : start + WINDOW_SIZE]
            # Perturbation will be applied per alpha
            X_list.append(window)
            y_list.append(labs[start + WINDOW_SIZE - 1])
    y_all = np.array(y_list)

    # Load ANN model
    ann_path = os.path.join(files_dir, "_ann_last_ann_physics.pth")
    if not os.path.exists(ann_path):
        ann_path = os.path.join(files_dir, "_ann_last.pth")
    # Try to find any saved ann
    if not os.path.exists(ann_path):
        for f in os.listdir(files_dir):
            if "ann" in f and "physics" not in f and f.endswith(".pth"):
                ann_path = os.path.join(files_dir, f)
                break

    # Import model classes
    import sys
    sys.path.insert(0, current)
    from ann import LeakANN
    from ann_physics import LeakPhysicsANN

    input_dim = WINDOW_SIZE * 100  # 1200
    ann_model = LeakANN(input_dim).to(DEVICE)
    if os.path.exists(ann_path):
        ann_model.load_state_dict(torch.load(ann_path, map_location=DEVICE))
    ann_model.eval()

    # Physics ANN
    phys_path = os.path.join(files_dir, "_ann_last_physics.pth")
    for f in os.listdir(files_dir):
        if "ann" in f and "physics" in f and f.endswith(".pth"):
            phys_path = os.path.join(files_dir, f)
            break
    phys_model = LeakPhysicsANN(input_dim).to(DEVICE)
    if os.path.exists(phys_path):
        phys_model.load_state_dict(torch.load(phys_path, map_location=DEVICE))
    phys_model.eval()

    alphas = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    results = {"ANN": {}, "Physics-Aware ANN": {}}

    def norm_window(w, m, s):
        return ((w - m) / (s + 1e-6)).astype(np.float32)

    for alpha in alphas:
        X_ann, X_phys = [], []
        for w in X_list:
            w_pert = perturb_features(w, alpha)
            X_ann.append(norm_window(w_pert, mean, std).reshape(-1))
            X_phys.append(norm_window(w_pert, mean_phys, std_phys).reshape(-1))
        X_ann = np.array(X_ann, dtype=np.float32)
        X_phys = np.array(X_phys, dtype=np.float32)

        ds_ann = WindowedDataset(X_ann, y_all)
        ds_phys = WindowedDataset(X_phys, y_all)
        loader_ann = DataLoader(ds_ann, batch_size=BATCH_SIZE, shuffle=False)
        loader_phys = DataLoader(ds_phys, batch_size=BATCH_SIZE, shuffle=False)

        ann_preds, phys_preds = [], []
        with torch.no_grad():
            for xb, _ in loader_ann:
                ann_preds.append(ann_model(xb.to(DEVICE)).cpu().numpy())
            for xb, _ in loader_phys:
                lp, _, _ = phys_model(xb.to(DEVICE))
                phys_preds.append(lp.cpu().numpy())

        ann_preds = np.concatenate(ann_preds)
        phys_preds = np.concatenate(phys_preds)
        ann_labels = (ann_preds >= 0.5).astype(int)
        phys_labels = (phys_preds >= 0.5).astype(int)

        _, _, ann_f1, _ = precision_recall_fscore_support(y_all, ann_labels, average="binary", zero_division=0)
        _, _, phys_f1, _ = precision_recall_fscore_support(y_all, phys_labels, average="binary", zero_division=0)

        results["ANN"][alpha] = ann_f1
        results["Physics-Aware ANN"][alpha] = phys_f1
        print(f"alpha={alpha:+.2f}: ANN F1={ann_f1:.4f}, Physics-Aware ANN F1={phys_f1:.4f}")

    # Write results for paper
    out_path = os.path.join(log_dir, "sensitivity_analysis_results.txt")
    os.makedirs(log_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Sensitivity Analysis Results\n")
        f.write("=" * 60 + "\n")
        for alpha in alphas:
            f.write(f"alpha={alpha:+.2f}: ANN F1={results['ANN'][alpha]:.4f}, PAANN F1={results['Physics-Aware ANN'][alpha]:.4f}\n")
        f.write("\nLaTeX table row (ANN):\n")
        f.write(" & ".join([f"{results['ANN'][a]:.4f}" for a in alphas]) + " \\\\\n")
        f.write("\nLaTeX table row (Physics-Aware ANN):\n")
        f.write(" & ".join([f"{results['Physics-Aware ANN'][a]:.4f}" for a in alphas]) + " \\\\\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
