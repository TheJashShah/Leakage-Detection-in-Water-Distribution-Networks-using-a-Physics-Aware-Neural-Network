import os, random, tqdm, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, global_mean_pool
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np

import os 
dir = os.getcwd()
leakdb = os.path.join(dir, "..")
text = os.path.join(leakdb, "txt")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def load_scenario_data(scenario_id, base_path=r"D:\LeakDB_full_data\Hanoi"):
    try:
        scenario_path = os.path.join(base_path, f"Scenario-{scenario_id}")
        if not os.path.exists(scenario_path):
            print(f"Scenario {scenario_id} not found.")
            return None
        
        leaks, timestamps = None, None
        demand_path = flow_path = pressure_path = None
        for sub in os.listdir(scenario_path):
            if sub in [f"Scenario-{scenario_id}", f"Scenario-{scenario_id}_info.csv", f"Hanoi_CMH_Scenario-{scenario_id}.inp"]:
                continue
            sub_path = os.path.join(scenario_path, sub)
            if sub == "Demands":
                demand_path = sub_path
            elif sub == "Flows":
                flow_path = sub_path
            elif sub == "Pressures":
                pressure_path = sub_path
            elif sub == "Labels.csv":
                leaks = pd.read_csv(sub_path).drop(columns=["Index"], errors="ignore")
            elif sub == "Timestamps.csv":
                timestamps = pd.read_csv(sub_path).drop(columns=["Index"], errors="ignore")

        if not all([demand_path, flow_path, pressure_path, leaks is not None, timestamps is not None]):
            print(f"Scenario {scenario_id} is incomplete.")
            return None

        df = pd.concat([leaks, timestamps], axis=1, ignore_index=True)
        df.columns = ["Leaks", "Timestamps"]

        def combined_feature_df(path, feature):
            dfs = []
            for file in sorted(os.listdir(path)):
                file_path = os.path.join(path, file)
                if not file.endswith(".csv"):
                    continue
                sub_df = pd.read_csv(file_path).drop(columns="Index", errors="ignore")
                sub_df.columns = [f"{feature}_{file.split('.')[0]}"]
                dfs.append(sub_df)
            return pd.concat(dfs, axis=1, ignore_index=True)

        demand_df = combined_feature_df(demand_path, "demand")
        pressure_df = combined_feature_df(pressure_path, "pressure")
        flow_df = combined_feature_df(flow_path, "flow")

        demand_df.columns = [f"demand_node_{i}" for i in range(1, demand_df.shape[1] + 1)]
        pressure_df.columns = [f"pressure_node_{i}" for i in range(1, pressure_df.shape[1] + 1)]
        flow_df.columns = [f"flow_link_{i}" for i in range(1, flow_df.shape[1] + 1)]

        final_df = pd.concat([demand_df, pressure_df, flow_df, df], axis=1)
        final_df["Leaks"] = final_df["Leaks"].astype(int)
        return final_df

    except Exception as e:
        print(f"Error loading scenario {scenario_id}: {e}")
        return None
    
def aggregate_nonleak_rows(df, leak_col="Leaks"):
    data_cols = [c for c in df.columns if c != "Timestamps" and c != leak_col]
    out_rows = []
    i = 0
    n = len(df)
    while i < n:
        if df.iloc[i][leak_col] == 1:
            out_rows.append(df.iloc[i][data_cols + [leak_col]].to_dict())
            i += 1
            continue
        j = i
        while j < n and df.iloc[j][leak_col] == 0:
            j += 1
        run_len = j - i
        k = i
        while k + 3 <= j:
            window = df.iloc[k:k+3]
            avg_vals = window[data_cols].mean(axis=0).to_dict()
            avg_vals[leak_col] = 0
            out_rows.append(avg_vals)
            k += 3
        while k < j:
            out_rows.append(df.iloc[k][data_cols + [leak_col]].to_dict())
            k += 1
        i = j
    out_df = pd.DataFrame(out_rows)
    return out_df

def compute_global_mean_std(df_list):
    sum_ = None
    sumsq_ = None
    n_total = 0
    for df in df_list:
        arr = df.drop(columns=["Leaks"]).values.astype(np.float64)
        if sum_ is None:
            sum_ = arr.sum(axis=0)
            sumsq_ = (arr**2).sum(axis=0)
        else:
            sum_ += arr.sum(axis=0)
            sumsq_ += (arr**2).sum(axis=0)
        n_total += arr.shape[0]
    mean = sum_ / n_total
    var = (sumsq_ / n_total) - (mean**2)
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)

def normalize_df(df, mean, std):
    cols = [c for c in df.columns if c != "Leaks"]
    df2 = df.copy()
    df2[cols] = (df2[cols] - mean) / std
    return df2

def load_edge_index(pipes_path):
    edges = []
    with open(pipes_path, "r") as file:
        for line in file:
            if not line.strip() or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 3: continue
            try:
                edges.append([int(parts[1]) - 1, int(parts[2]) - 1])
            except:
                continue
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def build_graph_from_row(row, edge_index, num_nodes=32):
    node_feats = [[row.get(f"demand_node_{i+1}", 0.0),
                   row.get(f"pressure_node_{i+1}", 0.0)] for i in range(num_nodes)]
    x = torch.tensor(node_feats, dtype=torch.float)

    num_edges = edge_index.shape[1]
    edge_feats = [[row.get(f"flow_link_{i+1}", 0.0)] for i in range(num_edges)]
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    y = torch.tensor([row.get("Leaks", 0)], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class ScenarioGraphSequenceDataset(Dataset):
    def __init__(self, df_list, edge_index, num_nodes=32, seq_len=5):
        super().__init__()
        self.samples = []
        self.edge_index = edge_index
        for df in tqdm.tqdm(df_list, desc="Preparing sequences"):
            for i in range(len(df) - seq_len + 1):
                seq_df = df.iloc[i:i+seq_len]
                label = float(seq_df.iloc[-1]["Leaks"])
                seq_graphs = [build_graph_from_row(row, edge_index, num_nodes)
                              for _, row in seq_df.iterrows()]
                self.samples.append((seq_graphs, label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def stgnn_collate_fn(batch):
    seqs, labels = zip(*batch)
    seq_len = len(seqs[0])
    batch_by_timestep = [[seq[t] for seq in seqs] for t in range(seq_len)]
    labels = torch.tensor(labels, dtype=torch.float)
    return batch_by_timestep, labels

class LeakSTGNN(nn.Module):
    def __init__(self, node_in=2, edge_in=1, hidden_gnn=64, hidden_gru=32, dropout=0.3):
        super().__init__()
        nn_edge = nn.Sequential(nn.Linear(edge_in, 16), nn.ReLU(),
                                nn.Linear(16, node_in * hidden_gnn))
        self.conv = NNConv(node_in, hidden_gnn, nn_edge, aggr='mean')
        self.gru = nn.GRU(hidden_gnn, hidden_gru, batch_first=True)
        self.fc = nn.Linear(hidden_gru, 1)
        self.dropout = dropout

    def forward(self, x_list, edge_index_list, edge_attr_list, batch_list):
        gnn_outs = []
        for x, edge_index, edge_attr, batch in zip(x_list, edge_index_list, edge_attr_list, batch_list):
            x = F.relu(self.conv(x, edge_index, edge_attr))
            pooled = global_mean_pool(x, batch)
            gnn_outs.append(pooled.unsqueeze(1))
        seq = torch.cat(gnn_outs, dim=1)
        _, h = self.gru(seq)
        return torch.sigmoid(self.fc(h[-1])).squeeze(1)

def train_stgnn_pipeline(
    pipes_path, base_path, scenario_ids, seq_len=5,
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    epochs=20, batch_size=4, lr=1e-3, dropout=0.3,
    device=None, num_nodes=32, num_workers=0
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_dfs = []
    for sid in tqdm.tqdm(scenario_ids, desc="Loading scenarios"):
        df = load_scenario_data(sid, base_path)
        if df is None or len(df) == 0: continue
        if "Timestamps" in df.columns: df = df.drop(columns=["Timestamps"])
        df = aggregate_nonleak_rows(df)
        df.reset_index(drop=True, inplace=True)
        all_dfs.append(df)

    N = len(all_dfs)
    idxs = np.arange(N); np.random.shuffle(idxs)
    n_train, n_val = int(N*train_ratio), int(N*val_ratio)
    train_dfs = [all_dfs[i] for i in idxs[:n_train]]
    val_dfs   = [all_dfs[i] for i in idxs[n_train:n_train+n_val]]
    test_dfs  = [all_dfs[i] for i in idxs[n_train+n_val:]]

    mean, std = compute_global_mean_std(train_dfs)
    np.savetxt("mean_stgnn_fast.txt", mean); np.savetxt("std_stgnn_fast.txt", std)
    train_dfs = [normalize_df(df, mean, std) for df in train_dfs]
    val_dfs   = [normalize_df(df, mean, std) for df in val_dfs]
    test_dfs  = [normalize_df(df, mean, std) for df in test_dfs]

    edge_index = load_edge_index(pipes_path)
    train_set = ScenarioGraphSequenceDataset(train_dfs, edge_index, num_nodes, seq_len)
    val_set   = ScenarioGraphSequenceDataset(val_dfs, edge_index, num_nodes, seq_len)
    test_set  = ScenarioGraphSequenceDataset(test_dfs, edge_index, num_nodes, seq_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=stgnn_collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              collate_fn=stgnn_collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              collate_fn=stgnn_collate_fn, num_workers=num_workers, pin_memory=True)

    model = LeakSTGNN(dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    best_val_f1 = 0
    model_path = "best_leak_stgnn_fast.pth"

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for seq_graphs, labels in train_loader:
            x_list, edge_index_list, edge_attr_list, batch_list = [], [], [], []
            for seq_t in seq_graphs:
                batch = Batch.from_data_list(seq_t).to(device)
                x_list.append(batch.x); edge_index_list.append(batch.edge_index)
                edge_attr_list.append(batch.edge_attr); batch_list.append(batch.batch)
            labels = labels.to(device)
            preds = model(x_list, edge_index_list, edge_attr_list, batch_list)
            loss = criterion(preds, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)

        model.eval()
        preds_all, labs_all = [], []
        with torch.no_grad():
            for seq_graphs, labels in val_loader:
                x_list, edge_index_list, edge_attr_list, batch_list = [], [], [], []
                for seq_t in seq_graphs:
                    batch = Batch.from_data_list(seq_t).to(device)
                    x_list.append(batch.x); edge_index_list.append(batch.edge_index)
                    edge_attr_list.append(batch.edge_attr); batch_list.append(batch.batch)
                labels = labels.to(device)
                preds = model(x_list, edge_index_list, edge_attr_list, batch_list)
                preds_all.append(preds.cpu().numpy()); labs_all.append(labels.cpu().numpy())
        preds_all, labs_all = np.concatenate(preds_all), np.concatenate(labs_all)
        pred_labels = (preds_all >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labs_all, pred_labels, average="binary", zero_division=0)

        print(f"Epoch {epoch}/{epochs}  TrainLoss={total_loss:.4f}  ValF1={f1:.4f}")
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), model_path)
            
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")


    print(f"Training done. Best Val F1: {best_val_f1:.4f}")
    return model


if __name__ == "__main__":
    pipes_path = r"C:\Users\Jash\OneDrive\Desktop\Research-Project\LeakDB\txt\inp_1_text.txt"
    model = train_stgnn_pipeline(pipes_path=pipes_path,
                             base_path=r"D:\\LeakDB_full_data\\Hanoi",
                             scenario_ids=list(range(1, 41)))
