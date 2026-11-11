import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import warnings
from tqdm import tqdm

import random

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

# ===================== Global configuration =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path configuration
DATA_ROOT = os.environ.get("ADMFN_DATA_ROOT", r"D:\trainTest")
# Skeleton and visual feature directories
POSE_FEATURE_DIRS = {
    "keypoints_train": os.path.join(DATA_ROOT, "51normcharacteristic", "train_result", "features"),
    "keypoints_val": os.path.join(DATA_ROOT, "51normcharacteristic", "validation_result", "features"),
    "geometric_train": os.path.join(DATA_ROOT, "28normcharacteristic", "train_result", "features"),
    "geometric_val": os.path.join(DATA_ROOT, "28normcharacteristic", "validation_result", "features")
}

VISUAL_FEATURE_DIRS = {
    "train": os.path.join(DATA_ROOT, "standardResnet50", "03_Outputs", "resnet_features_engineering", "train_norm"),
    "val": os.path.join(DATA_ROOT, "standardResnet50", "03_Outputs", "resnet_features_engineering", "validation_norm")
}

TRAIN_LABEL_PATH = os.path.join(DATA_ROOT, "labels", "train_match.json")
VAL_LABEL_PATH = os.path.join(DATA_ROOT, "labels", "val_match.json")
# Hyperparameters
POSE_DIM = 79
VISUAL_DIM = 2048  # also validated with 1024, 512, 256, 128
EMBED_DIM = 128
BATCH_SIZE = 16
LR = 5e-4
# Training schedule
EPOCHS = 20
WEIGHT_DECAY = 2e-3

# Output paths
# RUN_TAG: suffix for result files so several runs do not overwrite each other, e.g. RUN_TAG=_bias-4
RUN_TAG = os.environ.get("RUN_TAG", "")
LOG_FILE = f"admfn_multimodal_log{RUN_TAG}.csv"
PLOT_FILE = f"admfn_curves{RUN_TAG}.png"
CURVES_PLOT_FILE = f"admfn_training_curves{RUN_TAG}.png"
MODEL_SAVE_PATH = f"admfn_best_model{RUN_TAG}.pth"

# Initial gate bias (switchable through an environment variable):
#   GATE_BIAS=0   neutral start, alpha_0=0.50, full visual contribution
#   GATE_BIAS=-2  gentle warm-up, alpha_0 approx. 0.12
#   GATE_BIAS=-4  low visual gain at start, alpha_0 approx. 0.018 (same order as the earlier version)
GATE_BIAS = float(os.environ.get("GATE_BIAS", "0"))


# ===================== Utility: feature validation =====================
def is_valid_feature(feat):
    """Check that a feature is numeric and has a valid shape."""
    # Reject strings and empty values
    if isinstance(feat, str) or feat is None:
        return False
    # Convert to a numpy array
    try:
        feat_np = np.array(feat, dtype=np.float32)
    except:
        return False
    # Check the shape (1D/2D allowed, flattened to 1D)
    if len(feat_np.shape) > 2:
        return False
    return True

def extract_visual_feature(pkl_path):
    """Safely load a visual feature and handle unexpected formats."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Case 1: the file contains an array directly
        if is_valid_feature(data):
            feat = np.array(data, dtype=np.float32)
            feat = feat.flatten()  # flatten to 1D
            if len(feat) >= VISUAL_DIM:
                return feat[:VISUAL_DIM]  # truncate to VISUAL_DIM
            else:
                return np.resize(feat, (VISUAL_DIM,))  # pad to VISUAL_DIM

        # Case 2: a dict; scan the values for a numeric feature
        elif isinstance(data, dict):
            for k, v in data.items():
                if is_valid_feature(v):
                    feat = np.array(v, dtype=np.float32)
                    feat = feat.flatten()
                    if len(feat) >= VISUAL_DIM:
                        return feat[:VISUAL_DIM]
                    else:
                        return np.resize(feat, (VISUAL_DIM,))

        # Otherwise return None
        return None
    except Exception as e:
        print(f"Could not read {pkl_path}: {str(e)}")
        return None


# ===================== Label loading =====================
def load_valid_labels(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    valid = {}
    for item in data:
        img_name = item.get("image", "")
        label = item.get("label", -1)
        if not img_name or label not in (0, 1):
            continue
        img_base = os.path.splitext(img_name)[0]
        valid[img_base] = label
    return valid


# ===================== Multimodal dataset (skeleton confidence filtering) =====================
class ADMFN_Dataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.label_dict = load_valid_labels(TRAIN_LABEL_PATH if is_train else VAL_LABEL_PATH)

        # Resolve directories
        if is_train:
            self.kp_dir = POSE_FEATURE_DIRS["keypoints_train"]
            self.geo_dir = POSE_FEATURE_DIRS["geometric_train"]
            self.visual_root = VISUAL_FEATURE_DIRS["train"]
        else:
            self.kp_dir = POSE_FEATURE_DIRS["keypoints_val"]
            self.geo_dir = POSE_FEATURE_DIRS["geometric_val"]
            self.visual_root = VISUAL_FEATURE_DIRS["val"]

        self.samples = self._build_samples()
        print(f"{'Train' if is_train else 'Validation'} samples: {len(self.samples)}")

    def _find_pkl(self, img_base):
        """Find the pkl file matching an image."""
        for folder in os.listdir(self.visual_root):
            folder_path = os.path.join(self.visual_root, folder)
            if not os.path.isdir(folder_path):
                continue
            pkl_path = os.path.join(folder_path, f"{img_base}.pkl")
            if os.path.exists(pkl_path):
                return pkl_path
        return None

    def _build_samples(self):
        """Build valid samples and filter out low-confidence skeletons."""
        samples = []
        invalid_count = 0

        for img_base, label in tqdm(self.label_dict.items(), desc="Building multimodal samples"):
            # 1. Check the skeleton files
            kp_path = os.path.join(self.kp_dir, f"{img_base}.npy")
            geo_path = os.path.join(self.geo_dir, f"{img_base}.npy")
            if not os.path.exists(kp_path) or not os.path.exists(geo_path):
                invalid_count += 1
                continue

            # 2. Check skeleton confidence
            kp_data = np.load(kp_path)
            conf = kp_data[2::3]   # confidence is every third entry
            if conf.mean() < 0.2:
                invalid_count += 1
                continue

            # 3. Check the visual feature file
            visual_pkl = self._find_pkl(img_base)
            if visual_pkl is None:
                invalid_count += 1
                continue

            # 4. Validate the visual feature in advance
            visual_feat = extract_visual_feature(visual_pkl)
            if visual_feat is None:
                invalid_count += 1
                continue

            samples.append((kp_path, geo_path, visual_pkl, float(label)))

        print(f"Skipped invalid samples: {invalid_count}")
        return samples

    def __getitem__(self, idx):
        kp_path, geo_path, vis_path, label = self.samples[idx]

        # Load the 79-dimensional skeleton feature
        kp = np.load(kp_path).astype(np.float32)
        geo = np.load(geo_path).astype(np.float32)
        pose = np.concatenate([kp, geo])

        # Load the visual feature (safe extraction)
        visual = extract_visual_feature(vis_path)

        return (
            torch.from_numpy(pose),
            torch.from_numpy(visual),
            torch.tensor(label, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.samples)



class OneWaySkeletonGateFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_proj = nn.Sequential(nn.Linear(POSE_DIM, EMBED_DIM), nn.LayerNorm(EMBED_DIM), nn.ReLU())
        self.vis_proj = nn.Sequential(nn.Linear(VISUAL_DIM, EMBED_DIM), nn.LayerNorm(EMBED_DIM), nn.ReLU())
        # Gate: generated from the skeleton embedding p alone,
        # one scalar logit per sample
        self.attn = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.ReLU(), nn.Linear(EMBED_DIM, 1))
        # GATE_BIAS sets the initial bias (default 0): logit 0 -> alpha 0.5 (neutral start)
        nn.init.constant_(self.attn[-1].bias, GATE_BIAS)
        self.gate = nn.Sequential(nn.Linear(EMBED_DIM * 2, EMBED_DIM), nn.LayerNorm(EMBED_DIM), nn.ReLU())
        # Store the gate values of the last forward pass for logging (no gradient)
        self.last_gate = None

    def forward(self, pose, vis):
        p = self.pose_proj(pose)
        v = self.vis_proj(vis)
        # Gate factor alpha in (0,1): scales the visual branch only; the skeleton branch passes intact
        # The gate depends on p only; appearance statistics cannot drive its distribution
        alpha = torch.sigmoid(self.attn(p))
        self.last_gate = alpha.detach()
        return self.gate(torch.cat([p, v * alpha], dim=-1))


class ADMFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = OneWaySkeletonGateFusion()
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, pose, vis):
        fusion_feat = self.fusion(pose, vis)
        return self.classifier(fusion_feat).squeeze(-1)


# ===================== Training and evaluation =====================
def run_epoch(model, loader, optimizer, criterion, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    preds, labels = [], []
    gate_chunks = []

    with torch.set_grad_enabled(is_train):
        for batch_idx, (pose, visual, lbl) in enumerate(tqdm(loader, desc="Train" if is_train else "Validation")):
            # Move the batch to the device
            pose = pose.to(device)
            visual = visual.to(device)
            lbl = lbl.to(device)

            # Clear gradients
            if is_train:
                optimizer.zero_grad()

            # Forward pass
            output = model(pose, visual)
            loss = criterion(output, lbl)

            # Backward pass
            if is_train:
                loss.backward()
                optimizer.step()

            # Accumulate statistics
            total_loss += loss.item() * pose.size(0)
            preds.extend(output.detach().cpu().numpy())
            labels.extend(lbl.cpu().numpy())

            # Collect gate values to check that the gate adapts rather than collapses to a constant
            if getattr(model.fusion, "last_gate", None) is not None:
                gate_chunks.append(model.fusion.last_gate.view(-1).detach().cpu())

    # Gate statistics: the mean shows overall participation, the std shows how much the weight varies across samples
    if gate_chunks:
        g_all = torch.cat(gate_chunks)
        model.gate_mean = g_all.mean().item()
        model.gate_std = g_all.std().item()
        model.gate_min = g_all.min().item()
        model.gate_max = g_all.max().item()

    # Compute metrics
    avg_loss = total_loss / len(loader.dataset)
    pred_labels = (np.array(preds) > 0.5).astype(int)
    acc = accuracy_score(labels, pred_labels)
    auc = roc_auc_score(labels, preds)
    recall = recall_score(labels, pred_labels, zero_division=0)
    return avg_loss, acc, auc, recall


# ===================== Main program =====================
if __name__ == "__main__":
    print("=" * 60)
    # GPU environment
    print("Checking the GPU environment")
    print("CUDA available:", torch.cuda.is_available())
    print("=" * 60)

    # Load the datasets
    train_dataset = ADMFN_Dataset(is_train=True)
    val_dataset = ADMFN_Dataset(is_train=False)

    # Data loaders (single worker to avoid shared-memory issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True  # drop the last incomplete batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # Model, optimizer and loss
    model = ADMFN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training record
    best_auc = 0.0
    log_data = []

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")

        # Training
        train_loss, train_acc, train_auc, train_recall = run_epoch(
            model, train_loader, optimizer, criterion, DEVICE, is_train=True
        )

        # Validation
        val_loss, val_acc, val_auc, val_recall = run_epoch(
            model, val_loader, None, criterion, DEVICE, is_train=False
        )

        # Update the learning rate
        scheduler.step()

        # Record the log
        log_data.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "train_recall": train_recall,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_recall": val_recall,
            "gate_mean": getattr(model, "gate_mean", float("nan")),
            "gate_std": getattr(model, "gate_std", float("nan")),
            "gate_min": getattr(model, "gate_min", float("nan")),
            "gate_max": getattr(model, "gate_max", float("nan"))
        })

        # Report the results
        print(f"\nTrain | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | Recall: {train_recall:.4f}")
        print(f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | Recall: {val_recall:.4f}")
        print(f"Gate | mean: {getattr(model, 'gate_mean', float('nan')):.4f} | std: {getattr(model, 'gate_std', float('nan')):.4f} | min: {getattr(model, 'gate_min', float('nan')):.4f} | max: {getattr(model, 'gate_max', float('nan')):.4f}")

        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc
            }, MODEL_SAVE_PATH)
            print(f"Best model saved | validation AUC: {best_auc:.4f}")

    # Save the log
    df_log = pd.DataFrame(log_data)
    df_log.to_csv(LOG_FILE, index=False, encoding='utf-8')
    print(f"\nTraining complete | best validation AUC: {best_auc:.4f}")
    print(f"Log saved to: {LOG_FILE}")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")

    # ===================== Plot training curves =====================
    epochs_range = df_log["epoch"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ADMFN training curves", fontsize=16, fontweight='bold')

    # Panel 1: loss
    ax = axes[0, 0]
    ax.plot(epochs_range, df_log["train_loss"], marker='o', label="Train loss", color='steelblue')
    ax.plot(epochs_range, df_log["val_loss"],   marker='s', label="Validation loss", color='tomato')
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel 2: accuracy
    ax = axes[0, 1]
    ax.plot(epochs_range, df_log["train_acc"], marker='o', label="Train accuracy", color='steelblue')
    ax.plot(epochs_range, df_log["val_acc"],   marker='s', label="Validation accuracy", color='tomato')
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel 3: AUC
    ax = axes[1, 0]
    ax.plot(epochs_range, df_log["train_auc"], marker='o', label="Train AUC", color='steelblue')
    ax.plot(epochs_range, df_log["val_auc"],   marker='s', label="Validation AUC", color='tomato')
    ax.set_title("AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel 4: recall
    ax = axes[1, 1]
    ax.plot(epochs_range, df_log["train_recall"], marker='o', label="Train recall", color='steelblue')
    ax.plot(epochs_range, df_log["val_recall"],   marker='s', label="Validation recall", 
color='tomato')
    ax.set_title("Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(CURVES_PLOT_FILE, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {CURVES_PLOT_FILE}")
