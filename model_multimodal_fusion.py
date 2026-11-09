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

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
DATA_ROOT = os.environ.get("ADMFN_DATA_ROOT", r"D:\trainTest")
# 1.
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
# 超参
POSE_DIM = 79
VISUAL_DIM = 2048#验证1024 512 256 128
EMBED_DIM = 128
BATCH_SIZE = 16
LR = 5e-4
# 2.
EPOCHS = 20
WEIGHT_DECAY = 2e-3

# 输出路径
# RUN_TAG: 结果文件后缀,用于一次跑多组对照而不互相覆盖,例如 RUN_TAG=_bias-4
RUN_TAG = os.environ.get("RUN_TAG", "")
LOG_FILE = f"admfn_multimodal_log{RUN_TAG}.csv"
PLOT_FILE = f"admfn_curves{RUN_TAG}.png"
CURVES_PLOT_FILE = f"admfn_training_curves{RUN_TAG}.png"
MODEL_SAVE_PATH = f"admfn_best_model{RUN_TAG}.pth"

# 门控初始偏置(环境变量可切换):
#   GATE_BIAS=0   中性起步,α0=0.50,视觉满增益参与
#   GATE_BIAS=-2  温和预热,α0≈0.12
#   GATE_BIAS=-4  视觉低增益起步,α0≈0.018(与旧版同量级)
GATE_BIAS = float(os.environ.get("GATE_BIAS", "0"))


# ===================== 工具函数：特征有效性校验 =====================
def is_valid_feature(feat):
    """校验特征是否为数值型且维度正确"""
    # 排除字符串/空值
    if isinstance(feat, str) or feat is None:
        return False
    # 转为numpy数组
    try:
        feat_np = np.array(feat, dtype=np.float32)
    except:
        return False
    # 校验维度（允许1D/2D，展平为1D）
    if len(feat_np.shape) > 2:
        return False
    return True

def extract_visual_feature(pkl_path):
    """安全提取视觉特征，处理各种异常格式"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 情况1：直接是数组
        if is_valid_feature(data):
            feat = np.array(data, dtype=np.float32)
            feat = feat.flatten()  # 展平为1D
            if len(feat) >= VISUAL_DIM:
                return feat[:VISUAL_DIM]  # 截断到2048维
            else:
                return np.resize(feat, (VISUAL_DIM,))  # 补全维度

        # 情况2：字典格式，遍历所有key找数值特征
        elif isinstance(data, dict):
            for k, v in data.items():
                if is_valid_feature(v):
                    feat = np.array(v, dtype=np.float32)
                    feat = feat.flatten()
                    if len(feat) >= VISUAL_DIM:
                        return feat[:VISUAL_DIM]
                    else:
                        return np.resize(feat, (VISUAL_DIM,))

        # 其他情况返回None
        return None
    except Exception as e:
        print(f"⚠️ 读取{pkl_path}失败: {str(e)}")
        return None


# ===================== 标签加载 =====================
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


# ===================== 多模态数据集（骨架特征置信度过滤） =====================
class ADMFN_Dataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.label_dict = load_valid_labels(TRAIN_LABEL_PATH if is_train else VAL_LABEL_PATH)

        # 路径初始化
        if is_train:
            self.kp_dir = POSE_FEATURE_DIRS["keypoints_train"]
            self.geo_dir = POSE_FEATURE_DIRS["geometric_train"]
            self.visual_root = VISUAL_FEATURE_DIRS["train"]
        else:
            self.kp_dir = POSE_FEATURE_DIRS["keypoints_val"]
            self.geo_dir = POSE_FEATURE_DIRS["geometric_val"]
            self.visual_root = VISUAL_FEATURE_DIRS["val"]

        self.samples = self._build_samples()
        print(f"✅ {('训练' if is_train else '验证')}集有效样本数: {len(self.samples)}")

    def _find_pkl(self, img_base):
        """查找对应图片的pkl文件"""
        for folder in os.listdir(self.visual_root):
            folder_path = os.path.join(self.visual_root, folder)
            if not os.path.isdir(folder_path):
                continue
            pkl_path = os.path.join(folder_path, f"{img_base}.pkl")
            if os.path.exists(pkl_path):
                return pkl_path
        return None

    def _build_samples(self):
        """构建有效样本 + 置信度无效样本过滤"""
        samples = []
        invalid_count = 0

        for img_base, label in tqdm(self.label_dict.items(), desc="构建多模态样本"):
            # 1. 检查姿态文件
            kp_path = os.path.join(self.kp_dir, f"{img_base}.npy")
            geo_path = os.path.join(self.geo_dir, f"{img_base}.npy")
            if not os.path.exists(kp_path) or not os.path.exists(geo_path):
                invalid_count += 1
                continue

#           2. 
            kp_data = np.load(kp_path)
            conf = kp_data[2::3]   # 每隔3个取1个，提取所有关键点置信度
            if conf.mean() < 0.2:
                invalid_count += 1
                continue

            # 3. 检查视觉文件
            visual_pkl = self._find_pkl(img_base)
            if visual_pkl is None:
                invalid_count += 1
                continue

            # 4. 预校验视觉特征（提前过滤异常）
            visual_feat = extract_visual_feature(visual_pkl)
            if visual_feat is None:
                invalid_count += 1
                continue

            samples.append((kp_path, geo_path, visual_pkl, float(label)))

        print(f"⚠️ 跳过无效样本数: {invalid_count}")
        return samples

    def __getitem__(self, idx):
        kp_path, geo_path, vis_path, label = self.samples[idx]

        # 加载姿态特征（79维）
        kp = np.load(kp_path).astype(np.float32)
        geo = np.load(geo_path).astype(np.float32)
        pose = np.concatenate([kp, geo])

        # 加载视觉特征（安全提取）
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
        # 门控:仅由骨架条件特征 p 生成,
        # 输出每个样本一个标量 logit
        self.attn = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.ReLU(), nn.Linear(EMBED_DIM, 1))
        # 初始偏置由 GATE_BIAS 控制(默认 0):logit 初值 0 → alpha 初值 0.5(中性起步)
        nn.init.constant_(self.attn[-1].bias, GATE_BIAS)
        self.gate = nn.Sequential(nn.Linear(EMBED_DIM * 2, EMBED_DIM), nn.LayerNorm(EMBED_DIM), nn.ReLU())
        # 记录最近一次前向的门控值,供训练循环统计(不参与梯度)
        self.last_gate = None

    def forward(self, pose, vis):
        p = self.pose_proj(pose)
        v = self.vis_proj(vis)
        # 门控因子 alpha ∈ (0,1):只调节视觉支路,姿态支路全额通过
        # 门控读数仅由骨架条件特征 p 生成,视觉表观统计量无法支配门控分布
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


# ===================== 训练函数 =====================
def run_epoch(model, loader, optimizer, criterion, device, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    preds, labels = [], []
    gate_chunks = []

    with torch.set_grad_enabled(is_train):
        for batch_idx, (pose, visual, lbl) in enumerate(tqdm(loader, desc="训练" if is_train else "验证")):
            # 数据移到设备(gpu)
            pose = pose.to(device)
            visual = visual.to(device)
            lbl = lbl.to(device)

            # 梯度清零
            if is_train:
                optimizer.zero_grad()

            # 前向传播
            output = model(pose, visual)
            loss = criterion(output, lbl)

            # 反向传播
            if is_train:
                loss.backward()
                optimizer.step()

            # 统计
            total_loss += loss.item() * pose.size(0)
            preds.extend(output.detach().cpu().numpy())
            labels.extend(lbl.cpu().numpy())

            # 收集门控值(用于判断门控是否真的在自适应,而非退化为常数)
            if getattr(model.fusion, "last_gate", None) is not None:
                gate_chunks.append(model.fusion.last_gate.view(-1).detach().cpu())

    # 门控统计:均值反映整体参与度,标准差反映"不同样本拿到不同权重"的程度
    if gate_chunks:
        g_all = torch.cat(gate_chunks)
        model.gate_mean = g_all.mean().item()
        model.gate_std = g_all.std().item()
        model.gate_min = g_all.min().item()
        model.gate_max = g_all.max().item()

    # 计算指标
    avg_loss = total_loss / len(loader.dataset)
    pred_labels = (np.array(preds) > 0.5).astype(int)
    acc = accuracy_score(labels, pred_labels)
    auc = roc_auc_score(labels, preds)
    recall = recall_score(labels, pred_labels, zero_division=0)
    return avg_loss, acc, auc, recall


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
# 4.
    print("🔧 GPU 环境检查")
    print("CUDA是否可用:", torch.cuda.is_available())
    print("=" * 60)

    # 加载数据集
    train_dataset = ADMFN_Dataset(is_train=True)
    val_dataset = ADMFN_Dataset(is_train=False)

    # 数据加载器（单线程避免冲突）
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整批次
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # 初始化模型/优化器/损失函数
    model = ADMFN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练记录
    best_auc = 0.0
    log_data = []

    # 训练循环
    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")

        # 训练
        train_loss, train_acc, train_auc, train_recall = run_epoch(
            model, train_loader, optimizer, criterion, DEVICE, is_train=True
        )

        # 验证
        val_loss, val_acc, val_auc, val_recall = run_epoch(
            model, val_loader, None, criterion, DEVICE, is_train=False
        )

        # 更新学习率
        scheduler.step()

        # 记录日志
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

        # 打印结果
        print(f"\n📊 训练结果 | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | Recall: {train_recall:.4f}")
        print(f"📊 验证结果 | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | Recall: {val_recall:.4f}")
        print(f"🚪 门控 | mean: {getattr(model, 'gate_mean', float('nan')):.4f} | std: {getattr(model, 'gate_std', float('nan')):.4f} | min: {getattr(model, 'gate_min', float('nan')):.4f} | max: {getattr(model, 'gate_max', float('nan')):.4f}")

        # 保存最优模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc
            }, MODEL_SAVE_PATH)
            print(f"🏆 保存最优模型 | 验证AUC: {best_auc:.4f}")

    # 保存日志
    df_log = pd.DataFrame(log_data)
    df_log.to_csv(LOG_FILE, index=False, encoding='utf-8')
    print(f"\n🎉 训练完成！最佳验证AUC: {best_auc:.4f}")
    print(f"📄 日志保存至: {LOG_FILE}")
    print(f"💾 最优模型保存至: {MODEL_SAVE_PATH}")

    # ===================== 绘制训练曲线图 =====================
    epochs_range = df_log["epoch"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ADMFN 训练过程曲线", fontsize=16, fontweight='bold')

    # 子图1：损失曲线
    ax = axes[0, 0]
    ax.plot(epochs_range, df_log["train_loss"], marker='o', label="训练集 Loss", color='steelblue')
    ax.plot(epochs_range, df_log["val_loss"],   marker='s', label="验证集 Loss", color='tomato')
    ax.set_title("损失曲线 (Loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 子图2：准确率曲线
    ax = axes[0, 1]
    ax.plot(epochs_range, df_log["train_acc"], marker='o', label="训练集 Accuracy", color='steelblue')
    ax.plot(epochs_range, df_log["val_acc"],   marker='s', label="验证集 Accuracy", color='tomato')
    ax.set_title("准确率曲线 (Accuracy)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 子图3：AUC曲线
    ax = axes[1, 0]
    ax.plot(epochs_range, df_log["train_auc"], marker='o', label="训练集 AUC", color='steelblue')
    ax.plot(epochs_range, df_log["val_auc"],   marker='s', label="验证集 AUC", color='tomato')
    ax.set_title("AUC曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 子图4：Recall曲线
    ax = axes[1, 1]
    ax.plot(epochs_range, df_log["train_recall"], marker='o', label="训练集 Recall", color='steelblue')
    ax.plot(epochs_range, df_log["val_recall"],   marker='s', label="验证集 Recall", 
color='tomato')
    ax.set_title("召回率曲线 (Recall)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(CURVES_PLOT_FILE, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 训练曲线图已保存至: {CURVES_PLOT_FILE}")
