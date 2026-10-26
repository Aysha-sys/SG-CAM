# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SG-CAM on PatternNet — reproducible experiment notebook (PeerJ CS cs-132552, revision 3)
#
# **Semantic–visual feature fusion via Segmentation-Guided Cross-Attention (SG-CAM)**
# MobileNetV2 visual stream + frozen SegFormer-B2 semantic stream + SG-CAM fusion, trained and evaluated on PatternNet (38 classes).
#
# This notebook regenerates **every PatternNet table and figure** of the manuscript from logged predictions:
#
# | Manuscript item | Produced here |
# |---|---|
# | Table 3 (training parameters) | `tables/T3_training_parameters.csv` (written from the actual config) |
# | Table 4 (main results, PatternNet row) | `tables/T4_main_results.csv` — mean ± std over seeds |
# | Per-class results | `tables/T_per_class_metrics.csv` |
# | Table 5 (complexity) | `tables/T5_complexity_measured.csv` — measured params / GFLOPs / latency |
# | Table 7 (backbones, with vs. without SG-CAM) | `tables/T7_backbones.csv` |
# | Table 8 (ablation, text labels) | `tables/T8_ablation.csv` (+ McNemar tests) |
# | Table 9 (sensitivity, selected on **validation** split) | `tables/T9_sensitivity.csv` |
# | Fig. 2 preprocessing | `figures/F2_preprocessing.png` |
# | Fig. 4 semantic maps / attention | `figures/F4_semantic_attention.png` |
# | Fig. 6 confusion matrix | `figures/F6_confusion_matrix_*.png` |
# | Fig. 8 per-class precision/recall/F1 | `figures/F8_per_class_metrics.png` |
# | Fig. 10 ROC (macro one-vs-rest) + PR curves | `figures/F10_roc.png`, `figures/F10b_pr.png` |
# | Misclassified examples (Reviewer 3, comment 4) | `figures/F_misclassified.png` |
# | Training curves, t-SNE, ablation & sensitivity plots | `figures/…` |
#
# ### Protocol (fixes the issues raised in review)
# * Input at PatternNet's native **256×256** resolution; stratified **60 / 20 / 20** train / validation / test split (fixed split seed, saved to `splits/patternnet_split.csv`).
# * Model selection (best epoch) and all hyper-parameter sensitivity analyses use the **validation** split only; the test split is touched once per run.
# * Every configuration is trained with several random seeds and reported as **mean ± std**.
# * SegFormer-B2 (ADE20K) is used as a **frozen** semantic encoder; its all-MLP decoder representation `F_dec` supplies keys/values. The loss is **cross-entropy only** (no segmentation loss — PatternNet has no pixel labels).
# * SG-CAM: `Q = W_Q F_img`, `K = W_K F_seg`, `V = W_V F_seg`, `A = softmax(QKᵀ/√d)`, `F_cam = A V`, `F_out = LN(F_img + W_O F_cam)`.
# * All tables/figures are computed from saved per-image predictions/probabilities (`results/preds/*.csv.gz`).
#
# ### How to run on Kaggle
# 1. **Settings → Accelerator: GPU (T4 or P100)**, **Internet: ON** (downloads ImageNet / ADE20K weights).
# 2. **Add Input → Datasets → search “PatternNet”** and add it (any copy with the 38 class folders works; the path is auto-detected).
# 3. Choose which experiment groups to run in `RUN_GROUPS` (cell 2). A full run does not fit in one 12-h session — run groups across sessions with **Save Version → Save & Run All**.
# 4. To continue in a new session, **Add Input → Notebook Output** of the previous version: finished runs are detected and skipped automatically, and all tables/figures are rebuilt from the union of results.

# %%
import importlib, subprocess, sys
for _pkg in ["transformers"]:
    try:
        importlib.import_module(_pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)

# %%
import os, json, time, math, glob, copy, random, shutil, hashlib, platform, warnings, zipfile
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import OrderedDict
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score, cohen_kappa_score,
                             roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, top_k_accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy import stats
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

SMOKE = os.environ.get("SGCAM_SMOKE", "0") == "1"   # internal self-test on synthetic data (leave off on Kaggle)

CFG = dict(
    # ---------------- paths ----------------
    DATA_ROOT=None,                         # None = auto-detect PatternNet under /kaggle/input
    OUT_DIR="/kaggle/working/sgcam_outputs",
    CACHE_DIR="/tmp/sgcam_cache",           # frozen-SegFormer feature cache (~12 GB per preprocessing setting)
    # ---------------- data ----------------
    IMG_SIZE=256,                           # PatternNet native resolution (256 x 256); no up-sampling
    SPLIT=(0.6, 0.2, 0.2),                  # train / val / test (stratified)
    SPLIT_SEED=42,
    CLAHE_CLIP=2.0, CLAHE_TILE=8,           # CLAHE on L channel of LAB
    # ---------------- model ----------------
    SEGFORMER="nvidia/segformer-b2-finetuned-ade-512-512",
    D_ATTN=256,                             # SG-CAM projection dimension d
    DROPOUT=0.5,
    PRETRAINED=True,
    # ---------------- training (Table 3) ----------------
    OPTIMIZER="Adam", LR=1e-4, BATCH=16, EPOCHS=15, STEP_SIZE=10, GAMMA=0.1, WEIGHT_DECAY=0.0,
    SEEDS=[0, 1, 2],
    BACKBONE_SEEDS=[0, 1, 2],
    SWEEP_EPOCHS=10,                        # shorter schedule for the sensitivity analysis (validation split)
    AMP=True, NUM_WORKERS=4,
    CACHE_BATCH=32,                         # SegFormer batch for feature caching (use 8 if IMG_SIZE=512 on a T4)
    SAVE_CHECKPOINTS=True,
    # ---------------- what to run in this session ----------------
    # groups: "main", "transfer_baselines", "robustness_extra", "ablation", "sensitivity", "backbones"  ([] = report only)
    # Attach the output of the previous SGCAM run as an input: finished runs are imported and skipped.
    RUN_GROUPS=["main", "transfer_baselines", "robustness_extra", "ablation", "sensitivity", "backbones"],
    MAX_SESSION_HOURS=9.5,                  # hard stop for training; Kaggle discards ALL output of a run cancelled at 12 h
)

if SMOKE:
    _s = "/tmp/sgcam_smoke"
    CFG.update(DATA_ROOT=f"{_s}/data/PatternNet/images", OUT_DIR=f"{_s}/out", CACHE_DIR=f"{_s}/cache",
               IMG_SIZE=64, EPOCHS=2, SWEEP_EPOCHS=1, SEEDS=[0, 1], BACKBONE_SEEDS=[0], BATCH=8,
               NUM_WORKERS=0, PRETRAINED=False, AMP=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
OUT = CFG["OUT_DIR"]
for sub in ["results/json", "results/preds", "results/feats", "results/history", "checkpoints",
            "tables", "figures", "splits", "logs"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
os.makedirs(CFG["CACHE_DIR"], exist_ok=True)
SESSION_T0 = time.time()
print("Device:", DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("torch", torch.__version__, "| torchvision", torchvision.__version__)

# %% [markdown]
# ## 1. Dataset discovery and stratified 60/20/20 split

# %%
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

def _make_smoke_dataset(root, n_cls=4, n_per=25, size=64):
    if os.path.isdir(root) and len(os.listdir(root)) >= n_cls:
        return
    rng = np.random.RandomState(0)
    names = ["forest", "river", "parking_lot", "solar_panel"][:n_cls]
    for ci, c in enumerate(names):
        os.makedirs(os.path.join(root, c), exist_ok=True)
        for i in range(n_per):
            base = np.zeros((size, size, 3), np.uint8)
            base[..., ci % 3] = 120 + 40 * ci
            noise = rng.randint(0, 90, (size, size, 3)).astype(np.uint8)
            img = cv2.add(base, noise)
            if ci == 1: cv2.line(img, (0, i % size), (size, (i * 3) % size), (255, 255, 255), 3)
            if ci == 2: cv2.rectangle(img, (8, 8), (40, 40), (30, 30, 30), -1)
            cv2.imwrite(os.path.join(root, c, f"{c}{i:03d}.jpg"), img)

def find_dataset_root(base="/kaggle/input", min_classes=20):
    best, best_n = None, 0
    for dirpath, dirnames, filenames in os.walk(base):
        if "sgcam_outputs" in dirpath:
            continue
        if len(dirnames) >= min_classes:
            n_ok = 0
            for d in dirnames[:60]:
                fs = os.listdir(os.path.join(dirpath, d))
                if any(f.lower().endswith(IMG_EXT) for f in fs[:50]):
                    n_ok += 1
            if n_ok >= min_classes and n_ok > best_n:
                best, best_n = dirpath, n_ok
    return best

if SMOKE:
    _make_smoke_dataset(CFG["DATA_ROOT"], size=CFG["IMG_SIZE"])
DATA_ROOT = CFG["DATA_ROOT"] or find_dataset_root()
assert DATA_ROOT is not None, "PatternNet not found: add the PatternNet dataset via 'Add Input'."
CLASSES = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
NUM_CLASSES = len(CLASSES)
print("Dataset root:", DATA_ROOT, "| classes:", NUM_CLASSES)

rows = []
for ci, c in enumerate(CLASSES):
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, c))):
        if f.lower().endswith(IMG_EXT):
            rows.append(dict(path=os.path.join(DATA_ROOT, c, f), rel=os.path.join(c, f), cls=c, label=ci))
DF = pd.DataFrame(rows)
tr_f, va_f, te_f = CFG["SPLIT"]
idx_tr, idx_tmp = train_test_split(np.arange(len(DF)), test_size=va_f + te_f, stratify=DF.label,
                                   random_state=CFG["SPLIT_SEED"])
idx_va, idx_te = train_test_split(idx_tmp, test_size=te_f / (va_f + te_f), stratify=DF.label.values[idx_tmp],
                                  random_state=CFG["SPLIT_SEED"])
DF["split"] = ""
DF.loc[idx_tr, "split"] = "train"; DF.loc[idx_va, "split"] = "val"; DF.loc[idx_te, "split"] = "test"
DF["gidx"] = np.arange(len(DF))           # row index into the semantic-feature cache
DF[["rel", "cls", "label", "split"]].to_csv(os.path.join(OUT, "splits/patternnet_split.csv"), index=False)
split_counts = DF.groupby(["cls", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
split_counts.to_csv(os.path.join(OUT, "tables/T_dataset_split_counts.csv"))
print(DF.split.value_counts().to_dict())
print("Images per class: min", DF.cls.value_counts().min(), "max", DF.cls.value_counts().max())

def pretty(c):
    return c.replace("_", " ").replace("-", " ").title()
PRETTY = [pretty(c) for c in CLASSES]

# %% [markdown]
# ## 2. Preprocessing: CLAHE on luminance → 256×256 (native) → ImageNet normalisation

# %%
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def apply_clahe(rgb, clip, tile):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile))).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

def preprocess(rgb, clahe, clip, tile, size):
    """Eq. (1)-(5): CLAHE (optional) -> bilinear resize -> [0,1] scaling -> ImageNet normalisation."""
    if clahe:
        rgb = apply_clahe(rgb, clip, tile)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())

def denorm(t):
    x = t.detach().cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(x, 0, 1)

def load_uint8(path, clahe, clip, tile, size):
    """CPU part of the pipeline: CLAHE (optional) + bilinear resize; returns uint8 CHW tensor."""
    rgb = read_rgb(path)
    if clahe:
        rgb = apply_clahe(rgb, clip, tile)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))

_MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1); _STD_T = torch.tensor(STD).view(1, 3, 1, 1)
def to_input(x):
    """GPU part: [0,1] scaling + ImageNet normalisation (identical to preprocess())."""
    x = x.to(DEVICE, non_blocking=True).float().div_(255.0)
    x = (x - _MEAN_T.to(DEVICE)) / _STD_T.to(DEVICE)
    return x.contiguous(memory_format=torch.channels_last)

class SceneDataset(Dataset):
    """Always returns (uint8 image, semantic_features_or_empty, label, global_index)."""
    def __init__(self, df, clahe, clip, tile, size, cache_path=None):
        self.paths = df.path.values; self.labels = df.label.values; self.gidx = df.gidx.values
        self.clahe, self.clip, self.tile, self.size = clahe, clip, tile, size
        self.cache_path = cache_path; self._mm = None
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        x = load_uint8(self.paths[i], self.clahe, self.clip, self.tile, self.size)
        if self.cache_path is not None:
            if self._mm is None:
                self._mm = np.load(self.cache_path, mmap_mode="r")
            s = torch.from_numpy(np.array(self._mm[self.gidx[i]], dtype=np.float32))
        else:
            s = torch.zeros(0)
        return x, s, int(self.labels[i]), int(self.gidx[i])

# %% [markdown]
# ## 3. Model
# * **Visual stream** `F_img ∈ R^{C_v×H_v×W_v}`: ImageNet-pretrained CNN (MobileNetV2 by default; ResNet-50, DenseNet-121, EfficientNet-B0 for Table 7), stride 32 → 8×8 for 256×256 input.
# * **Semantic stream** `F_seg ∈ R^{C_s×H_s×W_s}`: frozen SegFormer-B2 (ADE20K); hierarchical encoder + all-MLP decoder, decoder representation `F_dec` (C_s = 768, H/4×W/4) average-pooled to the visual grid. Logits Z are used only for visualising masks.
# * **Fusion**: none / concatenation / element-wise addition / SG-CAM (single-head cross-attention, visual queries, semantic keys & values, output projection W_O, residual + LayerNorm).
# * **Classifier**: global average pooling → Dropout(0.5) → FC(512) → ReLU → FC(K) → softmax.

# %%
class SemanticEncoder(nn.Module):
    """Frozen SegFormer: returns decoder representation F_dec (B,C_s,H/4,W/4) and ADE20K logits Z."""
    def __init__(self, name):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        if SMOKE:
            cfg = SegformerConfig(depths=[1, 1, 1, 1], hidden_sizes=[16, 32, 64, 128],
                                  num_attention_heads=[1, 1, 2, 4], decoder_hidden_size=32, num_labels=150)
            self.seg = SegformerForSemanticSegmentation(cfg)
        else:
            self.seg = SegformerForSemanticSegmentation.from_pretrained(name)
        self.seg.eval()
        for p in self.seg.parameters():
            p.requires_grad_(False)
        self.out_dim = self.seg.config.decoder_hidden_size
        self.id2label = self.seg.config.id2label
    def train(self, mode=True):           # always stays in eval mode (frozen BN / dropout)
        return super().train(False)
    @torch.no_grad()
    def forward(self, x):
        enc = self.seg.segformer(x, output_hidden_states=True, return_dict=True).hidden_states
        head = self.seg.decode_head
        projs = getattr(head, "linear_c", None) or head.linear_projections
        B = x.shape[0]; size0 = enc[0].shape[2:]; outs = []
        for h, mlp in zip(enc, projs):
            if h.ndim == 3:
                s = int(math.sqrt(h.shape[1])); h = h.reshape(B, s, s, -1).permute(0, 3, 1, 2)
            hh, ww = h.shape[2], h.shape[3]
            h = mlp(h).transpose(1, 2).reshape(B, -1, hh, ww)
            outs.append(F.interpolate(h, size=size0, mode="bilinear", align_corners=False))
        f = head.activation(head.batch_norm(head.linear_fuse(torch.cat(outs[::-1], dim=1))))
        return f, head.classifier(f)

def build_backbone(name, pretrained):
    tvm = torchvision.models
    if name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        return m.features, 1280
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        return nn.Sequential(*list(m.children())[:-2]), 2048
    if name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        return nn.Sequential(m.features, nn.ReLU(inplace=True)), 1024
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        return m.features, 1280
    raise ValueError(name)

class SGCAM(nn.Module):
    """Eq. (12)-(16). Q from visual tokens, K/V from semantic tokens; F_cam = A V; F_out = LN(F_img + W_O F_cam)."""
    def __init__(self, c_v, c_s, d):
        super().__init__()
        self.d = d
        self.W_Q = nn.Linear(c_v, d); self.W_K = nn.Linear(c_s, d); self.W_V = nn.Linear(c_s, d)
        self.W_O = nn.Linear(d, c_v)
        self.norm = nn.LayerNorm(c_v)
    def forward(self, f_img, f_seg):
        B, Cv, Hv, Wv = f_img.shape
        t_v = f_img.flatten(2).transpose(1, 2)          # (B, N_v, C_v)
        t_s = f_seg.flatten(2).transpose(1, 2)          # (B, N_s, C_s)
        Q, K, V = self.W_Q(t_v), self.W_K(t_s), self.W_V(t_s)
        A = torch.softmax(Q @ K.transpose(1, 2) / math.sqrt(self.d), dim=-1)   # (B, N_v, N_s)
        f_cam = A @ V                                   # (B, N_v, d)
        out = self.norm(t_v + self.W_O(f_cam))          # residual in C_v space
        return out.transpose(1, 2).reshape(B, Cv, Hv, Wv), A

class SceneModel(nn.Module):
    def __init__(self, backbone, fusion, n_classes, c_s, d, dropout, pretrained):
        super().__init__()
        self.backbone, c_v = build_backbone(backbone, pretrained)
        self.fusion = fusion
        if fusion == "sgcam":
            self.fuse = SGCAM(c_v, c_s, d); c_out = c_v
        elif fusion == "concat":
            self.fuse = nn.Sequential(nn.Conv2d(c_s, d, 1), nn.BatchNorm2d(d), nn.ReLU(inplace=True)); c_out = c_v + d
        elif fusion == "add":
            self.fuse = nn.Sequential(nn.Conv2d(c_s, c_v, 1), nn.BatchNorm2d(c_v)); c_out = c_v
        elif fusion == "none":
            self.fuse = None; c_out = c_v
        else:
            raise ValueError(fusion)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_out, 512), nn.ReLU(inplace=True),
                                  nn.Linear(512, n_classes))
        self.feat_dim = c_out
    def forward(self, x, f_seg=None):
        f_img = self.backbone(x); attn = None
        if self.fusion != "none":
            f_seg = F.adaptive_avg_pool2d(f_seg.float(), f_img.shape[2:]).to(f_img.dtype)
        if self.fusion == "sgcam":
            f, attn = self.fuse(f_img, f_seg)
        elif self.fusion == "concat":
            f = torch.cat([f_img, self.fuse(f_seg)], dim=1)
        elif self.fusion == "add":
            f = f_img + self.fuse(f_seg)
        else:
            f = f_img
        z = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return dict(logits=self.head(z), feat=z, attn=attn)

# %% [markdown]
# ## 4. Frozen-SegFormer feature cache
# SegFormer is frozen, so its decoder features are computed once per preprocessing setting and stored (float16) instead of being recomputed every epoch. This is numerically equivalent to running it online and makes multi-seed experiments affordable. If disk space is insufficient the notebook falls back to online computation.

# %%
_SEM = {}
def get_sem_encoder():
    if "enc" not in _SEM:
        _SEM["enc"] = SemanticEncoder(CFG["SEGFORMER"]).to(DEVICE).eval()
    return _SEM["enc"]

def sem_dim():
    return get_sem_encoder().out_dim

def cache_key(clahe, clip, tile):
    tag = f"cl{clip:g}t{tile}" if clahe else "nocl"
    return f"{CFG['SEGFORMER'].split('/')[-1]}_{CFG['IMG_SIZE']}_{tag}"

def ensure_cache(clahe, clip, tile):
    key = cache_key(clahe, clip, tile)
    path = os.path.join(CFG["CACHE_DIR"], key + ".npy"); done = path + ".done"
    if os.path.exists(done):
        return path
    enc = get_sem_encoder()
    g = CFG["IMG_SIZE"] // 32
    need = len(DF) * enc.out_dim * g * g * 2 * 1.05
    free = shutil.disk_usage(CFG["CACHE_DIR"]).free
    if free < need:   # evict other caches (they can be rebuilt)
        for f in glob.glob(os.path.join(CFG["CACHE_DIR"], "*.npy*")):
            os.remove(f)
        free = shutil.disk_usage(CFG["CACHE_DIR"]).free
    if free < need:
        print(f"[cache] not enough disk ({free/1e9:.1f} GB < {need/1e9:.1f} GB): online mode")
        return None
    print(f"[cache] building {key} ({need/1e9:.1f} GB) ...")
    t0 = time.time()
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(DF), enc.out_dim, g, g))
    ds = SceneDataset(DF, clahe, clip, tile, CFG["IMG_SIZE"])
    dl = DataLoader(ds, batch_size=CFG["CACHE_BATCH"], shuffle=False, num_workers=CFG["NUM_WORKERS"], pin_memory=True)
    with torch.no_grad():
        for x, _, _, gi in dl:
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=CFG["AMP"] and DEVICE.type == "cuda"):
                f, _ = enc(to_input(x).contiguous())
            mm[gi.numpy()] = F.adaptive_avg_pool2d(f.float(), (g, g)).cpu().numpy().astype(np.float16)
    mm.flush(); del mm; torch.cuda.empty_cache()
    open(done, "w").write("ok")
    print(f"[cache] done in {(time.time()-t0)/60:.1f} min")
    return path

# %% [markdown]
# ## 5. Training / evaluation engine

# %%
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def exp_key(e):
    cl = f"cl{e['clip']:g}t{e['tile']}" if e["clahe"] else "nocl"
    return f"{e['backbone']}_{e['fusion']}_{cl}_i{CFG['IMG_SIZE']}_d{e['d']}_lr{e['lr']:g}_bs{e['bs']}_ep{e['epochs']}_s{e['seed']}"

def make_exp(backbone="mobilenet_v2", fusion="sgcam", clahe=True, clip=None, tile=None, d=None,
             lr=None, bs=None, epochs=None, seed=0):
    e = dict(backbone=backbone, fusion=fusion, clahe=clahe,
             clip=CFG["CLAHE_CLIP"] if clip is None else clip, tile=CFG["CLAHE_TILE"] if tile is None else tile,
             d=CFG["D_ATTN"] if d is None else d, lr=CFG["LR"] if lr is None else lr,
             bs=CFG["BATCH"] if bs is None else bs, epochs=CFG["EPOCHS"] if epochs is None else epochs, seed=seed)
    e["key"] = exp_key(e)
    return e

def compute_metrics(y, prob):
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    labels = np.arange(prob.shape[1])
    m = dict(accuracy=accuracy_score(y, pred) * 100, macro_precision=p * 100, macro_recall=r * 100,
             macro_f1=f * 100, weighted_f1=f1_score(y, pred, average="weighted") * 100,
             kappa=cohen_kappa_score(y, pred) * 100)
    try:
        m["macro_auc_ovr"] = roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        m["macro_auc_ovr"] = float("nan")
    if prob.shape[1] > 5:
        m["top5_accuracy"] = top_k_accuracy_score(y, prob, k=5, labels=labels) * 100
    return m

def run_epoch(model, dl, sem_online, train, opt=None, scaler=None, collect=False):
    model.train(train)
    tot, correct, n = 0.0, 0, 0
    P, Y, G, Fz = [], [], [], []
    crit = nn.CrossEntropyLoss()
    use_amp = CFG["AMP"] and DEVICE.type == "cuda"
    for x, s, y, gi in dl:
        x = to_input(x); y = y.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                if model.fusion == "none":
                    f_seg = None
                elif s.numel() > 0:
                    f_seg = s.to(DEVICE, non_blocking=True)
                else:
                    f_seg = sem_online(x.contiguous())[0]
                out = model(x, f_seg)
                loss = crit(out["logits"].float(), y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
        correct += (out["logits"].argmax(1) == y).sum().item()
        if collect:
            P.append(torch.softmax(out["logits"].float(), 1).detach().cpu()); Y.append(y.cpu()); G.append(gi)
            Fz.append(out["feat"].float().detach().cpu())
    res = dict(loss=tot / n, acc=100 * correct / n)
    if collect:
        res.update(prob=torch.cat(P).numpy(), y=torch.cat(Y).numpy(), gidx=torch.cat(G).numpy(),
                   feat=torch.cat(Fz).numpy())
    return res

def result_path(key):
    return os.path.join(OUT, "results/json", key + ".json")

RUN_DURATIONS = []
def run_experiment(e, eval_test=True):
    key = e["key"]
    if os.path.exists(result_path(key)):
        return json.load(open(result_path(key)))
    elapsed_h = (time.time() - SESSION_T0) / 3600
    heavy = e["backbone"] in ("resnet50", "densenet121")
    est_h = max(max(RUN_DURATIONS) / 3600 if RUN_DURATIONS else 0.5, 0.75 if heavy else 0.5) * 1.2
    if elapsed_h + est_h > CFG["MAX_SESSION_HOURS"]:
        print(f"[skip] {key}: session time budget reached; run it in the next session")
        return None
    print(f"\n=== {key} ===")
    t0 = time.time(); set_seed(e["seed"])
    needs_sem = e["fusion"] != "none"
    cache = ensure_cache(e["clahe"], e["clip"], e["tile"]) if needs_sem else None
    c_s = sem_dim() if needs_sem else 1
    mk = lambda df, shuffle: DataLoader(
        SceneDataset(df, e["clahe"], e["clip"], e["tile"], CFG["IMG_SIZE"], cache), batch_size=e["bs"],
        shuffle=shuffle, num_workers=CFG["NUM_WORKERS"], pin_memory=True, drop_last=False,
        persistent_workers=CFG["NUM_WORKERS"] > 0, generator=torch.Generator().manual_seed(e["seed"]),
        worker_init_fn=lambda w: np.random.seed(e["seed"] * 100 + w))
    dl_tr, dl_va = mk(DF[DF.split == "train"], True), mk(DF[DF.split == "val"], False)
    sem_online = (lambda x: get_sem_encoder()(x)) if needs_sem and cache is None else None
    model = SceneModel(e["backbone"], e["fusion"], NUM_CLASSES, c_s, e["d"], CFG["DROPOUT"], CFG["PRETRAINED"]).to(DEVICE).to(memory_format=torch.channels_last)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=e["lr"], weight_decay=CFG["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=CFG["STEP_SIZE"], gamma=CFG["GAMMA"])
    scaler = make_scaler(CFG["AMP"] and DEVICE.type == "cuda")
    hist, best_acc, best_state, best_ep = [], -1, None, -1
    for ep in range(1, e["epochs"] + 1):
        tr = run_epoch(model, dl_tr, sem_online, True, opt, scaler)
        va = run_epoch(model, dl_va, sem_online, False)
        sched.step()
        hist.append(dict(epoch=ep, train_loss=tr["loss"], train_acc=tr["acc"], val_loss=va["loss"], val_acc=va["acc"],
                         lr=opt.param_groups[0]["lr"]))
        if va["acc"] > best_acc:
            best_acc, best_ep = va["acc"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep:3d} | train {tr['loss']:.4f}/{tr['acc']:.2f}% | val {va['loss']:.4f}/{va['acc']:.2f}%")
    model.load_state_dict(best_state)
    val_full = run_epoch(model, dl_va, sem_online, False, collect=True)
    res = dict(exp=e, best_epoch=best_ep, history=hist, val_metrics=compute_metrics(val_full["y"], val_full["prob"]),
               n_trainable_params=sum(p.numel() for p in params), train_minutes=None, timestamp=time.ctime(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    if eval_test:
        te = run_epoch(model, mk(DF[DF.split == "test"], False), sem_online, False, collect=True)
        res["test_metrics"] = compute_metrics(te["y"], te["prob"])
        pdf = pd.DataFrame(te["prob"], columns=[f"p_{i}" for i in range(NUM_CLASSES)])
        pdf.insert(0, "y_pred", te["prob"].argmax(1)); pdf.insert(0, "y_true", te["y"]); pdf.insert(0, "gidx", te["gidx"])
        pdf.insert(1, "rel", DF.rel.values[te["gidx"]])
        pdf.to_csv(os.path.join(OUT, "results/preds", key + ".csv.gz"), index=False, float_format="%.6f")
        np.save(os.path.join(OUT, "results/feats", key + ".npy"), te["feat"].astype(np.float16))
    if CFG["SAVE_CHECKPOINTS"]:
        torch.save(best_state, os.path.join(OUT, "checkpoints", key + ".pt"))
    res["train_minutes"] = (time.time() - t0) / 60
    json.dump(res, open(result_path(key), "w"), indent=1)
    RUN_DURATIONS.append(time.time() - t0)
    msg = f"  -> best epoch {best_ep}, val acc {res['val_metrics']['accuracy']:.2f}%"
    if eval_test:
        msg += f", TEST acc {res['test_metrics']['accuracy']:.2f}%, macro-F1 {res['test_metrics']['macro_f1']:.2f}%"
    print(msg + f" ({res['train_minutes']:.1f} min)")
    del model, opt; torch.cuda.empty_cache()
    return res

# %% [markdown]
# ## 6. Experiment registry
# * **main** — full model (CLAHE + SegFormer + SG-CAM, MobileNetV2), all seeds → Table 4, Figs. 6/8/10.
# * **ablation** — Table 8 with explicit text labels (Reviewer 3, comment 5), including concatenation and addition fusion baselines.
# * **sensitivity** — Table 9, one-factor-at-a-time around the default, selected on the **validation** split.
# * **backbones** — Table 7, every backbone **with and without** SG-CAM.

# %%
SEEDS = CFG["SEEDS"]
ABLATION = OrderedDict([
    ("MobileNetV2 (baseline, no CLAHE)",            dict(fusion="none",   clahe=False, cl="No",  sem="No",  fus="None")),
    ("MobileNetV2 + CLAHE",                         dict(fusion="none",   clahe=True,  cl="Yes", sem="No",  fus="None")),
    ("MobileNetV2 + CLAHE + SegFormer (concatenation)", dict(fusion="concat", clahe=True, cl="Yes", sem="Yes", fus="Concatenation")),
    ("MobileNetV2 + CLAHE + SegFormer (addition)",  dict(fusion="add",    clahe=True,  cl="Yes", sem="Yes", fus="Element-wise addition")),
    ("MobileNetV2 + SegFormer + SG-CAM (no CLAHE)", dict(fusion="sgcam",  clahe=False, cl="No",  sem="Yes", fus="SG-CAM")),
    ("Full model: CLAHE + SegFormer + SG-CAM",      dict(fusion="sgcam",  clahe=True,  cl="Yes", sem="Yes", fus="SG-CAM")),
])
BACKBONES = OrderedDict([("MobileNetV2", "mobilenet_v2"), ("ResNet-50", "resnet50"),
                         ("DenseNet-121", "densenet121"), ("EfficientNet-B0", "efficientnet_b0")])
SENS = OrderedDict([
    ("CLAHE clip limit", ("clip", [1.5, 2.0, 3.0])),
    ("CLAHE tile grid size", ("tile", [6, 8, 10])),
    ("Embedding dimension d", ("d", [128, 256, 512])),
    ("Learning rate", ("lr", [1e-5, 1e-4, 1e-3])),
    ("Batch size", ("bs", [8, 16, 32])),
])

def main_exps():
    return [make_exp(seed=s) for s in SEEDS]
def ablation_exps():
    return {lab: [make_exp(fusion=v["fusion"], clahe=v["clahe"], seed=s) for s in SEEDS] for lab, v in ABLATION.items()}
def backbone_exps():
    return {(lab, fu): [make_exp(backbone=bb, fusion=fu, seed=s) for s in CFG["BACKBONE_SEEDS"]]
            for lab, bb in BACKBONES.items() for fu in ["none", "sgcam"]}
def sens_exps():
    out = OrderedDict()
    for lab, (field, vals) in SENS.items():
        for v in vals:
            out[(lab, v)] = make_exp(epochs=CFG["SWEEP_EPOCHS"], seed=SEEDS[0], **{field: v})
    return out

# %% [markdown]
# ## 7. Import results from previous sessions (Notebook Output added as input)

# %%
def import_previous_results():
    n = 0
    for base in glob.glob("/kaggle/input/*"):
        for dirpath, dirnames, _ in os.walk(base):
            if os.path.basename(dirpath) == "sgcam_outputs" and os.path.abspath(dirpath) != os.path.abspath(OUT):
                for sub in ["results/json", "results/preds", "results/feats", "checkpoints"]:
                    for f in glob.glob(os.path.join(dirpath, sub, "*")):
                        dst = os.path.join(OUT, sub, os.path.basename(f))
                        if not os.path.exists(dst):
                            shutil.copy(f, dst); n += 1
                dirnames[:] = []
            if dirpath.count(os.sep) - base.count(os.sep) > 4:
                dirnames[:] = []
    print(f"Imported {n} files from previous sessions")
if not SMOKE:
    import_previous_results()

# %% [markdown]
# ## 8. Run experiments

# %%
def flat(x):
    return [e for v in x.values() for e in v] if isinstance(x, dict) else list(x)

ORDER = []
if "main" in CFG["RUN_GROUPS"]:
    ORDER += main_exps()
if "transfer_baselines" in CFG["RUN_GROUPS"]:
    ORDER += ablation_exps()["MobileNetV2 + CLAHE"]
if "robustness_extra" in CFG["RUN_GROUPS"]:
    # concatenation-fusion checkpoints (3 seeds) + two additional seeds for B and C (robustness analysis)
    ORDER += [make_exp(fusion="concat", seed=s) for s in [0, 1, 2]]
    ORDER += [make_exp(fusion="none", clahe=True, seed=s) for s in [3, 4]] + [make_exp(seed=s) for s in [3, 4]]
if "ablation" in CFG["RUN_GROUPS"]:
    ORDER += flat(ablation_exps())
if "sensitivity" in CFG["RUN_GROUPS"]:
    ORDER += list(sens_exps().values())
if "backbones" in CFG["RUN_GROUPS"]:
    ORDER += flat(backbone_exps())
seen = set(); ORDER = [e for e in ORDER if not (e["key"] in seen or seen.add(e["key"]))]
todo = [e for e in ORDER if not os.path.exists(result_path(e["key"]))]
print(f"{len(ORDER)} runs requested, {len(todo)} still to run")
for e in ORDER:
    run_experiment(e)

# %% [markdown]
# ## 9. Tables (computed only from saved results)

# %%
def load_res(key):
    p = result_path(key)
    return json.load(open(p)) if os.path.exists(p) else None

def load_preds(key):
    p = os.path.join(OUT, "results/preds", key + ".csv.gz")
    return pd.read_csv(p) if os.path.exists(p) else None

def ms(vals, nd=2):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals: return "–"
    if len(vals) == 1: return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} ± {np.std(vals, ddof=1):.{nd}f}"

METRICS = [("accuracy", "Accuracy (%)"), ("macro_precision", "Macro precision (%)"), ("macro_recall", "Macro recall (%)"),
           ("macro_f1", "Macro F1 (%)"), ("weighted_f1", "Weighted F1 (%)"), ("kappa", "Cohen's κ (×100)"),
           ("top5_accuracy", "Top-5 accuracy (%)"), ("macro_auc_ovr", "Macro AUC (OvR)")]

def summarize(exps, split="test_metrics"):
    rs = [load_res(e["key"]) for e in exps]; rs = [r for r in rs if r and split in r]
    row = {lab: ms([r[split].get(k) for r in rs], 4 if k == "macro_auc_ovr" else 2) for k, lab in METRICS}
    row["Seeds (n)"] = len(rs)
    return row, rs

TABLES = {}
# ---- Table 3: actual training configuration
T3 = pd.DataFrame([("Optimizer", CFG["OPTIMIZER"]), ("Initial learning rate", f"{CFG['LR']:g}"), ("Batch size", CFG["BATCH"]),
                   ("Learning-rate schedule", f"Step decay ×{CFG['GAMMA']} every {CFG['STEP_SIZE']} epochs"),
                   ("Epochs", f"{CFG['EPOCHS']} (best epoch selected on validation accuracy)"),
                   ("Loss function", "Cross-entropy (scene labels only)"), ("Input size", f"{CFG['IMG_SIZE']}×{CFG['IMG_SIZE']}"),
                   ("CLAHE", f"clip limit {CFG['CLAHE_CLIP']}, tile grid {CFG['CLAHE_TILE']}×{CFG['CLAHE_TILE']} (L channel, LAB)"),
                   ("Data split", "Stratified 60/20/20 train/validation/test, split seed %d" % CFG["SPLIT_SEED"]),
                   ("Random seeds", ", ".join(map(str, SEEDS))), ("Mixed precision", "FP16 autocast" if CFG["AMP"] else "FP32"),
                   ("Semantic encoder", f"{CFG['SEGFORMER']} (frozen)"), ("SG-CAM dimension d", CFG["D_ATTN"])],
                  columns=["Parameter", "Value"])
TABLES["T3_training_parameters"] = T3

# ---- Table 4: main results
row, MAIN_RES = summarize(main_exps())
TABLES["T4_main_results"] = pd.DataFrame([dict(Dataset="PatternNet (test)", Model="SG-CAM (MobileNetV2)", **row)])

# ---- Representative seed (median test accuracy) for per-class tables and figures
REP_KEY = None
if MAIN_RES:
    accs = sorted([(r["test_metrics"]["accuracy"], r["exp"]["key"]) for r in MAIN_RES])
    REP_KEY = accs[len(accs) // 2][1]

# ---- per-class metrics (mean ± std over seeds)
if MAIN_RES:
    per = []
    for r in MAIN_RES:
        pr = load_preds(r["exp"]["key"])
        p, rc, f, sup = precision_recall_fscore_support(pr.y_true, pr.y_pred, labels=range(NUM_CLASSES), zero_division=0)
        per.append(np.stack([p, rc, f]) * 100)
    per = np.stack(per); sup = np.bincount(pr.y_true, minlength=NUM_CLASSES)
    TABLES["T_per_class_metrics"] = pd.DataFrame({
        "Class": PRETTY, "Test images": sup,
        "Precision (%)": [ms(per[:, 0, i]) for i in range(NUM_CLASSES)],
        "Recall (%)": [ms(per[:, 1, i]) for i in range(NUM_CLASSES)],
        "F1 (%)": [ms(per[:, 2, i]) for i in range(NUM_CLASSES)]})

# ---- Table 8: ablation with text labels + McNemar test vs full model (seed-matched)
def mcnemar_p(k1, k2):
    a, b = load_preds(k1), load_preds(k2)
    if a is None or b is None: return None
    a = a.set_index("gidx").sort_index(); b = b.set_index("gidx").sort_index()
    c1 = (a.y_pred == a.y_true).values; c2 = (b.y_pred == b.y_true).values
    n01 = int((c1 & ~c2).sum()); n10 = int((~c1 & c2).sum())
    if n01 + n10 == 0: return 1.0
    return stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue * 1.0

abl = ablation_exps(); full_label = list(ABLATION)[-1]; rows = []
for lab, exps in abl.items():
    row, rs = summarize(exps); v = ABLATION[lab]
    ps = [mcnemar_p(e["key"], f["key"]) for e, f in zip(exps, abl[full_label])] if lab != full_label else []
    ps = [p for p in ps if p is not None]
    rows.append({"Configuration": lab, "CLAHE": v["cl"], "Semantic stream (SegFormer-B2)": v["sem"], "Fusion": v["fus"],
                 "Accuracy (%)": row["Accuracy (%)"], "Macro F1 (%)": row["Macro F1 (%)"],
                 "Macro AUC (OvR)": row["Macro AUC (OvR)"], "Seeds (n)": row["Seeds (n)"],
                 "McNemar p vs. full (min over seeds)": (f"{min(ps):.2e}" if ps else "–"),
                 "McNemar p per seed": (", ".join(f"{q:.2e}" for q in ps) if ps else "–")})
TABLES["T8_ablation"] = pd.DataFrame(rows)

# ---- Table 7: backbones with / without SG-CAM
rows = []
for lab in BACKBONES:
    exps_n = backbone_exps()[(lab, "none")]; exps_s = backbone_exps()[(lab, "sgcam")]
    rn, rsn = summarize(exps_n); rsg, rss = summarize(exps_s)
    gain = (np.mean([r["test_metrics"]["accuracy"] for r in rss]) - np.mean([r["test_metrics"]["accuracy"] for r in rsn])
            if rsn and rss else None)
    rows.append({"Backbone": lab, "Without SG-CAM: accuracy (%)": rn["Accuracy (%)"],
                 "With SG-CAM: accuracy (%)": rsg["Accuracy (%)"], "Without SG-CAM: macro F1 (%)": rn["Macro F1 (%)"],
                 "With SG-CAM: macro F1 (%)": rsg["Macro F1 (%)"], "Δ accuracy (pp)": f"{gain:+.2f}" if gain is not None else "–",
                 "Trainable params with SG-CAM (M)": f"{rss[0]['n_trainable_params']/1e6:.2f}" if rss else "–",
                 "Seeds (n)": f"{len(rsn)}/{len(rss)}"})
TABLES["T7_backbones"] = pd.DataFrame(rows)

# ---- Table 9: sensitivity (validation split)
rows = []
for (lab, v), e in sens_exps().items():
    r = load_res(e["key"]); default = {"clip": CFG["CLAHE_CLIP"], "tile": CFG["CLAHE_TILE"], "d": CFG["D_ATTN"],
                                       "lr": CFG["LR"], "bs": CFG["BATCH"]}[SENS[lab][0]]
    vs = (f"{v}×{v}" if SENS[lab][0] == "tile" else f"{v:g}") + (" (default)" if v == default else "")
    rows.append({"Parameter": lab, "Value": vs,
                 "Validation accuracy (%)": f"{r['val_metrics']['accuracy']:.2f}" if r else "–",
                 "Validation macro F1 (%)": f"{r['val_metrics']['macro_f1']:.2f}" if r else "–",
                 "Epochs": CFG["SWEEP_EPOCHS"], "Seed": SEEDS[0]})
TABLES["T9_sensitivity"] = pd.DataFrame(rows)

for name, t in TABLES.items():
    t.to_csv(os.path.join(OUT, "tables", name + ".csv"), index=False)
    print(f"\n### {name}"); print(t.to_string(index=False))

# %% [markdown]
# ## 10. Figures

# %%
FIG = os.path.join(OUT, "figures")
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
def savefig(fig, name):
    print(f"[fig] {name}  ({(time.time()-SESSION_T0)/3600:.2f} h)")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight"); plt.close(fig)

# ---- Fig. 2: preprocessing stages
sample = DF[DF.split == "test"].iloc[0]
rgb = read_rgb(sample.path); cl = apply_clahe(rgb, CFG["CLAHE_CLIP"], CFG["CLAHE_TILE"])
rs = cv2.resize(cl, (CFG["IMG_SIZE"],) * 2, interpolation=cv2.INTER_LINEAR)
nm = preprocess(rgb, True, CFG["CLAHE_CLIP"], CFG["CLAHE_TILE"], CFG["IMG_SIZE"]).numpy().transpose(1, 2, 0)
fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
L0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]; L1 = cv2.cvtColor(cl, cv2.COLOR_RGB2LAB)[..., 0]
for a, im, t in zip(ax[:4], [rgb, cl, rs, (nm - nm.min()) / (nm.max() - nm.min())],
                    [f"(a) Original ({rgb.shape[1]}×{rgb.shape[0]})", "(b) CLAHE on luminance",
                     f"(c) Resized ({CFG['IMG_SIZE']}×{CFG['IMG_SIZE']})", "(d) ImageNet-normalised (rescaled for display)"]):
    a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
ax[4].hist(L0.ravel(), 64, alpha=.6, label="Original L"); ax[4].hist(L1.ravel(), 64, alpha=.6, label="CLAHE L")
ax[4].set_title("(e) Luminance histogram", fontsize=9); ax[4].legend(fontsize=8); ax[4].set_xlabel("L value")
savefig(fig, "F2_preprocessing")

# ---- dataset overview: one image per class
nc = 8; nr = math.ceil(NUM_CLASSES / nc)
fig, ax = plt.subplots(nr, nc, figsize=(nc * 1.9, nr * 2.1))
for i, a in enumerate(np.array(ax).ravel()):
    a.axis("off")
    if i < NUM_CLASSES:
        a.imshow(read_rgb(DF[DF.label == i].iloc[0].path)); a.set_title(PRETTY[i], fontsize=7)
savefig(fig, "F_dataset_samples")

if REP_KEY:
    pr = load_preds(REP_KEY); y, yp = pr.y_true.values, pr.y_pred.values
    prob = pr[[f"p_{i}" for i in range(NUM_CLASSES)]].values
    seed_lbl = load_res(REP_KEY)["exp"]["seed"]

    # ---- Fig. 6: confusion matrix (counts and row-normalised)
    cm = confusion_matrix(y, yp, labels=range(NUM_CLASSES))
    for norm in [False, True]:
        M = cm / cm.sum(1, keepdims=True) * 100 if norm else cm
        s = max(8, NUM_CLASSES * 0.42)
        fig, a = plt.subplots(figsize=(s + 2, s))
        im = a.imshow(M, cmap="Blues"); fig.colorbar(im, ax=a, fraction=0.046, pad=0.02,
                                                    label="Row-normalised (%)" if norm else "Number of test images")
        a.set_xticks(range(NUM_CLASSES)); a.set_yticks(range(NUM_CLASSES))
        a.set_xticklabels(PRETTY, rotation=90, fontsize=7); a.set_yticklabels(PRETTY, fontsize=7)
        thr = M.max() / 2
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if M[i, j] > 0:
                    a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                           color="white" if M[i, j] > thr else "black")
        a.set_xlabel("Predicted class"); a.set_ylabel("True class")
        a.set_title(f"Confusion Matrix — PatternNet test set ({len(y)} images), SG-CAM, seed {seed_lbl}")
        savefig(fig, "F6_confusion_matrix_" + ("normalised" if norm else "counts"))

    # ---- Fig. 8: per-class precision / recall / F1 (representative seed, with seed std as error bars)
    p, rc, f1, _ = precision_recall_fscore_support(y, yp, labels=range(NUM_CLASSES), zero_division=0)
    fig, a = plt.subplots(figsize=(max(10, NUM_CLASSES * 0.45), 4.5)); xx = np.arange(NUM_CLASSES); w = 0.27
    err = per.std(0, ddof=1) if per.shape[0] > 1 else np.zeros((3, NUM_CLASSES))
    for k, (vals, lab) in enumerate([(per[:, 0].mean(0), "Precision"), (per[:, 1].mean(0), "Recall"), (per[:, 2].mean(0), "F1-score")]):
        a.bar(xx + (k - 1) * w, vals, w, yerr=err[k], capsize=1.5, label=lab, error_kw=dict(lw=0.6))
    a.set_xticks(xx); a.set_xticklabels(PRETTY, rotation=90, fontsize=7); a.set_ylabel("Score (%)")
    a.set_ylim(max(0, per.min() - 5), 100.5); a.legend(ncol=3, loc="lower left")
    a.set_title(f"Per-class precision, recall and F1-score on PatternNet (mean ± std over {per.shape[0]} seeds)")
    savefig(fig, "F8_per_class_metrics")

    # ---- Fig. 10: ROC curves (one-vs-rest; computed from softmax scores)
    Yb = label_binarize(y, classes=range(NUM_CLASSES))
    if NUM_CLASSES == 2: Yb = np.hstack([1 - Yb, Yb])
    fpr, tpr, aucs = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(Yb[:, i], prob[:, i]); aucs[i] = auc(fpr[i], tpr[i])
    grid = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.mean([np.interp(grid, fpr[i], tpr[i]) for i in range(NUM_CLASSES)], 0)
    fpr_mi, tpr_mi, _ = roc_curve(Yb.ravel(), prob.ravel())
    worst = sorted(aucs, key=aucs.get)[:5]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for a, logx in zip(axs, [False, True]):
        a.plot(grid, mean_tpr, lw=2.2, label=f"Macro-average (AUC = {auc(grid, mean_tpr):.4f})")
        a.plot(fpr_mi, tpr_mi, lw=1.8, ls="--", label=f"Micro-average (AUC = {auc(fpr_mi, tpr_mi):.4f})")
        for i in worst:
            a.plot(fpr[i], tpr[i], lw=0.9, alpha=0.8, label=f"{PRETTY[i]} (AUC = {aucs[i]:.4f})")
        a.plot([1e-5 if logx else 0, 1], [1e-5 if logx else 0, 1], "k:", lw=0.8)
        a.set_xlabel("False positive rate" + (" (log scale)" if logx else "")); a.set_ylabel("True positive rate")
        if logx: a.set_xscale("log"); a.set_xlim(1e-4, 1)
        a.set_ylim(0, 1.01); a.legend(fontsize=7.5, loc="lower right")
    axs[0].set_title("ROC curves (one-vs-rest), PatternNet test set"); axs[1].set_title("Same curves, logarithmic FPR axis")
    savefig(fig, "F10_roc")
    pd.DataFrame({"Class": PRETTY, "AUC": [aucs[i] for i in range(NUM_CLASSES)]}).to_csv(
        os.path.join(OUT, "tables/T_per_class_auc.csv"), index=False)

    # ---- PR curves
    fig, a = plt.subplots(figsize=(6.5, 5))
    pm, rm, _ = precision_recall_curve(Yb.ravel(), prob.ravel())
    ap_macro = average_precision_score(Yb, prob, average="macro")
    a.plot(rm, pm, lw=2, label=f"Micro-average (AP = {average_precision_score(Yb, prob, average='micro'):.4f})")
    aps = {i: average_precision_score(Yb[:, i], prob[:, i]) for i in range(NUM_CLASSES)}
    for i in sorted(aps, key=aps.get)[:5]:
        pc, rcc, _ = precision_recall_curve(Yb[:, i], prob[:, i]); a.plot(rcc, pc, lw=0.9, label=f"{PRETTY[i]} (AP = {aps[i]:.4f})")
    a.set_xlabel("Recall"); a.set_ylabel("Precision"); a.set_ylim(0, 1.01)
    a.set_title(f"Precision–recall curves, PatternNet test set (macro AP = {ap_macro:.4f})"); a.legend(fontsize=7.5, loc="lower left")
    savefig(fig, "F10b_pr")

    # ---- Misclassified examples (Reviewer 3, comment 4)
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = [(i, j) for i, j in zip(*np.unravel_index(np.argsort(off.ravel())[::-1], off.shape)) if off[i, j] > 0][:6]
    if pairs:
        per_pair = 3
        fig, ax = plt.subplots(len(pairs), per_pair, figsize=(per_pair * 2.6, len(pairs) * 2.8), squeeze=False)
        for r, (i, j) in enumerate(pairs):
            sel = pr[(pr.y_true == i) & (pr.y_pred == j)].copy()
            sel["conf"] = sel[f"p_{j}"]; sel = sel.sort_values("conf", ascending=False).head(per_pair)
            for c in range(per_pair):
                a = ax[r, c]; a.axis("off")
                if c < len(sel):
                    s = sel.iloc[c]; a.imshow(read_rgb(DF.path.values[int(s.gidx)]))
                    a.set_title(f"True: {PRETTY[i]}\nPred: {PRETTY[j]} ({s.conf:.2f})", fontsize=7.5)
        fig.suptitle(f"Most frequent confusions on the PatternNet test set (SG-CAM, seed {seed_lbl})", y=1.0)
        savefig(fig, "F_misclassified")
        pd.DataFrame([dict(true=PRETTY[i], predicted=PRETTY[j], count=int(off[i, j])) for i, j in pairs]).to_csv(
            os.path.join(OUT, "tables/T_top_confusions.csv"), index=False)

# ---- training curves (mean ± std over seeds)
if MAIN_RES:
    H = [pd.DataFrame(r["history"]) for r in MAIN_RES]; ep = H[0].epoch.values
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for a, (k1, k2, yl) in zip(axs, [("train_loss", "val_loss", "Cross-entropy loss"), ("train_acc", "val_acc", "Accuracy (%)")]):
        for k, lab in [(k1, "Training"), (k2, "Validation")]:
            M = np.stack([h[k].values for h in H]); m, s = M.mean(0), M.std(0)
            a.plot(ep, m, label=lab); a.fill_between(ep, m - s, m + s, alpha=0.2)
        a.set_xlabel("Epoch"); a.set_ylabel(yl); a.legend()
    fig.suptitle(f"SG-CAM training on PatternNet (mean ± std over {len(H)} seeds)")
    savefig(fig, "F_training_curves")

# ---- t-SNE of pooled features: baseline vs SG-CAM (representative seed)
from sklearn.manifold import TSNE
base_key = ablation_exps()[list(ABLATION)[0]][0]["key"]
tsne_keys = [(k, t) for k, t in [(base_key, "MobileNetV2 baseline"), (REP_KEY, "SG-CAM (full model)")]
             if k and os.path.exists(os.path.join(OUT, "results/feats", k + ".npy"))]
if tsne_keys:
    fig, axs = plt.subplots(1, len(tsne_keys), figsize=(7 * len(tsne_keys), 6.5), squeeze=False)
    for a, (k, t) in zip(axs[0], tsne_keys):
        Fz = np.load(os.path.join(OUT, "results/feats", k + ".npy")).astype(np.float32); yk = load_preds(k).y_true.values
        rng = np.random.RandomState(0); sel = rng.choice(len(Fz), min(2000, len(Fz)), replace=False); _t = time.time()
        from sklearn.decomposition import PCA
        X = np.nan_to_num(Fz[sel]); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        X = PCA(min(50, X.shape[1], len(X) - 1), random_state=0).fit_transform(X)
        try:
            Z = TSNE(2, init="pca", perplexity=min(30, (len(sel) - 1) / 3), random_state=0,
                     method="barnes_hut" if len(sel) >= 500 else "exact").fit_transform(X)
        except Exception as ex:
            print("[t-SNE skipped]", ex); a.axis("off"); continue
        sc = a.scatter(Z[:, 0], Z[:, 1], c=yk[sel], cmap="nipy_spectral", s=4); print(f"[t-SNE] {t}: {time.time()-_t:.0f}s")
        a.set_title(f"t-SNE of pooled features — {t}"); a.set_xticks([]); a.set_yticks([])
    savefig(fig, "F_tsne")

# ---- ablation bar chart
abl_rows = []
for lab, exps in ablation_exps().items():
    rs = [load_res(e["key"]) for e in exps]; v = [r["test_metrics"]["accuracy"] for r in rs if r]
    if v: abl_rows.append((lab, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0))
if abl_rows:
    fig, a = plt.subplots(figsize=(8, 4))
    a.barh([r[0] for r in abl_rows][::-1], [r[1] for r in abl_rows][::-1], xerr=[r[2] for r in abl_rows][::-1], capsize=3)
    lo = min(r[1] - r[2] for r in abl_rows); a.set_xlim(max(0, lo - 2), 100)
    a.set_xlabel("Test accuracy (%)"); a.set_title("Ablation on PatternNet (mean ± std over seeds)")
    savefig(fig, "F_ablation")

# ---- sensitivity plots
T9 = TABLES["T9_sensitivity"]
if (T9["Validation accuracy (%)"] != "–").any():
    fig, axs = plt.subplots(1, len(SENS), figsize=(3.3 * len(SENS), 3.2))
    for a, (lab, (field, vals)) in zip(axs, SENS.items()):
        ys = [load_res(sens_exps()[(lab, v)]["key"]) for v in vals]
        xs = [str(v) for v, r in zip(vals, ys) if r]; yv = [r["val_metrics"]["accuracy"] for r in ys if r]
        a.plot(xs, yv, "o-"); a.set_title(lab, fontsize=9); a.set_ylabel("Val. accuracy (%)")
    fig.suptitle(f"Hyper-parameter sensitivity (validation split, {CFG['SWEEP_EPOCHS']} epochs)")
    fig.tight_layout(); savefig(fig, "F9_sensitivity")
print("Figures written to", FIG)

# %% [markdown]
# ## 11. Semantic masks and SG-CAM attention (Fig. 4)
# Attention shown as the mean attention weight each semantic token receives from all visual queries (A averaged over queries), upsampled to the image.

# %%
if REP_KEY and os.path.exists(os.path.join(OUT, "checkpoints", REP_KEY + ".pt")):
    enc = get_sem_encoder()
    model = SceneModel("mobilenet_v2", "sgcam", NUM_CLASSES, enc.out_dim, CFG["D_ATTN"], CFG["DROPOUT"], False).to(DEVICE).eval()
    model.load_state_dict(torch.load(os.path.join(OUT, "checkpoints", REP_KEY + ".pt"), map_location=DEVICE))
    rng = np.random.RandomState(1)
    pal = rng.randint(0, 255, (max(151, len(enc.id2label) + 1), 3)).astype(np.uint8)
    test = DF[DF.split == "test"]
    picks = [test[test.label == c].iloc[0] for c in rng.choice(NUM_CLASSES, min(6, NUM_CLASSES), replace=False)]
    fig, ax = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)), squeeze=False)
    for r, s in enumerate(picks):
        x = preprocess(read_rgb(s.path), True, CFG["CLAHE_CLIP"], CFG["CLAHE_TILE"], CFG["IMG_SIZE"])[None].to(DEVICE)
        with torch.no_grad():
            f_dec, logits_seg = enc(x); out = model(x, f_dec)
        mask = logits_seg.argmax(1)[0].cpu().numpy()
        ids, cnt = np.unique(mask, return_counts=True); top = ids[np.argsort(cnt)[::-1][:3]]
        A = out["attn"][0].mean(0).float().cpu().numpy(); g = int(math.sqrt(A.size)); A = A.reshape(g, g)
        A = cv2.resize((A - A.min()) / (np.ptp(A) + 1e-8), (CFG["IMG_SIZE"],) * 2, interpolation=cv2.INTER_CUBIC)
        img = denorm(x[0]); pred = int(out["logits"].argmax(1))
        ax[r, 0].imshow(cv2.resize(read_rgb(s.path), (CFG["IMG_SIZE"],) * 2)); ax[r, 0].set_title(f"(a) Input — {PRETTY[s.label]}", fontsize=8)
        ax[r, 1].imshow(img); ax[r, 1].set_title("(b) CLAHE-enhanced input", fontsize=8)
        ax[r, 2].imshow(pal[mask]); ax[r, 2].set_title("(c) SegFormer-B2 mask (ADE20K)\n" + ", ".join(enc.id2label.get(int(t), str(t)) for t in top), fontsize=7)
        ax[r, 3].imshow(img); ax[r, 3].imshow(A, cmap="jet", alpha=0.45)
        ax[r, 3].set_title(f"(d) SG-CAM attention — pred: {PRETTY[pred]}", fontsize=8)
        for a in ax[r]: a.axis("off")
    savefig(fig, "F4_semantic_attention")
    print("Fig. 4 written")

# %% [markdown]
# ## 12. Measured computational complexity (Table 5)

# %%
from torch.utils.flop_counter import FlopCounterMode
class Pipeline(nn.Module):
    def __init__(self, scene, enc):
        super().__init__(); self.scene, self.enc = scene, enc
    def forward(self, x):
        return self.scene(x, self.enc(x)[0] if self.enc is not None and self.scene.fusion != "none" else None)["logits"]

def measure(module, bs_lat=1, bs_thr=8, iters=30):
    module.eval(); s = CFG["IMG_SIZE"]
    x1 = torch.randn(1, 3, s, s, device=DEVICE)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        module(x1)
    gflops = fc.get_total_flops() / 1e9
    def timeit(bs, n):
        x = torch.randn(bs, 3, s, s, device=DEVICE)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=CFG["AMP"] and DEVICE.type == "cuda"):
            for _ in range(5): module(x)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n): module(x)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t) / n
    lat = timeit(bs_lat, iters) * 1000; thr = bs_thr / timeit(bs_thr, max(3, iters // 3))
    return gflops, lat, thr

enc = get_sem_encoder(); cs = enc.out_dim; rows = []
for name, fusion, use_enc in [("MobileNetV2 baseline (no semantic stream)", "none", False),
                              ("MobileNetV2 + SegFormer-B2 (concatenation)", "concat", True),
                              ("MobileNetV2 + SegFormer-B2 + SG-CAM (proposed)", "sgcam", True)]:
    sc = SceneModel("mobilenet_v2", fusion, NUM_CLASSES, cs, CFG["D_ATTN"], CFG["DROPOUT"], False).to(DEVICE)
    pipe = Pipeline(sc, enc if use_enc else None).to(DEVICE)
    torch.cuda.empty_cache(); g, lat, thr = measure(pipe)
    n_tr = sum(p.numel() for p in sc.parameters()); n_fz = sum(p.numel() for p in enc.parameters()) if use_enc else 0
    rows.append({"Model": name, "Trainable params (M)": f"{n_tr/1e6:.2f}", "Frozen params (M)": f"{n_fz/1e6:.2f}",
                 "Total params (M)": f"{(n_tr+n_fz)/1e6:.2f}", "GFLOPs / image": f"{g:.2f}",
                 "Latency, batch 1 (ms)": f"{lat:.1f}", "Throughput, batch 8 (img/s)": f"{thr:.1f}"})
sg = SGCAM(1280, cs, CFG["D_ATTN"]).to(DEVICE); g_ = CFG["IMG_SIZE"] // 32
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    sg(torch.randn(1, 1280, g_, g_, device=DEVICE), torch.randn(1, cs, g_, g_, device=DEVICE))
rows.append({"Model": "SG-CAM module only", "Trainable params (M)": f"{sum(p.numel() for p in sg.parameters())/1e6:.2f}",
             "Frozen params (M)": "0.00", "Total params (M)": f"{sum(p.numel() for p in sg.parameters())/1e6:.2f}",
             "GFLOPs / image": f"{fc.get_total_flops()/1e9:.3f}", "Latency, batch 1 (ms)": "–", "Throughput, batch 8 (img/s)": "–"})
T5 = pd.DataFrame(rows); T5.to_csv(os.path.join(OUT, "tables/T5_complexity_measured.csv"), index=False)
TABLES["T5_complexity_measured"] = T5
print(T5.to_string(index=False))
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "| FP16 autocast:", CFG["AMP"] and DEVICE.type == "cuda")

# %% [markdown]
# ## 13. Summary report and archive

# %%
def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join(out)

all_res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "results/json/*.json")))]
pd.DataFrame([dict(key=r["exp"]["key"], **{f"val_{k}": v for k, v in r["val_metrics"].items()},
                   **{f"test_{k}": v for k, v in r.get("test_metrics", {}).items()},
                   best_epoch=r["best_epoch"], minutes=r["train_minutes"], gpu=r["gpu"]) for r in all_res]
             ).to_csv(os.path.join(OUT, "tables/ALL_RUNS.csv"), index=False)
with open(os.path.join(OUT, "RESULTS_SUMMARY.md"), "w") as fh:
    fh.write(f"# SG-CAM — PatternNet results\n\nGenerated {time.ctime()} | device: "
             f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | torch {torch.__version__}\n\n")
    fh.write(f"Dataset root: `{DATA_ROOT}` | {len(DF)} images, {NUM_CLASSES} classes | split sizes: "
             f"{DF.split.value_counts().to_dict()}\n\nRepresentative run for figures: `{REP_KEY}`\n\n")
    for name, t in TABLES.items():
        fh.write(f"## {name}\n\n{md_table(t)}\n\n")
    todo = [e["key"] for e in ORDER if not os.path.exists(result_path(e["key"]))]
    fh.write("## Runs still pending\n\n" + ("\n".join(f"- {k}" for k in todo) if todo else "None") + "\n")
print(open(os.path.join(OUT, "RESULTS_SUMMARY.md")).read())

zp = os.path.join(os.path.dirname(OUT), "sgcam_tables_figures.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ["tables", "figures", "splits"]:
        for f in glob.glob(os.path.join(OUT, sub, "*")):
            z.write(f, os.path.relpath(f, OUT))
    z.write(os.path.join(OUT, "RESULTS_SUMMARY.md"), "RESULTS_SUMMARY.md")
print("Archive:", zp)
