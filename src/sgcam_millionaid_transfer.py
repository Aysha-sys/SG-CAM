# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SG-CAM — cross-dataset transfer (Million-AID) and robustness analysis (PeerJ CS cs-132552, revision 3)
#
# This notebook re-uses the networks trained on PatternNet by the companion notebook **SGCAM** (its output must be attached as an input) and answers the reviewers' concerns about the cross-dataset protocol with three pre-specified experiments:
#
# 1. **Frozen-feature linear probing on Million-AID.** A PatternNet classifier cannot output Million-AID labels, so the PatternNet-trained network is frozen and only a linear (logistic-regression) head is fitted on a labelled training split of the Million-AID target classes; the regularisation strength is selected on a separate validation split, and results are reported once on a held-out test split. Five stratified 60/20/20 splits × three PatternNet seeds.
# 2. **Few-shot transfer** (k = 1, 5, 10, 20 labelled images per class) with a parameter-free nearest-prototype (cosine) classifier.
# 3. **Robustness of the PatternNet classifiers** to image degradations typical of remote sensing (blur, noise, illumination, haze, JPEG compression, resolution loss).
#
# Feature extractors compared under identical protocols:
#
# | ID | Extractor | Purpose |
# |---|---|---|
# | A | ImageNet MobileNetV2 (no PatternNet training) | reference |
# | B | PatternNet MobileNetV2 + CLAHE, **without** SG-CAM | effect of PatternNet training |
# | C | PatternNet **SG-CAM** (MobileNetV2 + SegFormer-B2 + SG-CAM) | proposed |
# | D | Frozen SegFormer-B2 decoder features only (global average pooled) | semantic stream alone |
# | E | PatternNet MobileNetV2 + SegFormer-B2, **concatenation** fusion (if its checkpoints are attached) | isolates SG-CAM from the mere addition of the semantic stream |
#
# **Target data.** Million-AID labelled set (10,000 images, 51 fine classes; Long et al., IEEE JSTARS 2021) obtained from the Hugging Face mirror `jonathan-roberts1/Million-AID`. The four fine classes of the Million-AID *arable land* category are used: dry field, greenhouse, paddy field, terraced field.
#
# **Kaggle settings:** GPU (T4), Internet **ON**; inputs: (i) the **SGCAM** notebook output, (ii) the PatternNet dataset (for the robustness study).

# %%
import importlib, subprocess, sys
for _pkg in ["transformers", "datasets"]:
    try:
        importlib.import_module(_pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)

# %%
import os, re, json, time, math, glob, random, shutil, warnings, zipfile
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import OrderedDict, Counter
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, auc, confusion_matrix)
from scipy import stats
from PIL import Image
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

SMOKE = os.environ.get("SGCAM_SMOKE", "0") == "1"

CFG = dict(
    OUT_DIR="/kaggle/working/transfer_outputs",
    SGCAM_OUTPUT=None,                      # None = auto-detect the attached SGCAM notebook output (sgcam_outputs)
    PATTERNNET_ROOT=None,                   # None = auto-detect
    HF_DATASET="jonathan-roberts1/Million-AID",
    TARGET_CLASSES=["dry field", "greenhouse", "paddy field", "terraced field"],   # Million-AID "arable land"
    IMG_SIZE=256, CLAHE_CLIP=2.0, CLAHE_TILE=8, D_ATTN=256, DROPOUT=0.5,
    SEGFORMER="nvidia/segformer-b2-finetuned-ade-512-512",
    SPLIT_SEEDS=[0, 1, 2, 3, 4], SPLIT=(0.6, 0.2, 0.2),
    PROBE_C_GRID=[0.001, 0.01, 0.1, 1.0, 10.0],
    FEWSHOT_K=[1, 5, 10, 20], FEWSHOT_EPISODES=10,
    BATCH=64, NUM_WORKERS=4, AMP=True,
    RUN_ROBUSTNESS=True,
)
if SMOKE:
    _s = "/tmp/sgcam_smoke"
    CFG.update(OUT_DIR=f"{_s}/transfer_out", SGCAM_OUTPUT=f"{_s}/out", PATTERNNET_ROOT=f"{_s}/data/PatternNet/images",
               IMG_SIZE=64, BATCH=8, NUM_WORKERS=0, AMP=False, FEWSHOT_K=[1, 3], FEWSHOT_EPISODES=2, SPLIT_SEEDS=[0, 1])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
OUT = CFG["OUT_DIR"]
for sub in ["tables", "figures", "features", "splits", "preds"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
T0 = time.time()
print("Device:", DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% [markdown]
# ## 1. Shared preprocessing and model definitions (identical to the SGCAM notebook)

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
# ## 2. Locate the PatternNet checkpoints produced by the SGCAM notebook

# %%
def find_sgcam_output():
    """Locate the folder that contains checkpoints/ (and splits/) written by the SGCAM notebook."""
    if CFG["SGCAM_OUTPUT"]:
        return CFG["SGCAM_OUTPUT"]
    hits = glob.glob("/kaggle/input/**/checkpoints/mobilenet_v2_sgcam_*.pt", recursive=True)
    if hits:   # several attached outputs: use the one with the most checkpoints
        roots = Counter(os.path.dirname(os.path.dirname(h)) for h in glob.glob("/kaggle/input/**/checkpoints/*.pt", recursive=True))
        return roots.most_common(1)[0][0]
    print("Folders under /kaggle/input (depth <= 4):")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        depth = dirpath.count(os.sep) - "/kaggle/input".count(os.sep)
        if depth <= 4 and len(dirnames) < 30:
            print("  ", dirpath, f"({len(filenames)} files)")
        if depth >= 4 or len(dirnames) >= 30:
            dirnames[:] = []
    raise FileNotFoundError("SGCAM checkpoints not found. Add Input -> Your Work -> Notebooks -> SGCAM "
                            "(the successful version with ~245 MB output).")

SG_OUT = find_sgcam_output()
KIND_FUSION = {"B": "none", "C": "sgcam", "E": "concat"}
KIND_LABEL = {"B": "B: no SG-CAM", "C": "C: SG-CAM", "E": "E: concatenation fusion"}
CKPTS = {"B": {}, "C": {}, "E": {}}
# collect checkpoints from ALL attached SGCAM outputs (e.g. "SGCAM" + "SGCAM-extra"); de-duplicated by (kind, seed)
CKPT_FILES = sorted(glob.glob(os.path.join(SG_OUT, "checkpoints", "*.pt"))) if CFG["SGCAM_OUTPUT"] \
    else sorted(glob.glob("/kaggle/input/**/checkpoints/*.pt", recursive=True))
for p in CKPT_FILES:
    n = os.path.basename(p)
    m = re.match(r"mobilenet_v2_(none|sgcam|concat)_cl[\d.]+t\d+_i\d+_d\d+_lr[\de.-]+_bs\d+_ep\d+_s(\d+)\.pt$", n)
    if m:
        CKPTS[{"none": "B", "sgcam": "C", "concat": "E"}[m.group(1)]].setdefault(int(m.group(2)), p)
print("SGCAM output(s):", sorted({os.path.dirname(os.path.dirname(f)) for f in CKPT_FILES}))
print("B (no SG-CAM) seeds:", sorted(CKPTS["B"]), "| C (SG-CAM) seeds:", sorted(CKPTS["C"]), "| E (concat) seeds:", sorted(CKPTS["E"]))
KINDS = [k for k in ["B", "C", "E"] if CKPTS[k]]
assert CKPTS["B"] and CKPTS["C"], "Checkpoints of the 'main' and 'transfer_baselines' groups are required."
NUM_PN_CLASSES = None

_SEM = {}
def get_sem_encoder():
    if "enc" not in _SEM:
        _SEM["enc"] = SemanticEncoder(CFG["SEGFORMER"]).to(DEVICE).eval()
    return _SEM["enc"]

def load_scene_model(kind, seed):
    global NUM_PN_CLASSES
    sd = torch.load(CKPTS[kind][seed], map_location="cpu")
    NUM_PN_CLASSES = sd["head.3.weight"].shape[0]
    c_s = get_sem_encoder().out_dim
    m = SceneModel("mobilenet_v2", KIND_FUSION[kind], NUM_PN_CLASSES, c_s, CFG["D_ATTN"], CFG["DROPOUT"], False)
    m.load_state_dict(sd); return m.to(DEVICE).to(memory_format=torch.channels_last).eval()

def load_imagenet_mnv2():
    m = torchvision.models.mobilenet_v2(weights=None if SMOKE else torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    return m.features.to(DEVICE).to(memory_format=torch.channels_last).eval()

# %% [markdown]
# ## 3. Million-AID target data (arable-land classes) and five stratified 60/20/20 splits

# %%
def load_maid():
    if SMOKE:
        rng = np.random.RandomState(0); recs = []
        for ci, c in enumerate(CFG["TARGET_CLASSES"]):
            for i in range(20):
                a = np.full((80, 80, 3), 60 + 40 * ci, np.uint8) + rng.randint(0, 60, (80, 80, 3)).astype(np.uint8)
                recs.append((Image.fromarray(a), c))
        return recs, {c: 20 for c in CFG["TARGET_CLASSES"]}
    from datasets import load_dataset
    ds = load_dataset(CFG["HF_DATASET"], split="train")
    f = ds.features["label_3"]
    lab = ds["label_3"]
    names = [f.names[v] for v in lab] if hasattr(f, "names") else list(lab)
    all_counts = Counter(names)
    pd.DataFrame(sorted(all_counts.items(), key=lambda x: -x[1]), columns=["Million-AID fine class", "Images"]).to_csv(
        os.path.join(OUT, "tables/T_maid_all_class_counts.csv"), index=False)
    idx = [i for i, n in enumerate(names) if n in CFG["TARGET_CLASSES"]]
    sub = ds.select(idx)
    recs = []
    for r, i in zip(sub, idx):          # convert immediately: some Million-AID images exceed 10,000 px
        im = r["image"].convert("RGB"); im.thumbnail((1024, 1024)); recs.append((im, names[i]))
    return recs, {c: all_counts[c] for c in CFG["TARGET_CLASSES"]}

t = time.time()
recs, counts = load_maid()
TCLASSES = list(CFG["TARGET_CLASSES"]); NT = len(TCLASSES)
print("Target class counts:", counts, f"({time.time()-t:.0f}s)")

def to_uint8_array(pil, clahe=True):
    rgb = np.array(pil.convert("RGB"))
    if clahe:
        rgb = apply_clahe(rgb, CFG["CLAHE_CLIP"], CFG["CLAHE_TILE"])
    rgb = cv2.resize(rgb, (CFG["IMG_SIZE"], CFG["IMG_SIZE"]), interpolation=cv2.INTER_AREA)
    return rgb.transpose(2, 0, 1)

XM = np.stack([to_uint8_array(p) for p, _ in recs])                      # (N,3,S,S) uint8, CLAHE applied
XM_RAW = np.stack([cv2.resize(np.array(p.convert("RGB")), (CFG["IMG_SIZE"],) * 2, interpolation=cv2.INTER_AREA)
                   for p, _ in recs])                                       # for figures
YM = np.array([TCLASSES.index(c) for _, c in recs])
del recs
print("Million-AID tensor:", XM.shape, "| per class:", np.bincount(YM).tolist())

SPLITS = {}
for s in CFG["SPLIT_SEEDS"]:
    i_tr, i_tmp = train_test_split(np.arange(len(YM)), test_size=0.4, stratify=YM, random_state=s)
    i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=YM[i_tmp], random_state=s)
    SPLITS[s] = dict(train=i_tr, val=i_va, test=i_te)
    pd.DataFrame({"index": np.r_[i_tr, i_va, i_te], "class": [TCLASSES[y] for y in YM[np.r_[i_tr, i_va, i_te]]],
                  "split": ["train"] * len(i_tr) + ["val"] * len(i_va) + ["test"] * len(i_te)}).to_csv(
        os.path.join(OUT, f"splits/maid_split_seed{s}.csv"), index=False)
s0 = SPLITS[CFG["SPLIT_SEEDS"][0]]
pd.DataFrame({"Class": TCLASSES, "Total": np.bincount(YM, minlength=NT),
              "Train": np.bincount(YM[s0["train"]], minlength=NT), "Validation": np.bincount(YM[s0["val"]], minlength=NT),
              "Test": np.bincount(YM[s0["test"]], minlength=NT)}).to_csv(os.path.join(OUT, "tables/T_maid_dataset.csv"), index=False)

# %% [markdown]
# ## 4. Frozen feature extraction (A, B×seeds, C×seeds, D)

# %%
@torch.no_grad()
def extract(X, kind, model=None):
    feats = []; use_amp = CFG["AMP"] and DEVICE.type == "cuda"
    enc = get_sem_encoder() if kind in ("C", "D", "E") else None
    for i in range(0, len(X), CFG["BATCH"]):
        x = to_input(torch.from_numpy(X[i:i + CFG["BATCH"]]))
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            if kind == "A":
                f = F.adaptive_avg_pool2d(model(x), 1).flatten(1)
            elif kind == "B":
                f = model(x)["feat"]
            elif kind in ("C", "E"):
                f = model(x, enc(x.contiguous())[0])["feat"]
            else:
                f = F.adaptive_avg_pool2d(enc(x.contiguous())[0], 1).flatten(1)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)

EXTRACTORS = OrderedDict()     # name -> list of (ckpt_seed, features)
t = time.time()
EXTRACTORS["A: ImageNet MobileNetV2"] = [(None, extract(XM, "A", load_imagenet_mnv2()))]
EXTRACTORS["B: PatternNet MobileNetV2 (no SG-CAM)"] = [(s, extract(XM, "B", load_scene_model("B", s))) for s in sorted(CKPTS["B"])]
EXTRACTORS["C: PatternNet SG-CAM (proposed)"] = [(s, extract(XM, "C", load_scene_model("C", s))) for s in sorted(CKPTS["C"])]
EXTRACTORS["D: SegFormer-B2 features only"] = [(None, extract(XM, "D"))]
if CKPTS["E"]:
    EXTRACTORS["E: PatternNet concatenation fusion"] = [(s, extract(XM, "E", load_scene_model("E", s))) for s in sorted(CKPTS["E"])]
for k, v in EXTRACTORS.items():
    print(f"{k}: {len(v)} model(s), dim {v[0][1].shape[1]}")
print(f"feature extraction {time.time()-t:.0f}s")
np.savez_compressed(os.path.join(OUT, "features/maid_features.npz"),
                    **{f"{k[:1]}_s{s}": f for k, v in EXTRACTORS.items() for s, f in v}, y=YM)

# %% [markdown]
# ## 5. Linear probe (logistic regression; C selected on validation macro-F1; test evaluated once)

# %%
def probe(feat, sp):
    sc = StandardScaler().fit(feat[sp["train"]])
    Xtr, Xva, Xte = sc.transform(feat[sp["train"]]), sc.transform(feat[sp["val"]]), sc.transform(feat[sp["test"]])
    best = (-1, None)
    for Cr in CFG["PROBE_C_GRID"]:
        clf = LogisticRegression(C=Cr, max_iter=5000, class_weight="balanced").fit(Xtr, YM[sp["train"]])
        f1v = f1_score(YM[sp["val"]], clf.predict(Xva), average="macro")
        if f1v > best[0]:
            best = (f1v, Cr, clf)
    clf = best[2]; prob = clf.predict_proba(Xte); yt = YM[sp["test"]]; yp = prob.argmax(1)
    m = dict(accuracy=accuracy_score(yt, yp) * 100, balanced_accuracy=balanced_accuracy_score(yt, yp) * 100,
             macro_f1=f1_score(yt, yp, average="macro") * 100,
             macro_auc=roc_auc_score(yt, prob, multi_class="ovr", average="macro") if NT > 2 else roc_auc_score(yt, prob[:, 1]),
             C=best[1], val_macro_f1=best[0] * 100)
    return m, prob, yp

PROBE_ROWS = []; PROBE_PREDS = {}
for name, models in EXTRACTORS.items():
    for ck_seed, feat in models:
        for ss, sp in SPLITS.items():
            m, prob, yp = probe(feat, sp)
            PROBE_ROWS.append(dict(extractor=name, ckpt_seed=ck_seed, split_seed=ss, **m))
            PROBE_PREDS[(name, ck_seed, ss)] = (prob, yp)
PR = pd.DataFrame(PROBE_ROWS); PR.to_csv(os.path.join(OUT, "tables/ALL_probe_runs.csv"), index=False)

def ms(v, nd=2):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}" if len(v) > 1 else f"{v.mean():.{nd}f}"

rows = []
for name in EXTRACTORS:
    d = PR[PR.extractor == name]
    rows.append({"Feature extractor (frozen)": name, "Accuracy (%)": ms(d.accuracy), "Balanced accuracy (%)": ms(d.balanced_accuracy),
                 "Macro F1 (%)": ms(d.macro_f1), "Macro AUC (OvR)": ms(d.macro_auc, 4), "Runs (n)": len(d)})
T_PROBE = pd.DataFrame(rows); T_PROBE.to_csv(os.path.join(OUT, "tables/T_maid_linear_probe.csv"), index=False)
print(T_PROBE.to_string(index=False))

# paired comparison: C vs B on identical (checkpoint seed, split seed) pairs
def paired(nameX, nameY, metric):
    a = PR[PR.extractor == nameX].set_index(["ckpt_seed", "split_seed"])[metric]
    b = PR[PR.extractor == nameY].set_index(["ckpt_seed", "split_seed"])[metric]
    if a.index.equals(b.index) is False:
        b = b.reindex(a.index)
    d = (a - b).dropna().values
    if len(d) < 2: return dict(n=len(d), mean_diff=float(d.mean()) if len(d) else np.nan, wilcoxon_p=np.nan, ttest_p=np.nan)
    w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(n=len(d), mean_diff=d.mean(), std_diff=d.std(ddof=1), wilcoxon_p=w, ttest_p=stats.ttest_1samp(d, 0).pvalue)
names = list(EXTRACTORS)
cmp_rows = []
for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
    for other, lab in [(names[1], "C (SG-CAM) − B (no SG-CAM)")] + ([(names[4], "C (SG-CAM) − E (concatenation)")] if len(names) > 4 else []):
        r = paired(names[2], other, metric)
        cmp_rows.append({"Comparison": lab, "Metric": metric, "Pairs (n)": r["n"],
                         "Mean difference (pp)": f"{r['mean_diff']:+.2f}", "Wilcoxon p": f"{r['wilcoxon_p']:.3g}",
                         "Paired t-test p": f"{r['ttest_p']:.3g}"})
T_CMP = pd.DataFrame(cmp_rows); T_CMP.to_csv(os.path.join(OUT, "tables/T_maid_probe_paired_tests.csv"), index=False)
print(T_CMP.to_string(index=False))

# %% [markdown]
# ## 6. Few-shot transfer (nearest prototype, cosine similarity; k labelled images per class from the training split)

# %%
def fewshot(feat, sp, k, rng):
    tr = sp["train"]; sup = []
    for c in range(NT):
        cand = tr[YM[tr] == c]
        sup.append(rng.choice(cand, min(k, len(cand)), replace=False))
    fn = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    protos = np.stack([fn[s].mean(0) for s in sup]); protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    yp = (fn[sp["test"]] @ protos.T).argmax(1); yt = YM[sp["test"]]
    return accuracy_score(yt, yp) * 100, balanced_accuracy_score(yt, yp) * 100, f1_score(yt, yp, average="macro") * 100

FS = []
for name, models in EXTRACTORS.items():
    for ck_seed, feat in models:
        for ss, sp in SPLITS.items():
            for k in CFG["FEWSHOT_K"]:
                rng = np.random.RandomState(1000 * ss + k)
                for ep in range(CFG["FEWSHOT_EPISODES"]):
                    a, ba, f1m = fewshot(feat, sp, k, rng)
                    FS.append(dict(extractor=name, ckpt_seed=ck_seed, split_seed=ss, k=k, episode=ep, accuracy=a,
                                   balanced_accuracy=ba, macro_f1=f1m))
FS = pd.DataFrame(FS); FS.to_csv(os.path.join(OUT, "tables/ALL_fewshot_episodes.csv"), index=False)
T_FS = FS.groupby(["extractor", "k"]).agg(mean_bal_acc=("balanced_accuracy", "mean"), std_bal_acc=("balanced_accuracy", "std"),
                                          mean_f1=("macro_f1", "mean"), std_f1=("macro_f1", "std"), n=("macro_f1", "size")).reset_index()
T_FS_fmt = T_FS.pivot(index="extractor", columns="k", values="mean_bal_acc").round(2)
T_FS.to_csv(os.path.join(OUT, "tables/T_maid_fewshot.csv"), index=False)
print("Few-shot balanced accuracy (%):"); print(T_FS_fmt.to_string())

# %% [markdown]
# ## 7. Robustness of the PatternNet classifiers (B vs. C) to image degradations
# Degradations are applied to the raw image **before** CLAHE, i.e. as they would occur at acquisition. SegFormer features are recomputed for every degraded image (no cache).

# %%
CORRUPTIONS = OrderedDict([
    ("Gaussian blur (σ)", [1, 2, 3]),
    ("Gaussian noise (σ)", [0.04, 0.08, 0.12]),
    ("Brightness (factor)", [0.5, 0.75, 1.5]),
    ("Haze (α)", [0.2, 0.4, 0.6]),
    ("JPEG (quality)", [30, 15, 5]),
    ("Down-sampling (×)", [2, 4, 8]),
])

def corrupt(rgb, kind, lvl, seed):
    x = rgb.astype(np.float32)
    if kind.startswith("Gaussian blur"):
        return cv2.GaussianBlur(rgb, (0, 0), lvl)
    if kind.startswith("Gaussian noise"):
        x = x + np.random.RandomState(seed).normal(0, lvl * 255, x.shape)
    elif kind.startswith("Brightness"):
        x = x * lvl
    elif kind.startswith("Haze"):
        x = x * (1 - lvl) + 255 * lvl
    elif kind.startswith("JPEG"):
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(lvl)])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif kind.startswith("Down-sampling"):
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (max(1, w // lvl), max(1, h // lvl)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(x, 0, 255).astype(np.uint8)

class CorruptDS(Dataset):
    def __init__(self, paths, labels, kind, lvl):
        self.p, self.y, self.k, self.l = paths, labels, kind, lvl
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        rgb = read_rgb(self.p[i])
        if self.k is not None:
            rgb = corrupt(rgb, self.k, self.l, i)
        rgb = apply_clahe(rgb, CFG["CLAHE_CLIP"], CFG["CLAHE_TILE"])
        if rgb.shape[0] != CFG["IMG_SIZE"]:
            rgb = cv2.resize(rgb, (CFG["IMG_SIZE"],) * 2, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))), int(self.y[i])

ROB = None
if CFG["RUN_ROBUSTNESS"]:
    split_csv = os.path.join(SG_OUT, "splits/patternnet_split.csv")
    pn_root = CFG["PATTERNNET_ROOT"]
    if pn_root is None:
        for dirpath, dirnames, _ in os.walk("/kaggle/input"):
            if "sgcam_outputs" in dirpath: dirnames[:] = []; continue
            if len(dirnames) >= 30 and os.path.isdir(os.path.join(dirpath, "airplane")):
                pn_root = dirpath; break
    sp_df = pd.read_csv(split_csv); te = sp_df[sp_df.split == "test"]
    paths = [os.path.join(pn_root, r) for r in te.rel]; labels = te.label.values
    print("PatternNet root:", pn_root, "| test images:", len(paths))
    MODELS = {(k, s): load_scene_model(k, s) for k in KINDS for s in sorted(CKPTS[k])}
    enc = get_sem_encoder(); use_amp = CFG["AMP"] and DEVICE.type == "cuda"
    settings = [("Clean", None, None)] + [(k, k, l) for k, ls in CORRUPTIONS.items() for l in ls]
    rows = []
    for label, kind, lvl in settings:
        t = time.time()
        dl = DataLoader(CorruptDS(paths, labels, kind, lvl), batch_size=CFG["BATCH"], num_workers=CFG["NUM_WORKERS"])
        correct = {mk: 0 for mk in MODELS}; n = 0
        with torch.no_grad():
            for x, y in dl:
                x = to_input(x); y = y.to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                    f_seg = enc(x.contiguous())[0]
                    for (k, s), m in MODELS.items():
                        out = m(x, None if k == "B" else f_seg)
                        correct[(k, s)] += (out["logits"].argmax(1) == y).sum().item()
                n += len(y)
        for (k, s), c in correct.items():
            rows.append(dict(corruption=kind or "Clean", level=0 if lvl is None else lvl, model=k, seed=s, accuracy=100 * c / n))
        print(f"  {label} {lvl}: " + "  ".join(f"{k}={np.mean([correct[(k, s)] for s in CKPTS[k]])/n*100:.2f}%" for k in KINDS)
              + f"  ({time.time()-t:.0f}s)")
    ROB = pd.DataFrame(rows); ROB.to_csv(os.path.join(OUT, "tables/ALL_robustness.csv"), index=False)
    tab = []
    for label, kind, lvl in settings:
        c, l = kind or "Clean", 0 if lvl is None else lvl
        sel = ROB[(ROB.corruption == c) & (ROB.level == l)]
        acc = {k: sel[sel.model == k].sort_values("seed").accuracy.values for k in KINDS}
        row = {"Degradation": c, "Level": "–" if lvl is None else lvl}
        for k in KINDS:
            row[KIND_LABEL[k] + " (%)"] = ms(acc[k])
        row["Δ C−B (pp)"] = f"{np.mean(acc['C']) - np.mean(acc['B']):+.2f}"
        if "E" in KINDS:
            row["Δ C−E (pp)"] = f"{np.mean(acc['C']) - np.mean(acc['E']):+.2f}"
        tab.append(row)
    T_ROB = pd.DataFrame(tab); T_ROB.to_csv(os.path.join(OUT, "tables/T_robustness.csv"), index=False)
    cor = ROB[ROB.corruption != "Clean"]
    mca = cor.groupby(["model", "seed"]).accuracy.mean().unstack(0)
    rows_m = [{"Model": KIND_LABEL[k], "Mean accuracy over all degradations (%)": ms(mca[k].dropna()),
               "Seeds (n)": int(mca[k].notna().sum())} for k in KINDS]
    for o in [k for k in ["B", "E"] if k in KINDS]:
        d = (mca["C"] - mca[o]).dropna().values
        rows_m.append({"Model": f"Δ C−{o} (pp), seed-paired", "Mean accuracy over all degradations (%)":
                       f"{d.mean():+.2f} (paired t-test p = {stats.ttest_1samp(d, 0).pvalue:.3g})" if len(d) > 1 else f"{d.mean():+.2f}",
                       "Seeds (n)": len(d)})
    T_MCA = pd.DataFrame(rows_m)
    T_MCA.to_csv(os.path.join(OUT, "tables/T_robustness_summary.csv"), index=False)
    print(T_ROB.to_string(index=False)); print(T_MCA.to_string(index=False))

# %% [markdown]
# ## 8. Figures

# %%
FIG = os.path.join(OUT, "figures")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight"); plt.close(fig); print("[fig]", name)

# samples per class
fig, ax = plt.subplots(NT, 5, figsize=(11, 2.3 * NT), squeeze=False)
for c in range(NT):
    ids = np.where(YM == c)[0][:5]
    for j in range(5):
        a = ax[c, j]; a.axis("off")
        if j < len(ids): a.imshow(XM_RAW[ids[j]])
        if j == 0: a.set_title(TCLASSES[c].title(), loc="left", fontsize=9)
fig.suptitle("Million-AID arable-land classes used as the target domain", y=1.0); savefig(fig, "FM1_maid_samples")

# linear probe bar chart
fig, a = plt.subplots(figsize=(8, 3.6))
nm = list(EXTRACTORS); vals = [PR[PR.extractor == n].balanced_accuracy for n in nm]
a.barh(nm[::-1], [v.mean() for v in vals][::-1], xerr=[v.std(ddof=1) if len(v) > 1 else 0 for v in vals][::-1], capsize=3)
a.set_xlabel("Balanced accuracy on Million-AID test split (%)"); a.set_xlim(max(0, min(v.mean() for v in vals) - 10), 100)
a.set_title("Frozen-feature linear probe: PatternNet → Million-AID (arable land)"); savefig(fig, "FM2_linear_probe")

# confusion matrices B vs C (split seed 0, first checkpoint seed)
ss0 = CFG["SPLIT_SEEDS"][0]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for a, (name, mods) in zip(axs, [(nm[1], EXTRACTORS[nm[1]]), (nm[2], EXTRACTORS[nm[2]])]):
    prob, yp = PROBE_PREDS[(name, mods[0][0], ss0)]; yt = YM[SPLITS[ss0]["test"]]
    cm = confusion_matrix(yt, yp, labels=range(NT)); M = cm / cm.sum(1, keepdims=True) * 100
    a.imshow(M, cmap="Blues", vmin=0, vmax=100)
    for i in range(NT):
        for j in range(NT):
            a.text(j, i, f"{cm[i, j]}\n({M[i, j]:.0f}%)", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 50 else "black")
    a.set_xticks(range(NT)); a.set_yticks(range(NT))
    a.set_xticklabels([c.title() for c in TCLASSES], rotation=30, ha="right"); a.set_yticklabels([c.title() for c in TCLASSES])
    a.set_xlabel("Predicted class"); a.set_ylabel("True class"); a.set_title(f"Confusion Matrix — {name.split(':')[0]}", fontsize=10)
fig.suptitle(f"Million-AID test split (split seed {ss0}; n = {len(yt)})"); savefig(fig, "FM3_confusion_matrices")

# ROC (macro one-vs-rest) for B and C
fig, a = plt.subplots(figsize=(5.5, 5))
for name in nm[1:3]:
    prob, _ = PROBE_PREDS[(name, EXTRACTORS[name][0][0], ss0)]; yt = YM[SPLITS[ss0]["test"]]
    Yb = label_binarize(yt, classes=range(NT)); grid = np.linspace(0, 1, 501)
    tprs = [np.interp(grid, *roc_curve(Yb[:, i], prob[:, i])[:2]) for i in range(NT)]
    mt = np.mean(tprs, 0); mt[0] = 0
    a.plot(grid, mt, label=f"{name.split(':')[0]} (macro AUC = {auc(grid, mt):.3f})")
a.plot([0, 1], [0, 1], "k:", lw=0.8); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
a.set_title("Macro-average ROC, Million-AID test split"); a.legend(loc="lower right"); savefig(fig, "FM4_roc")

# few-shot curves
fig, a = plt.subplots(figsize=(6.5, 4.2))
for name in nm:
    d = T_FS[T_FS.extractor == name]
    a.errorbar(d.k, d.mean_bal_acc, yerr=d.std_bal_acc, marker="o", capsize=3, label=name)
a.set_xscale("log"); a.set_xticks(CFG["FEWSHOT_K"]); a.set_xticklabels(CFG["FEWSHOT_K"])
a.set_xlabel("Labelled images per class (k)"); a.set_ylabel("Balanced accuracy (%)")
a.set_title("Few-shot transfer to Million-AID (nearest prototype)"); a.legend(fontsize=7.5); savefig(fig, "FM5_fewshot")

# misclassified Million-AID examples (C)
prob, yp = PROBE_PREDS[(nm[2], EXTRACTORS[nm[2]][0][0], ss0)]; te_idx = SPLITS[ss0]["test"]; yt = YM[te_idx]
wrong = np.where(yp != yt)[0]
if len(wrong):
    wrong = wrong[np.argsort(-prob[wrong, yp[wrong]])][:12]
    nc = 4; nr = math.ceil(len(wrong) / nc)
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 2.7, nr * 3), squeeze=False)
    for i, a in enumerate(ax.ravel()):
        a.axis("off")
        if i < len(wrong):
            w = wrong[i]; a.imshow(XM_RAW[te_idx[w]])
            a.set_title(f"True: {TCLASSES[yt[w]].title()}\nPred: {TCLASSES[yp[w]].title()} ({prob[w, yp[w]]:.2f})", fontsize=8)
    fig.suptitle("Misclassified Million-AID test images (SG-CAM features, linear probe)", y=1.0); savefig(fig, "FM6_maid_misclassified")

# robustness curves
if ROB is not None:
    nk = len(CORRUPTIONS); fig, axs = plt.subplots(1, nk, figsize=(3.2 * nk, 3.3), squeeze=False)
    clean = ROB[ROB.corruption == "Clean"].groupby("model").accuracy.mean()
    for a, (k, ls) in zip(axs[0], CORRUPTIONS.items()):
        for mdl, lab in [(k, KIND_LABEL[k]) for k in KINDS]:
            g = ROB[(ROB.corruption == k) & (ROB.model == mdl)].groupby("level").accuracy
            mu = [clean[mdl]] + [g.mean()[l] for l in ls]; sd = [0] + [g.std()[l] if not np.isnan(g.std()[l]) else 0 for l in ls]
            a.errorbar(range(len(ls) + 1), mu, yerr=sd, marker="o", capsize=2, label=lab)
        a.set_xticks(range(len(ls) + 1)); a.set_xticklabels(["clean"] + [str(l) for l in ls], fontsize=8)
        a.set_title(k, fontsize=9); a.set_ylabel("Test accuracy (%)")
    axs[0][0].legend(fontsize=8); fig.suptitle("Robustness of PatternNet classifiers to image degradations (mean ± std over seeds)")
    fig.tight_layout(); savefig(fig, "FM7_robustness")

# %% [markdown]
# ## 9. Summary and archive

# %%
def md_table(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values])
with open(os.path.join(OUT, "TRANSFER_SUMMARY.md"), "w") as fh:
    fh.write(f"# SG-CAM transfer and robustness results\n\nGenerated {time.ctime()} | "
             f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | runtime {(time.time()-T0)/60:.1f} min\n\n")
    fh.write(f"Target classes: {TCLASSES} | counts: {np.bincount(YM).tolist()} | split seeds: {CFG['SPLIT_SEEDS']} | "
             f"PatternNet checkpoint seeds B: {sorted(CKPTS['B'])}, C: {sorted(CKPTS['C'])}, E: {sorted(CKPTS['E'])}\n\n")
    for title, t in [("Million-AID subset", pd.read_csv(os.path.join(OUT, "tables/T_maid_dataset.csv"))),
                     ("Linear probe", T_PROBE), ("Paired tests (C vs B)", T_CMP),
                     ("Few-shot balanced accuracy (%)", T_FS_fmt.reset_index())] + \
                    ([("Robustness", T_ROB), ("Robustness summary", T_MCA)] if ROB is not None else []):
        fh.write(f"## {title}\n\n{md_table(t)}\n\n")
print(open(os.path.join(OUT, "TRANSFER_SUMMARY.md")).read())
zp = os.path.join(os.path.dirname(OUT), "transfer_tables_figures.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ["tables", "figures", "splits"]:
        for f in glob.glob(os.path.join(OUT, sub, "*")):
            z.write(f, os.path.relpath(f, OUT))
    z.write(os.path.join(OUT, "TRANSFER_SUMMARY.md"), "TRANSFER_SUMMARY.md")
print("Archive:", zp, f"| total runtime {(time.time()-T0)/60:.1f} min")
