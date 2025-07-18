import os
import re
import datetime
import subprocess
import json
import csv

from pathlib import Path

import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from torcheeg.models import EEGNet
from torcheeg import transforms
from run_index import append_run_index

"""02_train_decoder_dec1_5class.py
5-class EEGNet decoder for **–1 transitions**.
Classes = landing digits 1-5 (after –1 change).
No data augmentation.  Produces per-fold & overall 5×5 confusion matrices and a
rich TXT report with hyper-parameters and recalls.
"""

import argparse, yaml, textwrap, sys

BATCH_DEFAULTS = {
    "batch": 64,
    "lr": 1e-4,
    "epochs": 100,
    "early_stop": 15,
    "shift_p": 0.5,
    "shift_min_frac": 0.005,
    "shift_max_frac": 0.04,
    "scale_p": 0.5,
    "scale_min": 0.9,
    "scale_max": 1.1,
    "noise_p": 0.3,
    "noise_std": 0.02,
}
DEFAULTS = {"dataset_dir": str(Path("data_preprocessed/acc_1_dataset")), "channel_drop":0.0, "max_folds":None, **BATCH_DEFAULTS}

def load_cfg(p):
    if p is None: return {}
    fp=Path(p);
    if not fp.exists(): sys.exit(f"cfg {p} missing")
    txt=fp.read_text(); return yaml.safe_load(txt) if fp.suffix.lower() in {'.yml','.yaml'} else json.loads(txt)

def parse_args():
    pr=argparse.ArgumentParser(description="dec1 5class decoder YAML")
    pr.add_argument('--cfg');
    for k in DEFAULTS: pr.add_argument(f"--{k}")
    pr.add_argument('--set', nargs='*')
    return pr.parse_args()

args=parse_args(); cfg=DEFAULTS.copy(); cfg.update(load_cfg(args.cfg)); cfg.update({k:getattr(args,k) for k in DEFAULTS if getattr(args,k) is not None})
if args.set:
    for kv in args.set:
        k,v=kv.split('=',1)
        cfg[k]=yaml.safe_load(v) # Rely on yaml.safe_load for type inference

DATASET_DIR=Path(cfg['dataset_dir']); BATCH=cfg['batch']; LR=cfg['lr']; EPOCHS=cfg['epochs']; EARLY_STOP=cfg['early_stop']
SHIFT_P=cfg['shift_p']; SHIFT_MIN_FRAC=cfg['shift_min_frac']; SHIFT_MAX_FRAC=cfg['shift_max_frac']; SCALE_P=cfg['scale_p']; SCALE_MIN=cfg['scale_min']; SCALE_MAX=cfg['scale_max']; NOISE_P=cfg['noise_p']; NOISE_STD=cfg['noise_std']; CHANNEL_DROP=cfg['channel_drop']; MAX_FOLDS=cfg['max_folds']

DEC_SET = {"21","32","43","54","65"}
SCRIPT_NAME = Path(__file__).stem
RUN_ID = datetime.datetime.now().strftime(f"%Y%m%d_%H%M_{SCRIPT_NAME}")
RUN_DIR = Path("results") / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

BATCH = 64
LR = 1e-4
EPOCHS = 100
EARLY_STOP = 15

NOISE_STD = 0.01
CHANNEL_DROP = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- augmentation hyper-params (fractions refer to window length T) -----
SHIFT_P = 0.5  # probability to apply time-shift
SHIFT_MIN_FRAC = 0.005
SHIFT_MAX_FRAC = 0.04
SCALE_P = 0.5
SCALE_MIN = 0.9
SCALE_MAX = 1.1
NOISE_P = 0.3
NOISE_STD = 0.02  # updated noise std based on landing-digit script

torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)

# ----- helpers -----

def load_epochs():
    files=sorted(DATASET_DIR.glob("sub-*preprocessed-epo.fif"))
    pat=re.compile(r"sub-(\d+)_preprocessed"); eps=[]
    for fp in files:
        ep=mne.read_epochs(fp, preload=True, verbose=False)
        sid=int(pat.search(fp.name).group(1)); ep.metadata['subject']=sid
        mask=ep.metadata['Condition'].astype(str).isin(DEC_SET); ep=ep[mask]; eps.append(ep)
    all_ep=mne.concatenate_epochs(eps)
    cond=all_ep.metadata['Condition'].astype(int); landing=(cond%10).astype(str)  # '1'..'5'
    all_ep.metadata['land_digit']=landing; return all_ep

def build_tensors(ep):
    X=ep.get_data(copy=False)*1e6; X=transforms.MeanStdNormalize(axis=-1)(eeg=X)['eeg']; X_t=torch.from_numpy(X).float().unsqueeze(1)
    classes=np.array(['1','2','3','4','5']); label_to_idx={c:i for i,c in enumerate(classes)}
    y=torch.tensor([label_to_idx[l] for l in ep.metadata['land_digit']],dtype=torch.long)
    groups=ep.metadata['subject'].values; return X_t,y,groups,classes

# ----- augmentation classes (GPU) -----


class RandomAmpScale:
    def __init__(self, p: float = SCALE_P, scale_min: float = SCALE_MIN, scale_max: float = SCALE_MAX):
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=eeg.device).item() < self.p:
            s = torch.empty(1, device=eeg.device).uniform_(self.scale_min, self.scale_max)
            eeg = eeg * s
        return eeg


class RandomTimeShiftTorch:
    def __init__(self, p: float, shift_min: int, shift_max: int, dim: int = -1):
        self.p = p; self.shift_min = shift_min; self.shift_max = shift_max; self.dim = dim

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.shift_max > 0 and torch.rand(1, device=eeg.device).item() < self.p:
            shift = torch.randint(self.shift_min, self.shift_max + 1, (1,), device=eeg.device).item()
            if torch.rand(1, device=eeg.device).item() < 0.5:
                shift = -shift
            eeg = torch.roll(eeg, shift, dims=self.dim)
        return eeg


class RandomNoiseTorch:
    def __init__(self, p: float, std: float):
        self.p = p; self.std = std

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.std > 0 and torch.rand(1, device=eeg.device).item() < self.p:
            eeg = eeg + self.std * torch.randn_like(eeg)
        return eeg


class ComposeTorch:
    def __init__(self, tlist):
        self.tlist = tlist

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        for t in self.tlist:
            eeg = t(eeg)
        return eeg


def make_train_aug(T: int):
    return ComposeTorch([
        RandomTimeShiftTorch(SHIFT_P, max(1, int(SHIFT_MIN_FRAC*T)), int(SHIFT_MAX_FRAC*T)),
        RandomAmpScale(),
        RandomNoiseTorch(NOISE_P, NOISE_STD)
    ])

# ----- training helpers -----

def train_epoch(model, ld, opt, loss, aug):
    model.train(); tot = 0
    for xb, yb in ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xb = aug(xb)
        opt.zero_grad(); l = loss(model(xb), yb); l.backward(); opt.step(); tot += l.item()
    return tot/len(ld)

def eval_epoch(m,ld,loss):
    m.eval(); tot=0; corr=0; N=0
    with torch.no_grad():
        for xb,yb in ld:
            out=m(xb.to(DEVICE)); tot+=loss(out,yb.to(DEVICE)).item(); pred=out.argmax(1).cpu(); corr+=(pred==yb).sum().item(); N+=yb.size(0)
    return tot/len(ld),100*corr/N

if __name__=='__main__':
    ep_all=load_epochs(); X_t,y_t,groups,cls=build_tensors(ep_all); ds=TensorDataset(X_t,y_t); num_classes=5
    csv_f=open(RUN_DIR/f'logs_{SCRIPT_NAME}.csv','w',newline=''); cw=csv.writer(csv_f); cw.writerow(['fold','epoch','tr','val','acc'])
    cm_tot=np.zeros((num_classes,num_classes),int); accs=[]
    summary = {
        "run_id": RUN_ID,
        "git": git_hash(),
        "device": str(DEVICE),
        "task": "dec1_5class",
        "classes": [str(i) for i in range(1,6)], # Classes 1-5
        "hyper": cfg.copy(),
        "fold_metrics":[]
    }
    for fold,(tr_idx,val_idx) in enumerate(LeaveOneGroupOut().split(np.zeros(len(ds)), np.zeros(len(ds)), groups)):
        print(f"\n=== Fold {fold+1} ===")
        tr_ld=DataLoader(Subset(ds,tr_idx),batch_size=BATCH,shuffle=True); val_ld=DataLoader(Subset(ds,val_idx),batch_size=BATCH)
        cls_w=compute_class_weight('balanced',classes=np.arange(num_classes),y=y_t[tr_idx].numpy()); loss_fn=nn.CrossEntropyLoss(torch.tensor(cls_w,dtype=torch.float32).to(DEVICE))
        _,_,C,T=X_t.shape; model=EEGNet(num_classes=num_classes,num_electrodes=C,chunk_size=T,dropout=0.5).to(DEVICE)
        train_aug = make_train_aug(T)
        opt=optim.Adam(model.parameters(),lr=LR); sched=ReduceLROnPlateau(opt,'min',patience=5,factor=0.5)
        best_v=float('inf'); best_a=0; pat=0
        for ep in range(1,EPOCHS+1):
            tr_l = train_epoch(model, tr_ld, opt, loss_fn, train_aug); val_l, val_a = eval_epoch(model, val_ld, loss_fn); sched.step(val_l)
            cw.writerow([fold+1,ep,f"{tr_l:.3f}",f"{val_l:.3f}",f"{val_a:.2f}"])
            if val_l<best_v: best_v,best_a,pat=val_l,val_a,0
            else: pat+=1
            if pat>=EARLY_STOP: break
        print(f"Fold {fold+1} best acc {best_a:.2f}%")
        accs.append(best_a)
        pred=model(X_t[val_idx].to(DEVICE)).argmax(1).cpu().numpy(); cm=confusion_matrix(y_t[val_idx].numpy(),pred,labels=np.arange(num_classes)); cm_tot+=cm
        cm_pct=cm/cm.sum(axis=1,keepdims=True)*100; annot=np.round(cm_pct,1)
        plt.figure(figsize=(6,5)); ax=sns.heatmap(cm_pct,annot=annot,fmt='.1f',cmap='Blues',xticklabels=[f'land {d}' for d in cls],yticklabels=[f'land {d}' for d in cls]); plt.xticks(rotation=0); plt.yticks(rotation=0)
        plt.title(f"Fold {fold+1} Confusion (%) - {SCRIPT_NAME}")
        for i in range(num_classes): ax.add_patch(Rectangle((i,i),1,1,fill=False,edgecolor='black',lw=1.5)); j=int(np.argmax(cm_pct[i])); ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='red',lw=2.5))
        plt.tight_layout(); plt.savefig(RUN_DIR/f'fold{fold+1}_confusion_{SCRIPT_NAME}.png'); plt.close()
    cm_pct_tot=cm_tot/cm_tot.sum(axis=1,keepdims=True)*100; annot=np.round(cm_pct_tot,1); plt.figure(figsize=(6,5)); ax=sns.heatmap(cm_pct_tot,annot=annot,fmt='.1f',cmap='Blues',xticklabels=[f'land {d}' for d in cls],yticklabels=[f'land {d}' for d in cls]); plt.xticks(rotation=0); plt.yticks(rotation=0)
    plt.title(f"Overall Confusion (%) - {SCRIPT_NAME}")
    for i in range(num_classes): ax.add_patch(Rectangle((i,i),1,1,fill=False,edgecolor='black',lw=1.5)); j=int(np.argmax(cm_pct_tot[i])); ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='red',lw=2.5))
    plt.tight_layout(); plt.savefig(RUN_DIR/f'overall_confusion_{SCRIPT_NAME}.png'); plt.close()
    mean_acc=np.mean(accs); std_acc=np.std(accs)
    chance = 100/num_classes
    recalls = np.divide(cm_tot.diagonal(),cm_tot.sum(axis=1),out=np.zeros(num_classes),where=cm_tot.sum(axis=1)!=0)
    best_idx = int(np.argmax(recalls)); worst_idx = int(np.argmin(recalls))
    summary.update({
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc),
        "chance_level_pct": float(chance),
        "confusion_total": cm_tot.tolist(),
        "per_digit_recall": [float(r) for r in recalls],
        "best_digit": str(cls[best_idx]),
        "worst_digit": str(cls[worst_idx]),
    })

    # Persist the enriched summary
    summ_path = RUN_DIR / f"summary_{SCRIPT_NAME}.json"
    with open(summ_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    # Reload to guarantee that the txt report is generated strictly from persisted JSON data
    with open(summ_path) as fp:
        summary_loaded = json.load(fp)

    # Textual report from loaded summary (guarantees consistency)
    report_lines = [
        f"{SCRIPT_NAME} report (raw z-score)",
        f"Run ID        : {summary_loaded['run_id']}",
        "",
        "--- Hyper-parameters ---",
        f"  Batch size          : {summary_loaded['hyper']['batch']}",
        f"  Learning rate       : {summary_loaded['hyper']['lr']}",
        f"  Max epochs          : {summary_loaded['hyper']['epochs']}",
        f"  Early-stop patience : {summary_loaded['hyper']['early_stop']}",
        f"  Shift P / range     : {summary_loaded['hyper']['shift_p']} | {summary_loaded['hyper']['shift_min_frac']}–{summary_loaded['hyper']['shift_max_frac']} × T",
        f"  Scale P / range     : {summary_loaded['hyper']['scale_p']} | {summary_loaded['hyper']['scale_min']}-{summary_loaded['hyper']['scale_max']}",
        f"  Noise P / std       : {summary_loaded['hyper']['noise_p']} | {summary_loaded['hyper']['noise_std']}",
        "",
        "--- Results ---",
        f"Mean LOSO accuracy : {summary_loaded['mean_acc']:.2f}% (±{summary_loaded['std_acc']:.2f}%)",
        f"Chance level      : {summary_loaded['chance_level_pct']:.2f}% (5-class)",
        "",
        "Per-digit recall (TPR):"
    ]
    for i,r in enumerate(summary_loaded['per_digit_recall']):
        report_lines.append(f"  {summary_loaded['classes'][i]} : {r*100:.1f}%   | TP={summary_loaded['confusion_total'][i][i]} / {np.sum(summary_loaded['confusion_total'][i])}")
    report_lines.extend([
        "",
        f"Best recalled digit : {summary_loaded['best_digit']}",
        f"Worst recalled digit: {summary_loaded['worst_digit']}"
    ])
    with open(RUN_DIR / f"report_{SCRIPT_NAME}.txt", "w") as rp:
        rp.write("\n".join(report_lines))
    csv_f.close()

    # Append to expandable index CSV
    append_run_index(summary_loaded, SCRIPT_NAME) 