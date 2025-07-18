import os, re, datetime, subprocess, json, csv
from pathlib import Path
import numpy as np, mne, torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torcheeg.models import EEGNet
from torcheeg import transforms
import argparse, yaml, textwrap, sys  # NEW
from run_index import append_run_index

"""02_train_decoder_inc1_binary.py
Binary EEGNet decoder: detects whether the transition is an **increment of 1**
(e.g., 12, 23, 34, 45, 56) versus any other transition.
Uses accuracy-only dataset (acc_1_dataset), per-channel Z-score features, data
augmentation (noise + channel dropout), LOSO cross-validation.
Outputs confusion matrices with diagonal (black) and max-row (red) boxes, plus a
text report. All artefacts are tagged with the script name for clarity.
"""

# -------------------- CONFIG --------------------
DATASET_DIR = Path("data_preprocessed/acc_1_dataset")
SCRIPT_NAME = Path(__file__).stem
RUN_ID = datetime.datetime.now().strftime(f"%Y%m%d_%H%M_{SCRIPT_NAME}")
RUN_DIR = Path("results") / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

INC_SET = {"12","23","34","45","56"}

DEFAULTS = {
    "dataset_dir": str(DATASET_DIR),
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 100,
    "early_stop": 15,
    "noise_std": 0.0,
    "channel_dropout_p": 0.0,
    "max_folds": None,
}

def load_cfg(p):
    if p is None: return {}
    fp=Path(p)
    if not fp.exists(): sys.exit(f"cfg {p} not found")
    txt=fp.read_text(); return yaml.safe_load(txt) if fp.suffix.lower() in {'.yml','.yaml'} else json.loads(txt)

def parse_args():
    pr=argparse.ArgumentParser(description="inc1 binary decoder with YAML overrides")
    pr.add_argument('--cfg')
    for k in DEFAULTS: pr.add_argument(f"--{k}")
    pr.add_argument('--set', nargs='*')
    return pr.parse_args()

args=parse_args(); cfg=DEFAULTS.copy(); cfg.update(load_cfg(args.cfg)); cfg.update({k:getattr(args,k) for k in DEFAULTS if getattr(args,k) is not None})
if args.set:
    for kv in args.set:
        k,v=kv.split('=',1)
        cfg[k]=yaml.safe_load(v) # Rely on yaml.safe_load for type inference

DATASET_DIR=Path(cfg['dataset_dir']); BATCH_SIZE=cfg['batch_size']; LEARNING_RATE=cfg['lr']; NUM_EPOCHS=cfg['epochs']; EARLY_STOPPING_PATIENCE=cfg['early_stop']; NOISE_STD=cfg['noise_std']; CHANNEL_DROPOUT_P=cfg['channel_dropout_p']; MAX_FOLDS=cfg['max_folds']

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device",DEVICE)

# -------------- helpers --------------

def git_hash():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"],stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"

def load_epochs():
    files=sorted(DATASET_DIR.glob("sub-*preprocessed-epo.fif"))
    subj_re=re.compile(r"sub-(\d+)_preprocessed")
    eps=[]
    for fp in files:
        ep=mne.read_epochs(fp, preload=True, verbose=False)
        sid=int(subj_re.search(fp.name).group(1))
        if ep.metadata is None:
            ep.metadata=mne.create_info([],0)
        ep.metadata['subject']=sid
        eps.append(ep)
    all_ep=mne.concatenate_epochs(eps)
    cond=all_ep.metadata['Condition'].astype(str)
    all_ep.metadata['inc1']=(cond.isin(INC_SET)).astype(int)
    return all_ep

def build_tensors(ep):
    X=ep.get_data(copy=False)*1e6
    X=transforms.MeanStdNormalize(axis=-1)(eeg=X)['eeg']
    X_t=torch.from_numpy(X).float().unsqueeze(1)
    y=np.asarray(ep.metadata['inc1'],dtype=int)
    y_t=torch.from_numpy(y).long()
    groups=ep.metadata['subject'].values
    return X_t,y_t,groups

# -------------- train helpers --------------

def train_epoch(model,ld,opt,loss):
    model.train();tot=0
    for xb,yb in ld:
        xb,yb=xb.to(DEVICE), yb.to(DEVICE)
        if NOISE_STD>0: xb+=NOISE_STD*torch.randn_like(xb)
        if CHANNEL_DROPOUT_P>0:
            m=(torch.rand(xb.size(0),1,xb.size(2),1,device=xb.device)>CHANNEL_DROPOUT_P).float(); xb*=m
        opt.zero_grad(); out=model(xb); l=loss(out,yb); l.backward(); opt.step(); tot+=l.item()
    return tot/len(ld)

def eval_epoch(model,ld,loss):
    model.eval(); tot=0; corr=0; N=0
    with torch.no_grad():
        for xb,yb in ld:
            xb,yb=xb.to(DEVICE), yb.to(DEVICE)
            out=model(xb); tot+=loss(out,yb).item(); pred=out.argmax(1); corr+=(pred==yb).sum().item(); N+=yb.size(0)
    return tot/len(ld),100*corr/N

# -------------- main --------------
if __name__=='__main__':
    torch.backends.cudnn.benchmark=True
    ep_all=load_epochs(); X_t,y_t,groups=build_tensors(ep_all)
    full_ds=TensorDataset(X_t,y_t)
    num_classes=2
    summary={
        "run_id":RUN_ID,
        "task":"inc1_binary",
        "git":git_hash(),
        "device":str(DEVICE),
        "classes": [str(i) for i in range(2)],
        "hyper": cfg.copy(),
        "fold_metrics":[]
    }
    csv_f=open(RUN_DIR/f"logs_{SCRIPT_NAME}.csv","w",newline=""); cw=csv.writer(csv_f); cw.writerow(["fold","epoch","tr","val","acc"])
    logo=LeaveOneGroupOut(); cm_total=np.zeros((2,2),int); accs=[]
    for fold,(tr_idx,val_idx) in enumerate(logo.split(np.zeros(len(full_ds)), np.zeros(len(full_ds)), groups)):
        if MAX_FOLDS and fold>=MAX_FOLDS: break
        train_ld=DataLoader(Subset(full_ds,tr_idx),batch_size=BATCH_SIZE,shuffle=True)
        val_ld=DataLoader(Subset(full_ds,val_idx),batch_size=BATCH_SIZE,shuffle=False)
        cls_w = compute_class_weight('balanced', classes=np.array([0,1]), y=y_t[tr_idx].numpy())
        loss_fn = nn.CrossEntropyLoss(torch.tensor(cls_w, dtype=torch.float32).to(DEVICE))
        _,_,C,T=X_t.shape; model=EEGNet(num_classes=2,num_electrodes=C,chunk_size=T,dropout=0.5).to(DEVICE)
        opt=optim.Adam(model.parameters(),lr=LEARNING_RATE); sched=ReduceLROnPlateau(opt,'min',patience=5,factor=0.5)
        best_v=float('inf'); best_a=0; pat=0
        for ep in range(1,NUM_EPOCHS+1):
            tr_l=train_epoch(model,train_ld,opt,loss_fn); val_l,val_a=eval_epoch(model,val_ld,loss_fn); sched.step(val_l)
            cw.writerow([fold+1,ep,f"{tr_l:.3f}",f"{val_l:.3f}",f"{val_a:.2f}"])
            if val_l<best_v:
                best_v, best_a, pat = val_l, val_a, 0
            else: pat+=1
            if pat>=EARLY_STOPPING_PATIENCE: break
        accs.append(best_a); summary['folds'].append(best_a)
        y_true=y_t[val_idx].numpy(); y_pred=model(X_t[val_idx].to(DEVICE)).argmax(1).cpu().numpy(); cm=confusion_matrix(y_true,y_pred,labels=[0,1]); cm_total+=cm
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot_pct = np.round(cm_pct, 1)
        plt.figure(figsize=(4,4))
        ax = sns.heatmap(cm_pct, annot=annot_pct, fmt='.1f', cmap='Blues', xticklabels=['other','inc1'], yticklabels=['other','inc1'])
        plt.xticks(rotation=0); plt.yticks(rotation=0)
        # outline diagonal and max-percentage cells
        for i in range(2):
            ax.add_patch(Rectangle((i,i),1,1,fill=False,edgecolor='black',lw=1.5))
            j = int(np.argmax(cm_pct[i]))
            ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='red',lw=2.5))
        plt.tight_layout()
        plt.savefig(RUN_DIR / f'fold{fold+1}_confusion_{SCRIPT_NAME}.png')
        plt.close()
    # Overall confusion and report
    summary["mean_acc"] = float(np.mean(accs))
    summary["std_acc"] = float(np.std(accs))
    chance = 100/num_classes
    recalls = np.divide(cm_total.diagonal(), cm_total.sum(axis=1), out=np.zeros(num_classes), where=cm_total.sum(axis=1)!=0)
    best_idx = int(np.argmax(recalls)); worst_idx = int(np.argmin(recalls))
    summary.update({
        "chance_level_pct": chance,
        "confusion_total": cm_total.tolist(),
        "per_digit_recall": [float(r) for r in recalls],
        "best_digit": str(best_idx),
        "worst_digit": str(worst_idx)
    })

    # Persist the enriched summary
    summ_path = RUN_DIR / f"summary_{SCRIPT_NAME}.json"
    with open(summ_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    # Reload to guarantee that the txt report is generated strictly from persisted JSON data
    with open(summ_path) as fp:
        summary_loaded = json.load(fp)

    # Overall confusion heatmap
    cm_total_pct = cm_total/cm_total.sum(axis=1,keepdims=True)*100
    annot_tot = np.round(cm_total_pct,1)
    plt.figure(figsize=(4,4))
    ax = sns.heatmap(cm_total_pct, annot=annot_tot, fmt='.1f', cmap='Blues', xticklabels=summary_loaded['classes'], yticklabels=summary_loaded['classes'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Overall Confusion (24 folds)')
    plt.xticks(rotation=0); plt.yticks(rotation=0)
    for i in range(num_classes):
        ax.add_patch(Rectangle((i,i),1,1,fill=False,edgecolor='black',lw=1.5))
        j=int(np.argmax(cm_total_pct[i]))
        ax.add_patch(Rectangle((j,i),1,1,fill=False,edgecolor='red',lw=2.5))
    plt.tight_layout(); plt.savefig(RUN_DIR / f'overall_confusion_{SCRIPT_NAME}.png'); plt.close()

    # Textual report from loaded summary (guarantees consistency)
    report_lines = [
        f"{SCRIPT_NAME} report (raw z-score)",
        f"Run ID        : {summary_loaded['run_id']}",
        "",
        "--- Hyper-parameters ---",
        f"  Batch size          : {summary_loaded['hyper']['batch_size']}",
        f"  Learning rate       : {summary_loaded['hyper']['lr']}",
        f"  Max epochs          : {summary_loaded['hyper']['epochs']}",
        f"  Early-stop patience : {summary_loaded['hyper']['early_stop']}",
        f"  Noise std           : {summary_loaded['hyper']['noise_std']}",
        f"  Channel dropout P   : {summary_loaded['hyper']['channel_dropout_p']}",
        "",
        "--- Results ---",
        f"Mean LOSO accuracy : {summary_loaded['mean_acc']:.2f}% (±{summary_loaded['std_acc']:.2f}%)",
        f"Chance level      : {summary_loaded['chance_level_pct']:.2f}% (binary)",
        "",
        "Per-class recall (TPR):"
    ]
    for i,r in enumerate(summary_loaded['per_digit_recall']):
        report_lines.append(f"  {summary_loaded['classes'][i]} : {r*100:.1f}%   | TP={summary_loaded['confusion_total'][i][i]} / {np.sum(summary_loaded['confusion_total'][i])}")
    report_lines.extend([
        "",
        f"Best recalled class : {summary_loaded['best_digit']}",
        f"Worst recalled class: {summary_loaded['worst_digit']}"
    ])
    with open(RUN_DIR / f"report_{SCRIPT_NAME}.txt", "w") as rp:
        rp.write("\n".join(report_lines))
    print(f"Report written to {RUN_DIR / f'report_{SCRIPT_NAME}.txt'}")
    csv_f.close()

    # Append to expandable index CSV
    append_run_index(summary_loaded, SCRIPT_NAME) 