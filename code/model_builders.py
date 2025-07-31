"""
Model and augmentation builders for the unified training framework.
"""
from __future__ import annotations
from typing import Dict, Any

import torch
import torch.nn as nn

# Guarded model imports
try:
    from torcheeg.models import EEGNet
except ImportError:
    EEGNet = None

try:
    from .models.cwa_transformer import CwaTransformer
except ImportError:
    CwaTransformer = None

try:
    from .models.multi_scale_cnn import MultiScaleCNN
except ImportError:
    MultiScaleCNN = None

try:
    from .models.eegnet_se import EEGNetWithSE
except ImportError:
    EEGNetWithSE = None

try:
    from .models.dual_stream_cnn import DualStreamCNN
except ImportError:
    DualStreamCNN = None

try:
    import timm
except ImportError:
    timm = None


# --- Raw EEG Model Builders ---

def build_eegnet(cfg: Dict[str, Any], num_classes: int, C: int, T: int) -> nn.Module:
    if EEGNet is None:
        raise ImportError("EEGNet not found. Please install torcheeg.")
    return EEGNet(
        num_classes=num_classes,
        num_electrodes=C,
        chunk_size=T,
        dropout=cfg.get('dropout', 0.5),
    )

def build_eegnet_se(cfg: Dict[str, Any], num_classes: int, C: int, T: int) -> nn.Module:
    if EEGNetWithSE is None:
        raise ImportError("EEGNetWithSE not found. Check code/models/eegnet_se.py")
    return EEGNetWithSE(
        num_classes=num_classes,
        num_electrodes=C,
        chunk_size=T,
        se_reduction_ratio=cfg.get('se_reduction_ratio', 4),
        F1=cfg.get('F1', 8), D=cfg.get('D', 2), F2=cfg.get('F2', 16),
        kernel_1=cfg.get('kernel_1', 64), kernel_2=cfg.get('kernel_2', 16),
        dropout=cfg.get('dropout', 0.5)
    )

def build_multi_scale_cnn(cfg: Dict[str, Any], num_classes: int, C: int, T: int) -> nn.Module:
    if MultiScaleCNN is None:
        raise ImportError("MultiScaleCNN not found. Check code/models/multi_scale_cnn.py")
    return MultiScaleCNN(
        chunk_size=T, num_electrodes=C, num_classes=num_classes,
        kernel_sizes=cfg.get('kernel_sizes', [7, 11, 15, 19]),
        F1=cfg.get('F1', 8), F2=cfg.get('F2', 16), D=cfg.get('D', 2),
        kernel_2=cfg.get('kernel_2', 16), dropout=cfg.get('dropout', 0.5)
    )

def build_cwat(cfg: Dict[str, Any], num_classes: int, C: int, T: int) -> nn.Module:
    if CwaTransformer is None:
        raise ImportError("CwaTransformer not found.")
    return CwaTransformer(
        in_channels=C, num_classes=num_classes,
        enc_kernel_size=cfg.get("enc_kernel", 7),
        enc_latent_dim=cfg.get("latent_dim", 64),
        num_heads=cfg.get("n_heads", 4),
        num_layers=cfg.get("depth", 4),
        dropout=cfg.get("dropout", 0.1),
        enc_stride=cfg.get("enc_stride", 2),
    )

def build_dual_stream_cnn(cfg: Dict[str, Any], num_classes: int, C: int, T: int) -> nn.Module:
    if DualStreamCNN is None:
        raise ImportError("DualStreamCNN not found. Check code/models/dual_stream_cnn.py")
    
    # Extract configs for each part of the model
    ts_config = cfg.get('ts_stream_config', {})
    ts_config.update({'chunk_size': T, 'num_electrodes': C})
    
    spec_config = cfg.get('spec_stream_config', {})
    fusion_config = cfg.get('fusion_head_config', {})

    return DualStreamCNN(
        ts_config=ts_config,
        spec_config=spec_config,
        fusion_config=fusion_config,
        num_classes=num_classes,
        num_channels=C,
    )

RAW_EEG_MODELS = {
    "eegnet": build_eegnet,
    "eegnet_se": build_eegnet_se,
    "multi_scale_cnn": build_multi_scale_cnn,
    "cwat": build_cwat,
    "dual_stream": build_dual_stream_cnn,
}

# --- Spectrogram Model Builders ---

def build_timm_model(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    if timm is None:
        raise ImportError("timm is not installed. Please run: pip install timm")
    return timm.create_model(
        cfg.get("model_name", "resnet18"),
        pretrained=True,
        num_classes=num_classes,
        in_chans=1,
    )

# --- Augmentation Builders ---

# These are the original, self-contained augmentation classes from the project.
# They are being restored here to remove the dependency on a specific torcheeg
# version, which was causing the TypeError.
class ComposeT:
    def __init__(self, ts): self.ts = ts
    def __call__(self, x):
        for t in self.ts: x = t(x)
        return x
class RTShift:
    def __init__(self, p, mn, mx): self.p, self.mn, self.mx = p, mn, mx
    def __call__(self,x):
        if self.mx and torch.rand(1,device=x.device) < self.p:
            sh = torch.randint(self.mn, self.mx+1,(1,),device=x.device).item()
            if torch.rand(1,device=x.device)<0.5: sh=-sh
            return torch.roll(x,shifts=sh,dims=-1)
        return x
class RScale:
    def __init__(self, p: float, lo: float, hi: float):
        self.p = p
        self.lo = lo
        self.hi = hi
    def __call__(self,x):
        if torch.rand(1,device=x.device) < self.p:
            return x*torch.empty(1,device=x.device).uniform_(self.lo,self.hi)
        return x
class RNoise:
    def __init__(self,p,std): self.p,self.std=p,std
    def __call__(self,x):
        if self.std>0 and torch.rand(1,device=x.device)<self.p:
            return x + self.std*torch.randn_like(x)
        return x
class RTMask:
    def __init__(self,p,frac): self.p,self.frac=p,frac
    def __call__(self,x):
        if self.p==0 or torch.rand(1,device=x.device)>=self.p: return x
        T=x.shape[-1]; m=max(1,int(self.frac*T)); st=torch.randint(0,T-m+1,(1,),device=x.device).item(); x=x.clone(); x[...,st:st+m]=0; return x
class RCMask:
    def __init__(self,p,ratio): self.p,self.r= p, ratio
    def __call__(self,x):
        if self.p==0 or torch.rand(1,device=x.device)>=self.p: return x
        C=x.shape[-2]; k=max(1,int(self.r*C)); idx=torch.randperm(C,device=x.device)[:k]; x=x.clone(); x[...,idx,:]=0; return x


def build_raw_eeg_aug(cfg: Dict[str, Any], T: int):
    return ComposeT([
        RTShift(cfg.get('shift_p', 0.0), max(1,int(cfg.get('shift_min_frac', 0.0)*T)), int(cfg.get('shift_max_frac', 0.0)*T)),
        RScale(cfg.get('scale_p', 0.0), cfg.get('scale_min', 1.0), cfg.get('scale_max', 1.0)),
        RNoise(cfg.get('noise_p', 0.0), cfg.get('noise_std', 0.0)),
        RTMask(cfg.get('time_mask_p', 0.0), cfg.get('time_mask_frac', 0.0)),
        RCMask(cfg.get('chan_mask_p', 0.0), cfg.get('chan_mask_ratio', 0.0))
    ])

def build_spectrogram_aug(cfg: Dict[str, Any], T: int):
    # This is a simplified version of SpecAugment
    # In a real scenario, you might use a library or a more robust implementation
    return ComposeT([
        RTMask(p=cfg.get('freq_mask_p', 0.0), frac=cfg.get('freq_mask_frac', 0.0)),
        RTMask(p=cfg.get('time_mask_p', 0.0), frac=cfg.get('time_mask_frac', 0.0)),
    ])

# Input adapter for models that expect (B, C, T) instead of (B, 1, C, T)
def squeeze_input_adapter(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(1)
