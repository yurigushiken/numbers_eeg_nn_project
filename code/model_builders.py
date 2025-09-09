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

# Optional Braindecode EEGNeX
try:
    from braindecode.models import EEGNeX as BD_EEGNeX
except Exception:
    BD_EEGNeX = None

# New imports for pre-trained models
try:
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    resnet18 = None
    ResNet18_Weights = None
try:
    from .models.channel_gate import ChannelGatedModel
except ImportError:
    ChannelGatedModel = None
try:
    from .models.temporal_gate import TemporalGatedModel
except ImportError:
    TemporalGatedModel = None


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

def _build_custom_spec_stream(spec_config: dict, num_channels: int) -> tuple[nn.Module, int]:
    """Builds the original, custom 2D CNN for spectrograms."""
    spec_channels = spec_config.get('channels', [32, 64])
    model = nn.Sequential(
        nn.Conv2d(num_channels, spec_channels[0], kernel_size=3, padding=1),
        nn.GroupNorm(1, spec_channels[0]),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(spec_channels[0], spec_channels[1], kernel_size=3, padding=1),
        nn.GroupNorm(1, spec_channels[1]),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    feature_dim = spec_channels[1]
    return model, feature_dim

def _build_pretrained_spec_stream(arch: str, use_pretrained: bool, num_channels: int) -> tuple[nn.Module, int]:
    """Builds a spectrogram stream from a pre-trained torchvision model."""
    if arch == "resnet18":
        if resnet18 is None:
            raise ImportError("torchvision is not installed or doesn't have resnet18.")
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = resnet18(weights=weights)
        feature_dim = model.fc.in_features
        # Replace final layer to make it a feature extractor
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported spectrogram architecture: {arch}")

    # Create the projection layer to handle N-channel input
    projection_layer = nn.Conv2d(num_channels, 3, kernel_size=1, stride=1, padding=0)

    # Combine projection and the pre-trained model
    full_stream = nn.Sequential(
        projection_layer,
        model
    )
    return full_stream, feature_dim


def build_dual_stream_cnn(cfg: Dict[str, Any], num_classes: int, C: int, T: int) -> nn.Module:
    if DualStreamCNN is None:
        raise ImportError("DualStreamCNN not found. Check code/models/dual_stream_cnn.py")
    
    # Extract configs for each part of the model
    ts_config = cfg.get('ts_stream_config', {})
    ts_config.update({'chunk_size': T, 'num_electrodes': C})
    
    spec_config = cfg.get('spec_stream_config', {})
    fusion_config = cfg.get('fusion_head_config', {})

    # --- Build the Spectrogram Stream ---
    spec_arch = cfg.get("spec_arch", "custom")
    if spec_arch == "custom":
        spec_stream_model, spec_feature_dim = _build_custom_spec_stream(spec_config, C)
    else:
        use_pretrained = spec_config.get("pretrained", True)
        spec_stream_model, spec_feature_dim = _build_pretrained_spec_stream(spec_arch, use_pretrained, C)

    return DualStreamCNN(
        ts_config=ts_config,
        spec_stream_model=spec_stream_model,
        spec_feature_dim=spec_feature_dim,
        fusion_config=fusion_config,
        num_classes=num_classes,
    )

def build_eegnex(cfg: Dict[str, Any], num_classes: int, C: int, T: int) -> nn.Module:
    """Wrapper around braindecode.models.EEGNeX.

    Exposes key hyper-parameters via cfg with sensible defaults.
    """
    if BD_EEGNeX is None:
        raise ImportError("Braindecode EEGNeX not available. Install braindecode>=1.1.0.")

    activation = nn.ELU
    depth_multiplier = int(cfg.get("depth_multiplier", 2))
    filter_1 = int(cfg.get("filter_1", 8))
    filter_2 = int(cfg.get("filter_2", 32))
    drop_prob = float(cfg.get("drop_prob", 0.5))

    # EEGNeX expects integers for kernel lengths, dilations and pooling sizes.
    kernel_block_1_2 = int(cfg.get("kernel_block_1_2", 32))
    dilation_block_4 = int(cfg.get("dilation_block_4", 2))
    dilation_block_5 = int(cfg.get("dilation_block_5", 4))

    max_norm_conv = float(cfg.get("max_norm_conv", 1.0))
    max_norm_linear = float(cfg.get("max_norm_linear", 0.25))

    model = BD_EEGNeX(
        n_chans=C,
        n_outputs=num_classes,
        n_times=T,
        activation=activation,
        depth_multiplier=depth_multiplier,
        filter_1=filter_1,
        filter_2=filter_2,
        drop_prob=drop_prob,
        kernel_block_1_2=kernel_block_1_2,
        dilation_block_4=dilation_block_4,
        dilation_block_5=dilation_block_5,
        max_norm_conv=max_norm_conv,
        max_norm_linear=max_norm_linear,
    )
    # Optional Temporal and Channel Gate wrappers
    if bool(cfg.get("temporal_gate", False)):
        if TemporalGatedModel is None:
            raise ImportError("TemporalGatedModel not available; check code/models/temporal_gate.py")
        tg_init = float(cfg.get("time_gate_init", 1.0))
        model = TemporalGatedModel(model, num_timepoints=T, gate_init=tg_init)

    if bool(cfg.get("channel_gate", False)):
        if ChannelGatedModel is None:
            raise ImportError("ChannelGatedModel not available; check code/models/channel_gate.py")
        gate_init = float(cfg.get("gate_init", 1.0))
        model = ChannelGatedModel(model, num_channels=C, gate_init=gate_init)
    return model

RAW_EEG_MODELS = {
    "eegnet": build_eegnet,
    "eegnet_se": build_eegnet_se,
    "multi_scale_cnn": build_multi_scale_cnn,
    "cwat": build_cwat,
    "dual_stream": build_dual_stream_cnn,
    "eegnex": build_eegnex,
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
    return ComposeT([
        RTMask(p=cfg.get('freq_mask_p', 0.0), frac=cfg.get('freq_mask_frac', 0.0)),
        RTMask(p=cfg.get('time_mask_p', 0.0), frac=cfg.get('time_mask_frac', 0.0)),
    ])

# Input adapter for models that expect (B, C, T) instead of (B, 1, C, T)
def squeeze_input_adapter(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(1)
