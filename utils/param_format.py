"""Formatting utilities for hyper-parameter annotations in plots."""
from pathlib import Path
from typing import Dict, List, Any
import yaml

def build_plot_lines(space: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> List[str]:
    """Return list of pretty strings with bars/choices for plotting.

    Parameters
    ----------
    space
        Optuna-style search-space dict (method, low/high or choices).
    cfg
        Resolved hyper-parameter dict containing the actual sampled values.
    """
    lines: List[str] = []
    for name, spec in space.items():
        if name not in cfg:
            continue
        val = cfg[name]
        m = spec["method"]
        if m in {"uniform", "log_uniform"}:
            lo, hi = float(spec["low"]), float(spec["high"])
            try:
                p = max(0.0, min(1.0, (val - lo) / (hi - lo)))
                pos = int(p * 10)
            except Exception:
                pos = 0
            bar = "|" + "-" * pos + "●" + "-" * (10 - pos) + "|"
            lines.append(f"{name}: {lo:.1e} {bar} {hi:.1e}\nval: {val:.1e}")
        elif m == "categorical":
            choices = spec["choices"]
            ch_str = [str(c) if c != val else f"[{c}]" for c in choices]
            lines.append(f"{name}: " + " | ".join(ch_str))
    return lines

def load_space_yaml(path: Path | str | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {} 