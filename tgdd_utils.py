# tgdd_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _unwrap_model(net: nn.Module) -> nn.Module:
    """
    Return the underlying model if DataParallel/Distributed, else net itself.
    """
    return net.module if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else net

@torch.no_grad()
def load_snapshot_params_into_net_(net: nn.Module, snapshot_params_list):
    """
    Load a list of parameter tensors (in the order of net.parameters())
    into the network. Raises if shapes mismatch.
    """
    base = _unwrap_model(net)
    params = list(base.parameters())
    if len(params) != len(snapshot_params_list):
        raise ValueError(f"Snapshot length {len(snapshot_params_list)} != net params {len(params)}")
    for p, p_src in zip(params, snapshot_params_list):
        if p.shape != p_src.shape:
            raise ValueError(f"Shape mismatch: net param {p.shape} vs snapshot {p_src.shape}")
        p.copy_(p_src.to(p.device))

def find_last_linear(net: nn.Module):
    """
    Find the last nn.Linear layer in the network (if any).
    """
    base = _unwrap_model(net)
    last = None
    for m in base.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last

class PenultimateHook:
    """
    Hook to capture the input to a module. We use it on the final linear to
    get penultimate features.
    """
    def __init__(self, module: nn.Module):
        self.feat = None
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, mod, inp, out):
        self.feat = inp[0]

    def close(self):
        self._handle.remove()

def forward_logits_and_feats(net: nn.Module, x: torch.Tensor, use_penultimate: bool):
    """
    Run net(x) and return (logits, features). If use_penultimate is True,
    attempt to return the input to the last Linear as features; else return logits as features.
    """
    if not use_penultimate:
        logits = net(x)
        return logits, logits

    last_fc = find_last_linear(net)
    if last_fc is None:
        logits = net(x)
        return logits, logits

    hook = PenultimateHook(last_fc)
    logits = net(x)
    feats = hook.feat
    hook.close()
    if feats is None:
        feats = logits
    return logits, feats

def classwise_mean(z: torch.Tensor, y: torch.Tensor, C: int) -> torch.Tensor:
    """
    Compute per-class mean of feature vectors.
    z: [B, D] tensor of features, y: [B] tensor of class labels (0..C-1).
    Returns a [C, D] tensor of means.
    """
    B, D = z.shape
    sums = torch.zeros(C, D, device=z.device, dtype=z.dtype)
    cnts = torch.zeros(C, device=z.device, dtype=z.dtype)
    sums.index_add_(0, y, z)
    cnts.index_add_(0, y, torch.ones_like(y, dtype=z.dtype))
    means = sums / cnts.clamp_min(1.0).unsqueeze(1)
    return means

# Optional: variance matching or repulsion terms could be added here.
# For example:
# def classwise_var(z: torch.Tensor, y: torch.Tensor, C: int) -> torch.Tensor:
#     ...

