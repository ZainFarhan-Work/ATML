"""From-scratch LoRA and OPLoRA adapters (FP32 adapters on a frozen FP16 backbone).

OPLoRA is implemented with effective factors; no dense projection matrices are formed:
    B~ = B - U_k (U_k^T B),   A~ = A - (A V_k) V_k^T,   dW = (alpha/r) B~ A~.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterLinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, init_seed: int,
                 Uk: torch.Tensor | None = None, Vk: torch.Tensor | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        g = torch.Generator().manual_seed(init_seed)
        bound = 1.0 / math.sqrt(d_in)  # == kaiming_uniform_(a=sqrt(5)) used by LoRA/PEFT
        A = torch.empty(r, d_in, dtype=torch.float32).uniform_(-bound, bound, generator=g)
        dev = base.weight.device
        self.A = nn.Parameter(A.to(dev))
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32, device=dev))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.protected = Uk is not None
        if self.protected:
            self.register_buffer("Uk", Uk.to(dev, torch.float32), persistent=False)  # [d_out, k]
            self.register_buffer("Vk", Vk.to(dev, torch.float32), persistent=False)  # [d_in, k]

    def forward(self, x):
        y = self.base(x)
        xd = self.drop(x).to(self.A.dtype)
        if self.protected:
            AVk = self.A @ self.Vk                          # [r, k]
            h = F.linear(xd, self.A) - (xd @ self.Vk) @ AVk.T   # A~ x
            out = F.linear(h, self.B)                        # B h
            out = out - (out @ self.Uk) @ self.Uk.T          # U_k-projected component removed
        else:
            out = F.linear(F.linear(xd, self.A), self.B)
        return y + (self.scale * out).to(y.dtype)

    @torch.no_grad()
    def effective_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(B~, A~) in FP32. For plain LoRA these are (B, A)."""
        if not self.protected:
            return self.B.detach().clone(), self.A.detach().clone()
        Bt = self.B - self.Uk @ (self.Uk.T @ self.B)
        At = self.A - (self.A @ self.Vk) @ self.Vk.T
        return Bt, At


def target_linears(model: nn.Module, targets: list[str]) -> dict[str, tuple[nn.Module, str, nn.Linear]]:
    found = {}
    for name, mod in model.named_modules():
        for child_name, child in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and child_name in targets and ".layers." in f".{full}":
                found[full] = (mod, child_name, child)
    return found


@torch.no_grad()
def compute_bases(W0: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading k left/right singular vectors of W0 [d_out, d_in], computed in FP32."""
    U, S, Vh = torch.linalg.svd(W0.float(), full_matrices=False)
    return U[:, :k].contiguous(), Vh[:k].T.contiguous()


def inject_adapters(model: nn.Module, targets: list[str], r: int, alpha: float, dropout: float,
                    seed: int, protect_k: int | None = None, bases: dict | None = None) -> tuple[dict, dict]:
    """Replace target Linear layers by AdapterLinear. Returns (adapter modules, preprocessing info)."""
    for p in model.parameters():
        p.requires_grad_(False)
    info = {"svd_seconds": 0.0, "basis_bytes": 0}
    adapters = {}
    for idx, (name, (parent, child_name, lin)) in enumerate(sorted(target_linears(model, targets).items())):
        Uk = Vk = None
        if protect_k:
            if bases is not None and name in bases:
                Uk, Vk = bases[name]
            else:
                t0 = time.perf_counter()
                Uk, Vk = compute_bases(lin.weight.data, protect_k)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                info["svd_seconds"] += time.perf_counter() - t0
                if bases is not None:
                    bases[name] = (Uk.cpu(), Vk.cpu())
            info["basis_bytes"] += (Uk.numel() + Vk.numel()) * 4
        mod = AdapterLinear(lin, r, alpha, dropout, init_seed=seed * 100003 + idx, Uk=Uk, Vk=Vk)
        setattr(parent, child_name, mod)
        adapters[name] = mod
    return adapters, info


def adapter_params(adapters: dict) -> list[nn.Parameter]:
    return [p for m in adapters.values() for p in (m.A, m.B)]


@torch.no_grad()
def export_adapters(adapters: dict, path: str, meta: dict) -> dict:
    """Save effective (projected, for OPLoRA) factors for inference/merging. Returns overhead info."""
    t0 = time.perf_counter()
    state = {}
    for name, m in adapters.items():
        Bt, At = m.effective_factors()
        state[name] = {"A": At.cpu(), "B": Bt.cpu()}
    torch.save({"meta": meta, "factors": state}, path)
    return {"export_seconds": time.perf_counter() - t0}


@torch.no_grad()
def merge_adapters(model: nn.Module, payload: dict) -> nn.Module:
    """Merge exported factors into a (fresh, FP16) backbone in place: W = W0 + alpha/r * B~ A~."""
    scale = payload["meta"]["alpha"] / payload["meta"]["r"]
    lins = dict(model.named_modules())
    for name, f in payload["factors"].items():
        lin = lins[name]
        dW = scale * (f["B"].to(lin.weight.device, torch.float32) @ f["A"].to(lin.weight.device, torch.float32))
        lin.weight.copy_((lin.weight.float() + dW).to(lin.weight.dtype))
    return model


def attach_exported(model: nn.Module, payload: dict) -> nn.Module:
    """Unmerged inference model: plain LoRA modules loaded with exported factors."""
    meta = payload["meta"]
    targets = list({n.split(".")[-1] for n in payload["factors"]})
    adapters, _ = inject_adapters(model, targets, meta["r"], meta["alpha"], 0.0, seed=0)
    for name, m in adapters.items():
        m.A.data.copy_(payload["factors"][name]["A"])
        m.B.data.copy_(payload["factors"][name]["B"])
    return model
