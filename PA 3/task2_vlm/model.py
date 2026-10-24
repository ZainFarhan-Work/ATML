"""SigLIP -> MLP connector -> Qwen3 VLM operating on cached SigLIP patch features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from task1_llm.lora import adapter_params, inject_adapters

SENTINEL = "<<<QUESTION>>>"
IM_END = "<|im_end|>"


class Connector(nn.Module):
    """Linear(768,1024) -> GELU -> Linear(1024,1024), plus a fixed calibration scale."""

    def __init__(self, d_vis: int = 768, d_llm: int = 1024, pool_2x2: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_vis, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        self.register_buffer("out_scale", torch.ones(()))
        self.pool_2x2 = pool_2x2

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # [B, 196, 768] -> [B, N, 1024]
        if self.pool_2x2:  # average each 2x2 block of the 14x14 grid -> 7x7 = 49 tokens
            B, T, D = feats.shape
            g = int(T ** 0.5)
            x = feats.float().view(B, g, g, D).permute(0, 3, 1, 2)
            feats = F.avg_pool2d(x, 2).flatten(2).transpose(1, 2)
        return self.net(feats) * self.out_scale


class VLM(nn.Module):
    def __init__(self, cfg: dict, compress: bool = False, device: str = "cuda"):
        super().__init__()
        self.cfg, self.device = cfg, device
        tcfg = cfg["train"]
        self.tok = AutoTokenizer.from_pretrained(cfg["llm_id"], revision=cfg["llm_revision"])
        self.llm = AutoModelForCausalLM.from_pretrained(cfg["llm_id"], revision=cfg["llm_revision"],
                                                        torch_dtype=torch.float16).to(device)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.config.use_cache = False
        if tcfg["grad_checkpointing"]:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.connector = Connector(cfg["features"]["vision_dim"], cfg["features"]["llm_dim"], pool_2x2=compress).to(device)
        self.adapters: dict = {}
        tmpl = self.tok.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        pre, post = tmpl.split(SENTINEL)
        self.pre_ids = self.tok(pre, add_special_tokens=False)["input_ids"]
        self.post_ids = self.tok(post, add_special_tokens=False)["input_ids"]
        self.eos_id = self.tok.convert_tokens_to_ids(IM_END)
        self.caps = {"question": tcfg["max_question_tokens"], "caption": tcfg["max_caption_tokens"],
                     "short": tcfg["max_answer_tokens"]}

    # ---- adapters -----------------------------------------------------------------------------------------
    def enable_lora(self, seed: int) -> list[nn.Parameter]:
        lc = self.cfg["train"]["lora"]
        self.adapters, _ = inject_adapters(self.llm, lc["targets"], lc["r"], lc["alpha"], lc["dropout"], seed=seed)
        return adapter_params(self.adapters)

    def lora_state(self) -> dict:
        return {n: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()} for n, m in self.adapters.items()}

    def load_lora_state(self, state: dict) -> None:
        for n, m in self.adapters.items():
            m.A.data.copy_(state[n]["A"])
            m.B.data.copy_(state[n]["B"])

    # ---- visual interface ---------------------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_scale(self, feats: torch.Tensor, texts: list[str]) -> dict:
        """Rule: scale the connector so its mean per-token L2 norm equals the mean text-embedding norm.

        Statistics come from training data only (training image features, training text strings).
        """
        emb = self.llm.get_input_embeddings()
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        flat = torch.tensor([i for s in ids for i in s], device=self.device)
        text_norm = emb(flat).float().norm(dim=-1).mean().item()
        self.connector.out_scale.fill_(1.0)
        vis = self.connector(feats.to(self.device).float())
        vis_norm = vis.float().norm(dim=-1).mean().item()
        scale = text_norm / vis_norm
        self.connector.out_scale.fill_(scale)
        return {"rule": "match mean per-token L2 norm of connector output to mean text-embedding norm (training data only)",
                "text_embed_norm": text_norm, "connector_norm_before": vis_norm, "scale": scale}

    def visual_tokens(self, feats: torch.Tensor) -> torch.Tensor:
        return self.connector(feats.to(self.device).float()).to(torch.float16)

    # ---- batch construction -------------------------------------------------------------------------------
    def build_batch(self, examples: list[dict], feats: torch.Tensor | None, side: str = "right",
                    with_target: bool = True) -> dict:
        """Assemble inputs_embeds / mask / position ids / labels. feats=None builds the no-image prefix."""
        emb = self.llm.get_input_embeddings()
        vis = self.visual_tokens(feats) if feats is not None else None
        seqs, labs, labs_noeos = [], [], []
        for i, ex in enumerate(examples):
            q = self.tok("\n" + ex["question"], add_special_tokens=False)["input_ids"][: self.caps["question"]]
            tgt = self.tok(ex["target"], add_special_tokens=False)["input_ids"][: self.caps[ex["max_target"]]]
            tgt = tgt + [self.eos_id]
            parts = [emb(torch.tensor(self.pre_ids, device=self.device))]
            if vis is not None:
                parts.append(vis[i])
            parts.append(emb(torch.tensor(q + self.post_ids, device=self.device)))
            n_prefix = sum(p.shape[0] for p in parts)
            lab = torch.full((n_prefix,), -100, dtype=torch.long, device=self.device)
            if with_target:
                parts.append(emb(torch.tensor(tgt, device=self.device)))
                t = torch.tensor(tgt, device=self.device)
                lab = torch.cat([lab, t])
                lab2 = torch.cat([torch.full((n_prefix,), -100, device=self.device), t.clone()])
                lab2[-1] = -100
                labs_noeos.append(lab2)
            seqs.append(torch.cat(parts, 0))
            labs.append(lab)
        L = max(s.shape[0] for s in seqs)
        H = seqs[0].shape[1]
        B = len(seqs)
        x = torch.zeros(B, L, H, dtype=torch.float16, device=self.device)
        mask = torch.zeros(B, L, dtype=torch.long, device=self.device)
        lab = torch.full((B, L), -100, dtype=torch.long, device=self.device)
        lab2 = torch.full((B, L), -100, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            n = s.shape[0]
            sl = slice(0, n) if side == "right" else slice(L - n, L)
            x[i, sl] = s.to(torch.float16)
            mask[i, sl] = 1
            lab[i, sl] = labs[i]
            if with_target:
                lab2[i, sl] = labs_noeos[i]
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        return {"inputs_embeds": x, "attention_mask": mask, "position_ids": pos, "labels": lab, "labels_noeos": lab2}

    # ---- losses / generation ------------------------------------------------------------------------------
    def per_example_nll(self, batch: dict, labels_key: str = "labels") -> torch.Tensor:
        """Mean token NLL within each example (supervised tokens only). Shape [B]."""
        hidden = self.llm.model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                                position_ids=batch["position_ids"]).last_hidden_state
        lab = batch[labels_key]
        m = lab[:, 1:] != -100
        h = hidden[:, :-1][m]
        logits = self.llm.lm_head(h).float()
        ce = F.cross_entropy(logits, lab[:, 1:][m], reduction="none")
        rows = torch.nonzero(m)[:, 0]
        B = lab.shape[0]
        tot = torch.zeros(B, device=lab.device).index_add(0, rows, ce)
        cnt = torch.zeros(B, device=lab.device).index_add(0, rows, torch.ones_like(ce)).clamp_min(1)
        return tot / cnt

    @torch.no_grad()
    def generate(self, batch: dict, max_new_tokens: int) -> list[str]:
        self.llm.config.use_cache = True
        out = self.llm.generate(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                                max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                                pad_token_id=self.tok.pad_token_id)
        self.llm.config.use_cache = False
        return [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]

    def trainable_summary(self) -> dict:
        conn = sum(p.numel() for p in self.connector.parameters())
        lora = sum(p.numel() for m in self.adapters.values() for p in (m.A, m.B))
        return {"connector": conn, "lora": lora, "total": conn + lora}
