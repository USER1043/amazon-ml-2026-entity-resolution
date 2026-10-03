"""Bi-encoder: fine-tune a small sentence encoder on (S2/S3 text, S1 text) pairs with in-batch
negatives, then encode records. Default base model: all-MiniLM-L6-v2 (Apache-2.0, 22M params)."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from . import config as C


class Enc(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.m = AutoModel.from_pretrained(path)

    def forward(self, ids, mask):
        h = self.m(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(e.float(), dim=-1)


def _dev():
    return "cuda" if (C.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"


def _wrap(enc):
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(enc)
    return enc


def train(q_texts, s_texts, out_dir, base=None):
    dev = _dev()
    base = base or C.BASE_MODEL
    tok = AutoTokenizer.from_pretrained(base)
    enc = Enc(base).to(dev)
    model = _wrap(enc)
    model.train()
    B = C.FT_BATCH
    n = len(q_texts)
    spe = max(1, n // B)
    total = spe * C.FT_EPOCHS
    opt = torch.optim.AdamW(enc.parameters(), lr=C.FT_LR, weight_decay=0.01)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(0)
    t0, step = time.time(), 0
    for ep in range(C.FT_EPOCHS):
        perm = rng.permutation(n)
        for i in range(spe):
            b = perm[i * B:(i + 1) * B]
            tq = tok([q_texts[j] for j in b], padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
            ts = tok([s_texts[j] for j in b], padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                eq = model(tq["input_ids"].to(dev), tq["attention_mask"].to(dev))
                es = model(ts["input_ids"].to(dev), ts["attention_mask"].to(dev))
            logits = (eq @ es.T) * 20.0
            lbl = torch.arange(len(b), device=logits.device)
            loss = (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0 or step == total:
                acc = (logits.argmax(1) == lbl).float().mean().item()
                print(f"  ft step {step}/{total} loss {loss.item():.4f} in-batch acc {acc:.3f} {time.time()-t0:.0f}s", flush=True)
    enc.m.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del model, enc
    if dev == "cuda":
        torch.cuda.empty_cache()


class Encoder:
    def __init__(self, path):
        self.dev = _dev()
        self.tok = AutoTokenizer.from_pretrained(path)
        enc = Enc(path).to(self.dev).eval()
        if self.dev == "cuda":
            enc = enc.half()
        self.model = _wrap(enc)
        self.dim = enc.m.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts, batch=None, log=True):
        batch = batch or C.ENC_BATCH
        n = len(texts)
        out = np.empty((n, self.dim), dtype=np.float16)
        order = np.argsort(np.fromiter((len(t) for t in texts), dtype=np.int32, count=n), kind="stable")
        t0 = time.time()
        for k, i in enumerate(range(0, n, batch)):
            idx = order[i:i + batch]
            tk = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
            e = self.model(tk["input_ids"].to(self.dev), tk["attention_mask"].to(self.dev))
            out[idx] = e.float().cpu().numpy().astype(np.float16)
            if log and k % 500 == 0 and k:
                print(f"    encoded {i:,}/{n:,} {time.time()-t0:.0f}s", flush=True)
        return out
