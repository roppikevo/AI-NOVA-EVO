"""
Long training of the best NOVA model (hours, not minutes).

Evolution compares cores with short 1000-step runs. Once a core is chosen,
this script trains the deployable model much longer:

    python -m evo.engine.long_train --steps 20000 --max-hours 2.5

  * starts from the current best weights (active_weights.json or the
    baseline checkpoint) - or --from-scratch
  * AdamW, warm-up + cosine decay, gradient clipping, bf16 autocast on GPU
  * evaluation every --eval-every steps on held-out validation data
  * checkpoint every --save-every steps; --resume continues after a stop
  * stops gracefully on --max-hours, early stopping (no improvement for
    --patience evaluations) or the file evo/engine/STOP_TRAINING
  * progress is streamed to evo/learning/long_train_progress.jsonl
  * the best checkpoint becomes the active weights; the evolution baseline
    (best_known, 1000-step protocol) is not touched

Hardware: one process, ~0.5-1 GB VRAM at batch 16, low CPU use.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

STATE = Path("evo/engine/evo_state.json")
OUT_DIR = Path("evo/learning/checkpoints")
PROGRESS = Path("evo/learning/long_train_progress.jsonl")
ACTIVE_WEIGHTS = Path("evo/learning/active_weights.json")
STOP_FILE = Path("evo/engine/STOP_TRAINING")


def load_tokens(path: Path) -> np.ndarray:
    """Packed sequences (one per line) -> int32 array [n, seq_len], cached as .npy."""
    cache = path.with_suffix(".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)
    rows = [np.array(line.split(), dtype=np.int32) for line in path.open(encoding="utf-8") if line.strip()]
    arr = np.stack(rows)
    np.save(cache, arr)
    return arr


def extra_sequences(dataset: Path, dirs: list[Path], seq_len: int = 128) -> np.ndarray:
    """Teacher answers (with <think> if present) and web texts, encoded with the dataset tokenizer."""
    from evo.corpus.build_text_v1 import encode_doc
    from evo.corpus.sources import teacher_jsonl, web_jsonl
    from nova.tokenizer import NovaTokenizer, pack_sequences

    tok = NovaTokenizer.load(dataset / "tokenizer.json")
    docs = []
    for d in dirs:
        if not d.exists():
            continue
        loader = web_jsonl if "web" in d.name else teacher_jsonl
        docs.extend(encode_doc(tok, doc) for doc in loader(d, 50_000_000))
    seqs = list(pack_sequences(docs, seq_len))
    return np.array(seqs, dtype=np.int32) if seqs else np.zeros((0, seq_len), dtype=np.int32)


def row_languages(seqs: np.ndarray, lang_ids: tuple[int, ...] = (4, 5, 6, 7, 8, 9), eos: int = 3,
                  chunk_rows: int = 20000) -> np.ndarray:
    """Language tag id that most tokens of each row belong to (0 = unknown).

    Rows are consecutive pieces of one token stream (`<lang> text <eos>` documents), so the language
    carries over from the row before - same rule as scoreboard.language_of_tokens, but vectorised
    (in chunks, so a big dataset does not need gigabytes of memory)."""
    seqs = np.asarray(seqs)
    out = np.zeros(len(seqs), dtype=np.int64)
    carry = 0  # language still open at the end of the previous chunk
    for start in range(0, len(seqs), chunk_rows):
        part = seqs[start:start + chunk_rows]
        flat = part.reshape(-1)
        is_tag = np.isin(flat, lang_ids)
        after_eos = np.zeros(len(flat), dtype=bool)
        after_eos[1:] = flat[:-1] == eos
        event = is_tag | after_eos
        value = np.where(is_tag, flat, 0)                   # a tag wins over the reset after <eos>
        last = np.maximum.accumulate(np.where(event, np.arange(len(flat), dtype=np.int32), -1))
        lang = np.where(last >= 0, value[np.maximum(last, 0)], carry).reshape(part.shape)
        carry = 0 if flat[-1] == eos else int(lang[-1, -1])
        best = np.zeros(len(part), dtype=np.int64)
        res = np.zeros(len(part), dtype=np.int64)
        for i in lang_ids:
            n = (lang == i).sum(axis=1)
            res = np.where(n > best, i, res)
            best = np.maximum(best, n)
        out[start:start + chunk_rows] = res
    return out


def join_sequences(arr: np.ndarray, mult: int) -> np.ndarray:
    """[n, L] -> [n // mult, L * mult] (neighbouring rows are consecutive text)."""
    n = (len(arr) // mult) * mult
    rows = arr[np.arange(n)] if not isinstance(arr, np.ndarray) else arr[:n]
    return np.ascontiguousarray(rows).reshape(n // mult, arr.shape[1] * mult)


def code_practice_sequences(dataset: Path, seq_len: int = 128) -> np.ndarray:
    """Code-school practice tasks (prompt + reference body), packed like every other document.

    Each task appears in 4 packings with different neighbours/offsets."""
    import random

    from evo.learning.code_tasks import task_bank
    from nova.tokenizer import EOS_ID, NovaTokenizer, pack_sequences

    tok = NovaTokenizer.load(dataset / "tokenizer.json")
    docs = [[tok.lang_id("py")] + tok.encode(t.prompt + t.solution + "\n") + [EOS_ID]
            for t in task_bank() if t.pool == "practice"]
    rows = []
    for k in range(4):
        random.Random(k).shuffle(docs)
        rows.extend(pack_sequences(([EOS_ID] * (k * 7),) + tuple(docs) if k else docs, seq_len))
    return np.array(rows, dtype=np.int32) if rows else np.zeros((0, seq_len), dtype=np.int32)


def lr_at(step: int, total: int, base: float, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def _logits(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def evaluate(model, val: np.ndarray, device, batch_size: int = 32, max_batches: int = 100) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, min(len(val), batch_size * max_batches), batch_size):
        b = torch.from_numpy(val[i:i + batch_size]).long().to(device)
        logits = _logits(model(b[:, :-1]))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), b[:, 1:].reshape(-1), reduction="sum")
        total += float(loss)
        n += b[:, 1:].numel()
    model.train()
    return total / max(n, 1)


def detach_states(states: Any) -> Any:
    """The carried state as plain numbers: the next row starts from it, gradients do not flow back into the last one."""
    if torch.is_tensor(states):
        return states.detach()
    if isinstance(states, (list, tuple)):
        return type(states)(detach_states(s) for s in states)
    return states


@torch.no_grad()
def evaluate_carried(model, rows: np.ndarray, device, streams: int = 32, max_rows: int = 3200) -> float:
    """Loss on consecutive rows of text with the state carried from row to row (how a recurrent core is used).

    `rows` are consecutive 128-token pieces; they are read as `streams` parallel texts. The same tokens count
    as in evaluate() (tokens 2..128 of every row), so the two numbers can be compared directly."""
    model.eval()
    rows = rows[:max_rows]
    length = len(rows) // streams
    total, n, states = 0.0, 0, None
    for k in range(length):
        b = torch.from_numpy(np.ascontiguousarray(rows[np.arange(streams) * length + k])).long().to(device)
        out = model(b, states)
        logits, states = out[0], detach_states(out[1])
        total += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), b[:, 1:].reshape(-1), reduction="sum"))
        n += b[:, 1:].numel()
    model.train()
    return total / max(n, 1)


def _dataset_rows(rng: np.random.Generator, n_train: int, n: int, focus: np.ndarray | None, focus_frac: float) -> np.ndarray:
    """Row numbers for the dataset part of a batch; a share of them comes from the focus rows (a specialist)."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if focus is None or not len(focus) or focus_frac <= 0:
        return rng.integers(0, n_train, size=n)
    nf = int(rng.binomial(n, focus_frac))
    return np.concatenate([focus[rng.integers(0, len(focus), size=nf)], rng.integers(0, n_train, size=n - nf)])


def long_train(
    model: torch.nn.Module,
    train: np.ndarray,
    val: np.ndarray,
    *,
    steps: int,
    out_dir: Path,
    meta: dict,
    batch_size: int = 16,
    lr: float = 3e-4,
    warmup: int = 500,
    weight_decay: float = 0.01,
    eval_every: int = 1000,
    save_every: int = 2000,
    patience: int = 4,
    max_hours: float | None = None,
    resume: dict | None = None,
    compile_model: bool = False,
    carry: int = 0,
    carry_share: float = 0.0,
    seed: int = 1001,
    bulk: np.ndarray | None = None,
    bulk_frac: float = 0.7,
    code: np.ndarray | None = None,
    code_frac: float = 0.0,
    warm_optimizer: dict | None = None,
    boost: np.ndarray | None = None,
    boost_frac: float = 0.15,
    focus: np.ndarray | None = None,
    focus_frac: float = 0.0,
    extra_val: np.ndarray | None = None,
    progress: Path = PROGRESS,
    stop_file: Path = STOP_FILE,
    log: Callable[[str], None] = print,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # the training forward pass may run compiled (many small operations fused into few kernels: the
    # generation-8 core trains almost twice as fast); measuring and saving always use the plain model
    forward = model
    if compile_model and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            forward = torch.compile(model)
        except Exception as exc:
            log(f"compiling is not available here ({type(exc).__name__}) - training without it")

    # which number decides "best": the dataset's validation loss, or - with a second, held-out web
    # validation set - the mean of both (a model that only memorises the small dataset must not win)
    metric = "val" if extra_val is None else "mean(val,web_val)"
    best_parts: dict[str, float] = {}
    start_step, best, bad_evals = 0, float("inf"), 0
    if resume:
        model.load_state_dict(resume["model_state_dict"])
        opt.load_state_dict(resume["optimizer"])
        start_step = resume["step"]
        if resume.get("select_metric", "val") == metric:
            best = resume.get("best_val", best)
            bad_evals = resume.get("bad_evals", 0)
            best_parts = dict(resume.get("best_parts") or {})
        log(f"resumed at step {start_step}, best val {best:.4f}")
    elif warm_optimizer:
        # continue with the previous run's AdamW moments: a cold optimizer + warm-up
        # first makes the model worse and wastes a third of every block recovering
        try:
            opt.load_state_dict(warm_optimizer)
            log("optimizer state carried over from the previous run")
        except (ValueError, KeyError, RuntimeError) as exc:
            log(f"optimizer state not reusable ({type(exc).__name__}) - cold start")

    out_dir.mkdir(parents=True, exist_ok=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    tag = meta.get("tag", "longtrain")
    last_ckpt, best_ckpt = out_dir / f"{tag}-last.pt", out_dir / f"{tag}-best.pt"
    started = time.time()
    val_start = evaluate(model, val, device)
    with progress.open("a", encoding="utf-8") as f:  # start row: benefit is measured from here
        f.write(json.dumps({"step": start_step, "val_loss": round(val_start, 4), "start": True,
                            "tag": meta.get("tag"), "time": time.time()}) + "\n")
    log(f"start: step {start_step}/{steps}, val {val_start:.4f}, device {device}, bf16 {use_bf16}")

    def save(path: Path, step: int, val_loss: float) -> None:
        torch.save({**meta, "step": step, "val_loss": val_loss, "best_val": best,
                    "bad_evals": bad_evals, "select_metric": metric, "best_parts": best_parts,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": opt.state_dict()}, path)

    stop_reason = "steps_done"
    running, running_n = 0.0, 0
    step = start_step
    # carry > 1: the web corpus is read as running text - `carry` consecutive rows per stream, the state handed
    # from row to row (truncated back-propagation: gradients stay inside a row). The core learns to use what it
    # remembers from before the row; a row costs the same as before. The share of web text stays bulk_frac.
    # carry_share > 0: mixed batches. That share of every batch is running text (streams of `carry` rows, the
    # state carried); the rest is the usual mix read with a fresh state. Every step then holds all kinds of
    # material and most rows still start cold, so the core keeps reading a single row as well as before.
    stream: dict | None = None
    stream_ok = carry > 1 and bulk is not None and len(bulk) > batch_size + carry
    stream_p = bulk_frac / (bulk_frac + carry * (1.0 - bulk_frac)) if stream_ok and bulk_frac < 1 else 1.0
    n_stream = max(1, min(batch_size - 1, int(round(batch_size * min(carry_share, bulk_frac))))) if stream_ok and carry_share > 0 else 0
    for step in range(start_step, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, warmup)
        states_in = None
        sb = None                                    # rows of running text in a mixed batch
        if n_stream:
            if stream is None:
                stream = {"start": np.sort(rng.integers(0, len(bulk) - carry, size=n_stream)), "k": 0, "states": None}
            sb = torch.from_numpy(np.asarray(bulk[stream["start"] + stream["k"]])).long().to(device)
            rest = batch_size - n_stream
            nc = int(rng.binomial(rest, min(1.0, code_frac * batch_size / rest))) if code is not None and len(code) and code_frac > 0 else 0
            nb = int(rng.binomial(rest - nc, min(1.0, max(0.0, (batch_size * bulk_frac - n_stream) / rest))))
            rows = [train[_dataset_rows(rng, len(train), rest - nb - nc, focus, focus_frac)]]
            if nb:
                rows.append(np.asarray(bulk[np.sort(rng.integers(0, len(bulk), size=nb))]))
            if nc:
                rows.append(code[rng.integers(0, len(code), size=nc)])
            b = torch.from_numpy(np.concatenate(rows)).long().to(device)
        elif stream_ok and stream is None and rng.random() < stream_p:
            stream = {"start": np.sort(rng.integers(0, len(bulk) - carry, size=batch_size)), "k": 0, "states": None}
        if n_stream:
            pass
        elif stream is not None:
            b = torch.from_numpy(np.asarray(bulk[stream["start"] + stream["k"]])).long().to(device)
            states_in = stream["states"]
        elif stream_ok:
            # between streams: a batch of the other material (dataset rows and code practice), read with a fresh state
            share = min(1.0, code_frac / max(1e-9, 1.0 - bulk_frac))
            nc = int(rng.binomial(batch_size, share)) if code is not None and len(code) and code_frac > 0 else 0
            rows = [train[_dataset_rows(rng, len(train), batch_size - nc, focus, focus_frac)]]
            if nc:
                rows.append(code[rng.integers(0, len(code), size=nc)])
            b = torch.from_numpy(np.concatenate(rows)).long().to(device)
        elif (bulk is not None and len(bulk)) or (code is not None and len(code) and code_frac > 0):
            # mix per batch: code-school practice (code_frac), big web corpus (bulk_frac), rest dataset
            nc = int(rng.binomial(batch_size, code_frac)) if code is not None and len(code) else 0
            nb = int(rng.binomial(batch_size - nc, bulk_frac)) if bulk is not None and len(bulk) else 0
            rows = [train[_dataset_rows(rng, len(train), batch_size - nb - nc, focus, focus_frac)]]
            if nb and boost is not None and len(boost):
                # part of the web share goes to the language that fell behind (scoreboard)
                nx = int(rng.binomial(nb, boost_frac))
                if nx:
                    rows.append(np.asarray(boost[rng.integers(0, len(boost), size=nx)]))
                nb -= nx
            if nb:
                rows.append(np.asarray(bulk[np.sort(rng.integers(0, len(bulk), size=nb))]))
            if nc:
                rows.append(code[rng.integers(0, len(code), size=nc)])
            b = torch.from_numpy(np.concatenate(rows)).long().to(device)
        else:
            idx = _dataset_rows(rng, len(train), batch_size, focus, focus_frac)
            b = torch.from_numpy(train[idx]).long().to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            # in a stream the whole row goes in (its last token too): the state must be the one the next row starts from
            args_in = (b, states_in) if stream is not None and sb is None else (b[:, :-1],)
            try:
                out = forward(*args_in)
                out_stream = forward(sb, stream["states"]) if sb is not None else None
            except Exception as exc:
                if forward is model:
                    raise
                log(f"the compiled model failed ({type(exc).__name__}: {str(exc)[:200]}) - going on without compiling")
                forward = model
                out = model(*args_in)
                out_stream = model(sb, stream["states"]) if sb is not None else None
            logits = _logits(out)
            if sb is not None:
                loss = (F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), b[:, 1:].reshape(-1), reduction="sum")
                        + F.cross_entropy(out_stream[0][:, :-1].reshape(-1, logits.size(-1)).float(), sb[:, 1:].reshape(-1), reduction="sum")
                        ) / (b[:, 1:].numel() + sb[:, 1:].numel())
                stream["states"], stream["k"] = detach_states(out_stream[1]), stream["k"] + 1
                if stream["k"] >= carry:
                    stream = None
            else:
                if stream is not None:
                    logits = logits[:, :-1]
                    stream["states"], stream["k"] = detach_states(out[1]), stream["k"] + 1
                    if stream["k"] >= carry:
                        stream = None
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), b[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += float(loss)
        running_n += 1

        done = step + 1
        if done % eval_every == 0 or done == steps:
            val_loss = evaluate(model, val, device)
            web_val = evaluate(model, extra_val, device) if extra_val is not None else None
            carried = evaluate_carried(model, extra_val, device) if extra_val is not None and carry > 1 else None
            select = val_loss if web_val is None else (val_loss + web_val) / 2
            improved = select < best - 1e-4
            if improved:
                best, bad_evals = select, 0
                best_parts = {"val": round(val_loss, 4), "step": done}
                if web_val is not None:
                    best_parts["web_val"] = round(web_val, 4)
                save(best_ckpt, done, val_loss)
            else:
                bad_evals += 1
            hours = (time.time() - started) / 3600
            rec = {"step": done, "train_loss": round(running / running_n, 4),
                   "val_loss": round(val_loss, 4), "best_val": round(best, 4),
                   "lr": round(opt.param_groups[0]["lr"], 7), "hours": round(hours, 3),
                   "improved": improved, "time": time.time()}
            if web_val is not None:
                rec["web_val"] = round(web_val, 4)
            if carried is not None:
                rec["web_carried"] = round(carried, 4)
            with progress.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"step {done}: train {rec['train_loss']} val {rec['val_loss']}"
                + (f" web {rec['web_val']}" if web_val is not None else "") + f" best {rec['best_val']} ({hours:.2f} h)")
            running, running_n = 0.0, 0
            if bad_evals >= patience:
                stop_reason = "early_stopping"
                break
        if done % save_every == 0:
            save(last_ckpt, done, best)
        if stop_file.exists():
            stop_file.unlink()
            stop_reason = "stop_file"
            break
        if max_hours and (time.time() - started) / 3600 >= max_hours:
            stop_reason = "time_limit"
            break

    save(last_ckpt, step + 1, best)
    return {"stop_reason": stop_reason, "steps_done": step + 1, "val_start": round(val_start, 4),
            "best_val": round(best, 4), "select_metric": metric, "best_parts": best_parts,
            "hours": round((time.time() - started) / 3600, 3),
            "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
            "last_checkpoint": str(last_ckpt)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=2.5)
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue the last interrupted run")
    ap.add_argument("--extra-dirs", default="",
                    help="comma list of teacher/web JSONL dirs mixed into training (e.g. data/teacher_v1,data/web_v1)")
    ap.add_argument("--bulk-dir", default="", help="bulk web corpus (evo.corpus.bulk_web), e.g. data/bulk_v1")
    ap.add_argument("--bulk-frac", type=float, default=0.7, help="share of each batch taken from the bulk corpus")
    ap.add_argument("--init", default="", help="start from this checkpoint instead of the active weights")
    ap.add_argument("--out-checkpoint", default="",
                    help="save the final weights (no optimizer) here; with --no-activate this is how a clone/node is trained")
    ap.add_argument("--boost-frac", type=float, default=0.15, help="share of the bulk part given to --boost-lang")
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--init-seed", type=int, default=None,
                    help="seed for the random start of a model trained from scratch (default: the library's fixed start, as always); "
                         "with --seed it makes a truly different run of the same experiment")
    ap.add_argument("--compile", action="store_true", help="compile the training forward pass (faster for the generation-8 core)")
    ap.add_argument("--carry", type=int, default=0,
                    help="read the web corpus as running text: this many consecutive rows per stream with the state carried over")
    ap.add_argument("--carry-share", type=float, default=0.0,
                    help="with --carry: this share of every batch is running text, the rest the usual mix read with a fresh "
                         "state (0 = whole batches of running text alternating with whole batches of the rest)")
    ap.add_argument("--config-override", default="",
                    help='experiment: JSON merged into the core config, e.g. {"d_embed": 128, "num_layers": 12}')
    ap.add_argument("--no-activate", action="store_true",
                    help="experiment: train and report, but leave the active weights untouched")
    ap.add_argument("--boost-lang", default="", help="sk/cs/pl/en: extra share of the bulk corpus for this language")
    ap.add_argument("--focus", default="",
                    help="specialist: language of dataset rows to prefer (sk/cs/pl/en/py/rs) or 'teacher' (teacher answers)")
    ap.add_argument("--focus-frac", type=float, default=0.6, help="share of the dataset part of a batch taken from --focus rows")
    ap.add_argument("--seq-mult", type=int, default=1,
                    help="train on N consecutive 128-token sequences joined together (longer context)")
    ap.add_argument("--code-frac", type=float, default=0.0,
                    help="share of each batch from code-school PRACTICE tasks (exam tasks stay unseen)")
    args = ap.parse_args(argv)

    from evo.engine.architecture_factory import build_model

    state = json.loads(STATE.read_text(encoding="utf-8"))
    best = state["best_known"]
    dataset = Path(best["dataset"])
    config = dict(state["primary_parent_config"])
    if args.config_override:
        if not (args.no_activate and args.from_scratch):
            raise SystemExit("--config-override is only for experiments: use with --from-scratch --no-activate")
        config.update(json.loads(args.config_override))
    from nova.weights import current_weights, set_active

    base_tag = f"longtrain-{best.get('candidate')}-{dataset.name}"
    previous = sorted(OUT_DIR.glob(f"{base_tag}-*-last.pt"), key=lambda p: p.stat().st_mtime)
    tag = (previous[-1].name[:-len("-last.pt")] if args.resume and previous
           else f"{base_tag}-{time.strftime('%Y%m%d-%H%M%S')}")
    if args.no_activate:
        tag = f"experiment-{time.strftime('%Y%m%d-%H%M%S')}"
    meta = {"tag": tag, "candidate": best.get("candidate"), "config": config,
            "dataset": str(dataset), "kind": "long_train"}

    from nova.gpu_guard import ensure_vram

    ensure_vram()
    resume = None
    warm_opt = None
    ck = None
    last = OUT_DIR / f"{tag}-last.pt"
    if args.resume and last.exists():
        resume = torch.load(last, map_location="cpu", weights_only=False)
    elif not args.from_scratch:
        init = args.init or current_weights(best)
        ck = torch.load(init, map_location="cpu", weights_only=False)
        if args.init and ck.get("config"):  # a clone keeps the shape of the model it was cloned from
            config = dict(ck["config"])
            meta["config"] = config
    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
    model = build_model(config)
    if ck is not None:
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model_state_dict"].items()})
        meta["init_from"] = init
        print(f"init from {init}")
        if ck.get("kind") == "long_train" and ck.get("optimizer"):
            warm_opt = ck["optimizer"]

    t0 = time.time()
    train = load_tokens(dataset / "train.txt")
    val = load_tokens(dataset / "val.txt")
    n_dataset, teacher_rows = len(train), np.zeros(0, dtype=np.int64)
    if args.extra_dirs:
        n_extra = 0
        for d in [Path(x) for x in args.extra_dirs.split(",") if x]:
            extra = extra_sequences(dataset, [d])
            if len(extra):
                if "web" not in d.name:
                    teacher_rows = np.concatenate([teacher_rows, np.arange(len(train), len(train) + len(extra))])
                train = np.concatenate([train, extra])
                n_extra += len(extra)
        print(f"extra teacher/web sequences: {n_extra}")
    focus = None
    if args.focus:
        if args.seq_mult > 1:
            raise SystemExit("--focus cannot be combined with --seq-mult")
        if args.focus == "teacher":
            focus = teacher_rows
        else:
            ids = {"sk": 4, "cs": 5, "pl": 6, "en": 7, "py": 8, "rs": 9}
            if args.focus not in ids:
                raise SystemExit(f"unknown --focus {args.focus}")
            focus = np.flatnonzero(row_languages(train[:n_dataset]) == ids[args.focus])
        print(f"focus {args.focus}: {len(focus)} rows (share {args.focus_frac})")
        if len(focus) < 64:
            print("focus has too few rows - ignored")
            focus = None
    bulk = None
    if args.bulk_dir and Path(args.bulk_dir).exists():
        from evo.corpus.bulk_web import load_bulk

        state_bulk = Path(args.bulk_dir) / "state.json"
        same_tok = (not state_bulk.exists()
                    or json.loads(state_bulk.read_text()).get("tokenizer_dataset") == str(dataset))
        bulk = load_bulk(Path(args.bulk_dir)) if same_tok else None
        print(f"bulk corpus: {0 if bulk is None else len(bulk)} sequences"
              + ("" if same_tok else " (skipped: built with another tokenizer)"))
    boost = None
    if bulk is not None and args.boost_lang:
        from evo.corpus.bulk_web import load_bulk_lang

        boost = load_bulk_lang(Path(args.bulk_dir), args.boost_lang)
        print(f"boost language {args.boost_lang}: {0 if boost is None else len(boost)} sequences")
    if args.seq_mult > 1:
        # sequences were packed from one continuous token stream, so neighbours are consecutive text
        train = join_sequences(train, args.seq_mult)
        bulk = None if bulk is None else join_sequences(bulk, args.seq_mult)
        boost = None if boost is None else join_sequences(boost, args.seq_mult)
        print(f"longer context: {train.shape[1]} tokens per sequence")
    code = None
    if args.code_frac > 0:
        code = code_practice_sequences(dataset, train.shape[1])
        print(f"code-school practice sequences: {len(code)} (share {args.code_frac})")
    print(f"data: train {train.shape}, val {val.shape} ({time.time() - t0:.0f}s)")

    report = long_train(model, train, val, steps=args.steps, out_dir=OUT_DIR, meta=meta,
                        batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
                        eval_every=args.eval_every, save_every=args.save_every,
                        patience=args.patience, max_hours=args.max_hours, resume=resume,
                        bulk=bulk, bulk_frac=args.bulk_frac, code=code, code_frac=args.code_frac,
                        warm_optimizer=warm_opt, boost=boost, boost_frac=args.boost_frac, seed=args.seed,
                        focus=focus, focus_frac=args.focus_frac if focus is not None else 0.0,
                        compile_model=args.compile, carry=args.carry, carry_share=args.carry_share)
    report["bulk_sequences"] = 0 if bulk is None else int(len(bulk))
    report["params"] = int(sum(p.numel() for p in model.parameters()))
    report["config"] = config
    report["tokens_seen"] = report["steps_done"] * args.batch_size * (train.shape[1] - 1)
    report["train_tokens"] = int(train.size)

    report["improved"] = bool(report["best_checkpoint"]) and report["best_val"] < report["val_start"]
    if args.out_checkpoint and report["last_checkpoint"]:
        ck_out = torch.load(report["last_checkpoint"], map_location="cpu", weights_only=False)
        ck_out.pop("optimizer", None)
        Path(args.out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ck_out, args.out_checkpoint)
        report["out_checkpoint"] = args.out_checkpoint
    if args.no_activate:
        print("experiment run (--no-activate): active weights unchanged")
        for f in (report["best_checkpoint"], report["last_checkpoint"]):
            if f:
                Path(f).unlink(missing_ok=True)
        report["best_checkpoint"] = None
    elif report["improved"]:
        set_active(report["best_checkpoint"], best, "long_train")
        for old in sorted(OUT_DIR.glob(f"{base_tag}-*-best.pt"), key=lambda p: p.stat().st_mtime)[:-2]:
            if str(old) != report["best_checkpoint"]:
                old.unlink(missing_ok=True)
    if report["best_checkpoint"]:
        from nova.generate import continuation_logprob, generate, load_checkpoint_model
        from nova.tokenizer import NovaTokenizer

        m, _ = load_checkpoint_model(report["best_checkpoint"])
        tok = NovaTokenizer.load(dataset / "tokenizer.json")
        print("=== SAMPLES (CPU, temperature 0.7) ===")
        torch.manual_seed(0)
        for lang, prompt in [("sk", "Môj tvorca je"), ("sk", "Bratislava je"),
                             ("en", "The capital of"), ("py", "def main("), ("rs", "fn main() {")]:
            print(f"[{lang}] {generate(m, tok, prompt, lang, temperature=0.7)!r}")
        lp = continuation_logprob(m, tok, "Môj tvorca je", " roppik", "sk")
        print(f"creator recall log-prob: {lp:.2f}")
    print("=== LONG TRAIN REPORT ===")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
