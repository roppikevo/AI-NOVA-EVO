"""
Re-baseline the active core on a new training dataset.

Validation losses are only comparable on the same data. When the
training data changes (e.g. synthetic data/gen4 -> real text
data/text_v1), the active core is re-trained from scratch on the new
data (robust, 3 seeds) and its loss becomes the new best_known baseline.

    python -m evo.engine.rebaseline --dataset data/text_v1 [--steps 1000]

evo_state.json is backed up first and restored if training fails.
The previous best_known is kept in best_known_history.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from evo.engine.training_runner import TrainingRunner, dataset_label

PROMPTS = [
    ("sk", "Môj tvorca je"),
    ("sk", "Kto ťa vytvoril? Vytvoril ma"),
    ("en", "My creator is"),
    ("sk", "Bratislava je"),
    ("en", "The capital of"),
    ("py", "def main("),
    ("rs", "fn main() {"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--root", default="/opt/ai/work/nova-evo")
    args = ap.parse_args(argv)

    root = Path(args.root)
    runner = TrainingRunner(root)
    state_file = runner.state_file
    state = json.loads(state_file.read_text(encoding="utf-8"))

    dataset = root / args.dataset
    for name in ("train.txt", "val.txt"):
        if not (dataset / name).exists():
            print(f"missing {dataset / name}", file=sys.stderr)
            return 1

    config = dict(state["primary_parent_config"])
    manifest_path = dataset / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["vocab_size"]) != int(config["vocab_size"]):
            print(
                f"vocab mismatch: dataset {manifest['vocab_size']} "
                f"vs model {config['vocab_size']}",
                file=sys.stderr,
            )
            return 1

    parent = state["primary_parent"]
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = state_file.with_name(f"evo_state.pre-rebaseline-{ts}.json")
    shutil.copy2(state_file, backup)
    print(f"state backup: {backup}")

    generation = int(str(state.get("parent_generation", "GEN4")).lstrip("GEN") or 4)
    candidate = {
        "candidate": f"{parent}-REBASE-{dataset.name}",
        "generation": generation,
        "parent": parent,
        "config": config,
    }

    state.setdefault("data_policy", {})["training_data"] = [args.dataset]
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"training {candidate['candidate']} on {args.dataset} ({args.steps} steps x 3 seeds)")
    result = runner.run_robust(candidate=candidate, steps=args.steps)

    if result.get("status") != "COMPLETED":
        shutil.copy2(backup, state_file)
        print("REBASELINE FAILED - state restored")
        print(json.dumps(result, indent=2, default=str)[-3000:])
        return 1

    m = result["metrics"]
    old = state.get("best_known")
    state.setdefault("best_known_history", []).append(
        {"replaced_at": time.time(), "reason": f"rebaseline:{args.dataset}", "best_known": old}
    )
    state["best_known"] = {
        **{k: v for k, v in (old or {}).items()
           if k in ("candidate", "core_source", "source_sha256", "checkpoint")},
        "source_generation": f"GEN{generation}",
        "validation_loss": m["validation_loss_mean"],
        "validation_loss_std": m["validation_loss_std"],
        "parameters": m["parameters"],
        "efficiency": m.get("efficiency"),
        "calibration": m.get("calibration"),
        "dataset": result.get("dataset", dataset_label(root, dataset)),
        "rebaseline_checkpoint": result["seed_results"][0]["metrics"].get("checkpoint"),
        "rebaselined_at": time.time(),
    }
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== NEW BASELINE ===")
    print(json.dumps(state["best_known"], indent=2, default=str))

    ckpt = state["best_known"]["rebaseline_checkpoint"]
    tok_path = dataset / "tokenizer.json"
    if ckpt and tok_path.exists():
        from nova.generate import generate, load_checkpoint_model
        from nova.tokenizer import NovaTokenizer

        model, _ = load_checkpoint_model(ckpt)
        tok = NovaTokenizer.load(tok_path)
        print("=== SAMPLES (greedy, CPU) ===")
        for lang, prompt in PROMPTS:
            print(f"[{lang}] {generate(model, tok, prompt, lang)!r}")

        from nova.generate import continuation_logprob
        from evo.corpus.identity import CREATOR

        print("=== CREATOR RECALL (mean log-prob of the name, higher = better) ===")
        for lang, prompt in (("sk", "Môj tvorca je"), ("en", "My creator is"),
                             ("sk", "Vytvoril ma")):
            lp = continuation_logprob(model, tok, prompt, " " + CREATOR, lang)
            ctrl = continuation_logprob(model, tok, prompt, " Peter", lang)
            print(f"[{lang}] {prompt!r}: {CREATOR}={lp:.2f}  control 'Peter'={ctrl:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
