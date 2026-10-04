from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from nova.model_scan import NovaModel


CANDIDATES = Path("evo/gen1/candidates.json")


def main():
    candidates = json.loads(CANDIDATES.read_text())

    print("=" * 78)
    print("NOVA-EVO GEN 1 — CANDIDATE VALIDATION")
    print("=" * 78)
    print()

    valid = []
    invalid = []

    for candidate in candidates:
        cid = candidate["candidate_id"]
        mutation = candidate["mutation"]
        cfg_dict = candidate["config"]

        try:
            class CandidateConfig:
                pass

            cfg = CandidateConfig()

            for key, value in cfg_dict.items():
                setattr(cfg, key, value)

            model = NovaModel(cfg)

            params = sum(
                p.numel()
                for p in model.parameters()
            )

            x = torch.randint(
                0,
                cfg.vocab_size,
                (2, 32),
                dtype=torch.long,
            )

            with torch.no_grad():
                result = model(x)

            logits = result[0] if isinstance(result, tuple) else result

            expected_shape = (
                2,
                32,
                cfg.vocab_size,
            )

            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(
                    f"wrong logits shape: {tuple(logits.shape)}"
                )

            valid.append(candidate)

            print(
                f"{cid:10s} "
                f"VALID   "
                f"{mutation:18s} "
                f"params={params:,} "
                f"logits={tuple(logits.shape)}"
            )

        except Exception as exc:
            invalid.append(
                {
                    "candidate": candidate,
                    "error": str(exc),
                }
            )

            print(
                f"{cid:10s} "
                f"INVALID "
                f"{mutation:18s} "
                f"reason={exc}"
            )

        finally:
            if "model" in locals():
                del model

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Valid candidates   : {len(valid)}")
    print(f"Invalid candidates : {len(invalid)}")

    Path("evo/gen1/valid_candidates.json").write_text(
        json.dumps(valid, indent=2),
        encoding="utf-8",
    )

    Path("evo/gen1/invalid_candidates.json").write_text(
        json.dumps(invalid, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print("  evo/gen1/valid_candidates.json")
    print("  evo/gen1/invalid_candidates.json")


if __name__ == "__main__":
    main()
