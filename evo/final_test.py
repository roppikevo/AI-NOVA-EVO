import json
import math
import os

import torch
import torch.nn.functional as F

from nova.model import NovaModel as ReferenceNovaModel
from nova.model_scan import NovaModel as ScanNovaModel
from nova.transformer_baseline import TransformerBaseline
from nova.config import NovaConfig
from nova.data import build_datasets


ROOT = "/opt/ai/work/nova-evo"

DATA_ROOT = os.path.join(
    ROOT,
    "data/gen_final_holdout",
)

MANIFEST_FILE = os.path.join(
    ROOT,
    "evo/final/manifest.json",
)

RESULTS_FILE = os.path.join(
    ROOT,
    "evo/final/results.json",
)

TEST_RESULTS_FILE = os.path.join(
    ROOT,
    "evo/final/test_results.json",
)

CHECKPOINT_ROOT = os.path.join(
    ROOT,
    "evo/final/checkpoints",
)

DEVICE = "cuda"

BATCH_SIZE = 8
SEQ_LEN = 128

# Pre-registered before looking at test.txt.
# GEN2-011 and GEN3-005 have identical architecture/configuration.
TEST_CANDIDATES = [
    "GEN0-NOVA",
    "TRANSFORMER-BASELINE",
    "GEN2-011",
    "GEN2-004",
    "GEN3-007",
    "GEN3-024",
    "GEN3-031",
]

TEST_SEED = 1001


def build_config(config_dict):
    cfg = type("Config", (), {})()

    for key, value in config_dict.items():
        setattr(cfg, key, value)

    return cfg


def create_model(candidate):
    cid = candidate["candidate_id"]

    if cid == "GEN0-NOVA":
        return (
            ReferenceNovaModel(NovaConfig()),
            "NOVA-GEN0-REFERENCE",
        )

    if cid == "TRANSFORMER-BASELINE":
        return (
            TransformerBaseline(),
            "TRANSFORMER-BASELINE",
        )

    cfg = build_config(candidate["config"])

    return (
        ScanNovaModel(cfg),
        "NOVA-EVOLVED-SCAN",
    )


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # Deterministic sampling for the final holdout.
    generator = torch.Generator()
    generator.manual_seed(20260926)

    n_batches = max(
        1,
        len(dataset) // BATCH_SIZE,
    )

    for _ in range(n_batches):

        indices = torch.randint(
            0,
            len(dataset),
            (BATCH_SIZE,),
            generator=generator,
        )

        xs = []
        ys = []

        for idx in indices.tolist():
            x, y = dataset[idx]

            xs.append(x)
            ys.append(y)

        x = torch.stack(xs).to(DEVICE)
        y = torch.stack(ys).to(DEVICE)

        result = model(x)

        logits = (
            result[0]
            if isinstance(result, tuple)
            else result
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            y.reshape(-1),
        )

        tokens = y.numel()

        total_loss += (
            loss.item() * tokens
        )

        total_tokens += tokens

    return total_loss / total_tokens


def main():

    print("=" * 78)
    print("NOVA-EVO FINAL HOLDOUT TEST")
    print("=" * 78)

    print("TEST SET: USED NOW")
    print("No training will occur.")
    print(
        f"Checkpoint seed: {TEST_SEED}"
    )
    print()

    with open(
        MANIFEST_FILE,
        "r",
    ) as f:
        manifest = json.load(f)

    manifest_by_id = {
        c["candidate_id"]: c
        for c in manifest["candidates"]
    }

    # Verify test file exists and is untouched.
    test_path = os.path.join(
        DATA_ROOT,
        "test.txt",
    )

    with open(test_path, "r") as f:
        test_lines = f.readlines()

    if len(test_lines) != 2000:
        raise RuntimeError(
            "Expected 2000 test samples, "
            f"got {len(test_lines)}"
        )

    print(
        f"Test samples: {len(test_lines)}"
    )

    print()

    # Verify every required checkpoint before
    # loading the test dataset.
    checkpoint_map = {}

    for cid in TEST_CANDIDATES:

        if cid not in manifest_by_id:
            raise RuntimeError(
                f"Missing candidate in manifest: {cid}"
            )

        checkpoint = os.path.join(
            CHECKPOINT_ROOT,
            cid,
            f"seed_{TEST_SEED}.pt",
        )

        if not os.path.exists(checkpoint):
            raise RuntimeError(
                f"Missing checkpoint: {checkpoint}"
            )

        checkpoint_map[cid] = checkpoint

    # Load only the test split here.
    _, _, test_dataset = build_datasets(
        DATA_ROOT,
        seq_len=SEQ_LEN,
    )

    if len(test_dataset) != 2000:
        raise RuntimeError(
            f"Expected 2000 test samples, "
            f"got {len(test_dataset)}"
        )

    print(
        f"Test dataset loaded: "
        f"{len(test_dataset)} samples"
    )

    print()

    results = []

    for cid in TEST_CANDIDATES:

        candidate = manifest_by_id[cid]

        model, model_type = create_model(
            candidate
        )

        model = model.to(DEVICE)

        checkpoint_path = (
            checkpoint_map[cid]
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        params = sum(
            p.numel()
            for p in model.parameters()
        )

        test_loss = evaluate(
            model,
            test_dataset,
        )

        perplexity = math.exp(
            test_loss
        )

        result = {
            "candidate": cid,
            "model_type": model_type,
            "checkpoint": checkpoint_path,
            "checkpoint_seed": TEST_SEED,
            "params": params,
            "test_loss": test_loss,
            "perplexity": perplexity,
        }

        results.append(result)

        print(
            f"{cid:22s} "
            f"test_loss={test_loss:.6f} "
            f"ppl={perplexity:.2f} "
            f"params={params:,}"
        )

        del model
        del checkpoint

        torch.cuda.empty_cache()

    output = {
        "benchmark":
            "NOVA-EVO-FINAL-HOLDOUT",

        "dataset":
            DATA_ROOT,

        "test_used":
            True,

        "test_evaluation_count":
            1,

        "test_samples":
            len(test_dataset),

        "seq_len":
            SEQ_LEN,

        "batch_size":
            BATCH_SIZE,

        "checkpoint_seed":
            TEST_SEED,

        "selection_note":
            "GEN2-011 and GEN3-005 have identical architecture/configuration; GEN2-011 represents this architecture in the holdout test.",

        "results":
            sorted(
                results,
                key=lambda x: x["test_loss"],
            ),
    }

    with open(
        TEST_RESULTS_FILE,
        "w",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()

    print("=" * 78)
    print("FINAL HOLDOUT TEST COMPLETE")
    print("=" * 78)

    print(
        f"Results: {TEST_RESULTS_FILE}"
    )

    print(
        "TEST SET WILL NOT BE USED "
        "FOR FURTHER MODEL SELECTION."
    )


if __name__ == "__main__":
    main()
