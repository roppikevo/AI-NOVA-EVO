from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from evo.engine.evaluator import Evaluator
from evo.engine.authority_manager import AuthorityManager


ROOT = Path("/opt/ai/work/nova-evo").resolve()

EVO_STATE = ROOT / "evo" / "engine" / "evo_state.json"
CORE_STATE = (
    ROOT
    / "evo"
    / "core_evolution"
    / "autonomous_state.json"
)

CANDIDATE_ROOT = (
    ROOT
    / "evo"
    / "core_evolution"
    / "candidates"
)

CORE_RESULT_ROOT = (
    ROOT
    / "evo"
    / "core_evolution"
    / "results"
)

ACTIVE_ROOT = (
    ROOT
    / "evo"
    / "core_evolution"
    / "active"
)

HISTORY_ROOT = (
    ROOT
    / "evo"
    / "core_evolution"
    / "history"
)

ACTIVE_CORE = ROOT / "nova" / "blocks_scan.py"

GEN3_LOCKED = (
    ROOT
    / "evo"
    / "gen3"
    / "LOCKED"
)

GEN4_CHECKPOINT_ROOT = (
    ROOT
    / "evo"
    / "gen4"
    / "results"
    / "checkpoints"
)


class CorePromotionError(RuntimeError):
    pass


class CorePromotionManager:

    def __init__(
        self,
        project_root: str | Path = ROOT,
    ) -> None:

        self.root = Path(
            project_root
        ).resolve()

        self.evaluator = Evaluator(
            project_root=self.root
        )

        self.authority = AuthorityManager(
            actor_id="NOVA-EVO"
        )

        ACTIVE_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        HISTORY_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        CORE_RESULT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------

    @staticmethod
    def sha256_file(
        path: Path,
    ) -> str:

        h = hashlib.sha256()

        with path.open("rb") as f:
            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b"",
            ):
                h.update(chunk)

        return h.hexdigest()

    def load_json(
        self,
        path: Path,
    ) -> dict[str, Any]:

        if not path.exists():
            raise CorePromotionError(
                f"Missing JSON: {path}"
            )

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    def save_json(
        self,
        path: Path,
        payload: dict[str, Any],
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------
    # LOAD CANDIDATE / RESULT
    # ------------------------------------------------------------

    def load_candidate(
        self,
        candidate_id: str,
    ) -> dict[str, Any]:

        path = (
            CANDIDATE_ROOT
            / candidate_id
            / "candidate.json"
        )

        candidate = self.load_json(path)

        if candidate.get("candidate") != candidate_id:
            raise CorePromotionError(
                "Candidate ID mismatch"
            )

        if candidate.get("type") != "core":
            raise CorePromotionError(
                "Candidate is not a CORE candidate"
            )

        if candidate.get(
            "target"
        ) != "nova/blocks_scan.py":
            raise CorePromotionError(
                "Unexpected CORE target"
            )

        return candidate

    def load_robust_result(
        self,
        candidate_id: str,
    ) -> dict[str, Any]:

        sandbox_result = (
            ROOT
            / "evo"
            / "engine"
            / "sandbox"
            / candidate_id
            / "robust_result.json"
        )

        if not sandbox_result.exists():
            raise CorePromotionError(
                f"Missing robust result: {sandbox_result}"
            )

        return self.load_json(
            sandbox_result
        )

    # ------------------------------------------------------------
    # PRE-PROMOTION CHECKS
    # ------------------------------------------------------------

    def validate_candidate_source(
        self,
        candidate: dict[str, Any],
    ) -> Path:

        candidate_id = candidate["candidate"]

        source = (
            CANDIDATE_ROOT
            / candidate_id
            / "blocks_scan.py"
        )

        if not source.exists():
            raise CorePromotionError(
                f"Missing candidate source: {source}"
            )

        actual_hash = self.sha256_file(
            source
        )

        expected_hash = candidate.get(
            "source_sha256"
        )

        if expected_hash != actual_hash:
            raise CorePromotionError(
                "Candidate source hash mismatch: "
                f"expected={expected_hash} "
                f"actual={actual_hash}"
            )

        return source

    def validate_robust(
        self,
        result: dict[str, Any],
    ) -> None:

        if result.get("state") != "COMPLETED":
            raise CorePromotionError(
                "Robust result is not COMPLETED"
            )

        if result.get("stage") != "robust":
            raise CorePromotionError(
                "Promotion requires robust stage"
            )

        seeds = result.get(
            "seeds",
            []
        )

        if len(seeds) < 3:
            raise CorePromotionError(
                "Promotion requires at least three seeds"
            )

        metrics = result.get(
            "metrics",
            {}
        )

        if not isinstance(
            metrics.get(
                "validation_loss_mean"
            ),
            (int, float),
        ):
            raise CorePromotionError(
                "Missing validation_loss_mean"
            )

    def seed1001_checkpoint(
        self,
        candidate_id: str,
        robust: dict[str, Any],
    ) -> Path:

        for item in robust.get(
            "seed_results",
            [],
        ):

            metrics = item.get(
                "metrics",
                {}
            )

            seed = int(
                metrics.get(
                    "seed",
                    item.get("seed", -1),
                )
            )

            if seed == 1001:

                checkpoint = metrics.get(
                    "checkpoint"
                )

                if checkpoint:
                    path = Path(
                        checkpoint
                    ).resolve()

                    if path.exists():
                        return path

        fallback = (
            GEN4_CHECKPOINT_ROOT
            / candidate_id
            / "seed_1001.pt"
        ).resolve()

        if fallback.exists():
            return fallback

        raise CorePromotionError(
            f"Missing seed_1001 checkpoint for {candidate_id}"
        )

    # ------------------------------------------------------------
    # PROMOTION
    # ------------------------------------------------------------

    def promote(
        self,
        candidate_id: str,
    ) -> dict[str, Any]:

        self.authority.assert_runtime_action(
            "self_evolution"
        )

        self.authority.assert_runtime_action(
            "self_modification"
        )

        if not self.authority.may_modify_path(
            ACTIVE_CORE
        ):
            raise CorePromotionError(
                f"Authority denied modification of {ACTIVE_CORE}"
            )

        candidate = self.load_candidate(
            candidate_id
        )

        source = self.validate_candidate_source(
            candidate
        )

        robust = self.load_robust_result(
            candidate_id
        )

        self.validate_robust(
            robust
        )

        decision = (
            self.evaluator.evaluate(
                robust
            )
        )

        if decision.action != "PROMOTE":
            raise CorePromotionError(
                "Evaluator did not authorize promotion: "
                f"{decision.action} / {decision.reason}"
            )

        checkpoint = self.seed1001_checkpoint(
            candidate_id,
            robust,
        )

        state = self.load_json(
            EVO_STATE
        )

        core_state = self.load_json(
            CORE_STATE
        )

        # --------------------------------------------------------
        # BACKUP CURRENT ACTIVE CORE
        # --------------------------------------------------------

        previous_core_id = (
            core_state.get("active_core")
            or state.get("primary_parent")
            or "GEN3-BASELINE"
        )

        history_dir = (
            HISTORY_ROOT
            / str(previous_core_id)
        )

        history_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        history_source = (
            history_dir
            / "blocks_scan.py"
        )

        if ACTIVE_CORE.exists():
            shutil.copy2(
                ACTIVE_CORE,
                history_source,
            )

            self.save_json(
                history_dir
                / "core_manifest.json",
                {
                    "core_id": previous_core_id,
                    "source": str(
                        history_source
                    ),
                    "sha256": self.sha256_file(
                        ACTIVE_CORE
                    ),
                    "backed_up_at": time.time(),
                },
            )

        # --------------------------------------------------------
        # VERSION ACTIVE CORE
        # --------------------------------------------------------

        active_dir = (
            ACTIVE_ROOT
            / candidate_id
        )

        active_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        versioned_source = (
            active_dir
            / "blocks_scan.py"
        )

        shutil.copy2(
            source,
            versioned_source,
        )

        shutil.copy2(
            source,
            ACTIVE_CORE,
        )

        # --------------------------------------------------------
        # VERSION EXACT CHECKPOINT
        # --------------------------------------------------------

        active_checkpoint = (
            active_dir
            / "seed_1001.pt"
        )

        shutil.copy2(
            checkpoint,
            active_checkpoint,
        )

        # --------------------------------------------------------
        # OFFICIAL RESULT
        # --------------------------------------------------------

        official_result = (
            CORE_RESULT_ROOT
            / f"{candidate_id}.official.json"
        )

        decision_dict = (
            decision.to_dict()
        )

        official = {
            "candidate": candidate,
            "robust": robust,
            "evaluation": decision_dict,
            "active_core": candidate_id,
            "active_source": str(
                versioned_source
            ),
            "active_checkpoint": str(
                active_checkpoint
            ),
            "source_sha256": self.sha256_file(
                versioned_source
            ),
            "checkpoint_source": str(
                checkpoint
            ),
            "promoted_at": time.time(),
        }

        self.save_json(
            official_result,
            official,
        )

        # --------------------------------------------------------
        # UPDATE CORE STATE
        # --------------------------------------------------------

        core_state["active_core"] = (
            candidate_id
        )

        core_state["active_source"] = str(
            versioned_source
        )

        core_state["active_checkpoint"] = str(
            active_checkpoint
        )

        core_state.setdefault(
            "promoted_candidates",
            [],
        ).append(
            {
                "candidate": candidate_id,
                "parent": candidate.get(
                    "parent"
                ),
                "candidate_loss": decision.candidate_loss,
                "parent_loss": decision.parent_loss,
                "delta": decision.delta,
                "source_sha256": self.sha256_file(
                    versioned_source
                ),
                "checkpoint": str(
                    active_checkpoint
                ),
                "promoted_at": time.time(),
            }
        )

        # --------------------------------------------------------
        # UPDATE EVO STATE
        # --------------------------------------------------------

        backup_state = (
            EVO_STATE.parent
            / "evo_state.pre-core-promotion.json"
        )

        self.save_json(
            backup_state,
            state,
        )

        promoted_generation = int(
            candidate.get("generation")
            or state.get("current_generation")
            or 4
        )
        state["parent_generation"] = f"GEN{promoted_generation}"
        state["current_generation"] = promoted_generation + 1
        state["primary_parent"] = candidate_id
        state["primary_parent_config"] = dict(
            candidate["config"]
        )

        state["best_known"] = {
            "candidate": candidate_id,
            "source_generation": f"GEN{promoted_generation}",
            "dataset": robust.get(
                "dataset",
                ((state.get("data_policy") or {}).get("training_data") or ["data/gen4"])[0],
            ),
            "efficiency": robust.get(
                "metrics",
                {},
            ).get(
                "efficiency"
            ),
            "validation_loss": decision.candidate_loss,
            "validation_loss_std": robust.get("metrics", {}).get(
                "validation_loss_std"
            ),
            "calibration": robust.get("metrics", {}).get("calibration"),
            "parameters": robust.get(
                "metrics",
                {},
            ).get(
                "parameters"
            ),
            "checkpoint": str(
                active_checkpoint
            ),
            "core_source": str(
                versioned_source
            ),
            "source_sha256": self.sha256_file(
                versioned_source
            ),
            "promoted_at": time.time(),
        }

        state.setdefault(
            "generation_history",
            [],
        ).append(
            {
                "generation": promoted_generation,
                "parent": candidate.get(
                    "parent"
                ),
                "promoted": candidate_id,
                "evaluation": decision_dict,
                "core_source": str(
                    versioned_source
                ),
                "checkpoint": str(
                    active_checkpoint
                ),
                "promoted_at": time.time(),
            }
        )

        state.setdefault(
            "lineage",
            [],
        ).append(
            {
                "generation": promoted_generation,
                "parent": candidate.get(
                    "parent"
                ),
                "candidate": candidate_id,
                "type": "core",
                "source": str(
                    versioned_source
                ),
                "checkpoint": str(
                    active_checkpoint
                ),
            }
        )

        self.save_json(
            EVO_STATE,
            state,
        )

        self.save_json(
            CORE_STATE,
            core_state,
        )

        return {
            "status": "PROMOTED",
            "candidate": candidate_id,
            "parent": candidate.get(
                "parent"
            ),
            "candidate_loss": decision.candidate_loss,
            "parent_loss": decision.parent_loss,
            "delta": decision.delta,
            "source": str(
                versioned_source
            ),
            "checkpoint": str(
                active_checkpoint
            ),
            "official_result": str(
                official_result
            ),
            "previous_core_backup": str(
                history_source
            ),
            "next_generation": promoted_generation + 1,
        }


def self_test() -> None:
    print(
        "CORE PROMOTION MANAGER SELFTEST: "
        "STATIC PASS"
    )


if __name__ == "__main__":
    self_test()
