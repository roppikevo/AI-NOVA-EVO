from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evo.engine.architecture_factory import (
    build_model,
    inspect,
)
from evo.engine.experiment_store import ExperimentStore
from evo.engine.knowledge_store import KnowledgeStore
from evo.engine.mutation_engine import (
    architecture_signature,
    generate_candidates,
    load_json,
)
from evo.engine.resource_manager import ResourceManager

ROOT = Path(__file__).resolve().parents[2]

STATE_FILE = ROOT / "evo" / "engine" / "evo_state.json"
SEARCH_FILE = ROOT / "evo" / "engine" / "search_space.json"
REGISTRY_FILE = ROOT / "evo" / "engine" / "mutation_registry.json"

CANDIDATE_DIR = ROOT / "evo" / "gen4" / "candidates"
RESULT_DIR = ROOT / "evo" / "gen4" / "results"
LINEAGE_DIR = ROOT / "evo" / "lineage"


@dataclass
class EvolutionDecision:
    action: str
    generation: int
    candidate_id: str | None
    reason: str
    metrics: dict[str, Any]


class EvolutionEngine:
    """
    NOVA-EVO Evolution Engine.

    This component orchestrates evolutionary generations.
    It does not replace the mutation engine.

    Mutation engine:
        creates candidate architectures.

    Evolution engine:
        decides what happens with those candidates.

    Initial implementation is deliberately DRY-RUN.
    No real training or promotion is performed.
    """

    def __init__(self, root: str | Path = ROOT) -> None:
        self.root = Path(root)

        self.state_file = self.root / "evo" / "engine" / "evo_state.json"
        self.search_file = self.root / "evo" / "engine" / "search_space.json"
        self.registry_file = (
            self.root / "evo" / "engine" / "mutation_registry.json"
        )

        self.candidate_dir = self.root / "evo" / "gen4" / "candidates"
        self.result_dir = self.root / "evo" / "gen4" / "results"
        self.lineage_dir = self.root / "evo" / "lineage"

        self.knowledge = KnowledgeStore(
            root=self.root / "evo" / "knowledge"
        )

        self.experiments = ExperimentStore(
            root=self.root / "evo" / "experiments"
        )

        self.resources = ResourceManager(
            project_root=self.root
        )

    def load_state(self) -> dict[str, Any]:
        return load_json(self.state_file)

    def load_search_space(self) -> dict[str, Any]:
        return load_json(self.search_file)

    def load_registry(self) -> dict[str, Any]:
        return load_json(self.registry_file)

    def prepare_parent(self) -> dict[str, Any]:
        state = self.load_state()

        parent = dict(
            state["primary_parent_config"]
        )

        parent.setdefault(
            "state_architecture",
            "equal",
        )

        parent.setdefault(
            "fusion_architecture",
            "standard",
        )

        parent.setdefault(
            "local_context",
            "single_depthwise",
        )

        return parent

    def generate_generation(
        self,
        count: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        state = self.load_state()
        search_space = self.load_search_space()
        registry = self.load_registry()

        generation = int(
            state["current_generation"]
        )

        parent_id = state["primary_parent"]

        parent = self.prepare_parent()

        if count is None:
            count = int(
                search_space["generation_policy"][
                    "max_candidates_per_generation"
                ]
            )

        if seed is None:
            seed = int(
                state.get(
                    "generation_seed",
                    20260926,
                )
            )

        candidates = generate_candidates(
            parent=parent,
            search_space=search_space,
            registry=registry,
            count=count,
            seed=seed,
        )

        for candidate in candidates:
            candidate["generation"] = generation
            candidate["parent"] = parent_id

        return candidates
    def implementation_screen(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        supported = []
        unsupported = []

        for candidate in candidates:
            config = candidate["config"]

            try:
                info = inspect(config)

                candidate["implementation_validation"] = {
                    "supported": bool(
                        info.get("supported", False)
                    ),
                    "architecture": info.get(
                        "architecture"
                    ),
                    "implementation": info.get(
                        "implementation"
                    ),
                    "reason": info.get(
                        "reason"
                    ),
                }

                if not info.get(
                    "supported",
                    False,
                ):
                    candidate["status"] = (
                        "unsupported"
                    )
                    unsupported.append(candidate)
                    continue

                model = build_model(config)

                parameters = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )

                candidate["implementation_validation"][
                    "model_build"
                ] = True

                candidate["implementation_validation"][
                    "parameters"
                ] = parameters

                candidate["status"] = (
                    "implementation_valid"
                )

                supported.append(candidate)

            except Exception as exc:
                candidate["implementation_validation"] = {
                    "supported": False,
                    "model_build": False,
                    "error": str(exc),
                }

                candidate["status"] = "implementation_failed"
                unsupported.append(candidate)

        return {
            "supported": supported,
            "unsupported": unsupported,
        }

    def technical_screen(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        valid = []
        invalid = []

        for candidate in candidates:
            validation = candidate.get(
                "technical_validation",
                {},
            )

            if validation.get("valid", False):
                valid.append(candidate)
            else:
                invalid.append(candidate)

        return {
            "valid": valid,
            "invalid": invalid,
        }

    def save_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> None:
        self.candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for candidate in candidates:
            path = (
                self.candidate_dir
                / f"{candidate['candidate']}.json"
            )

            with path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    candidate,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
                handle.write("\n")

    def create_manifest(
        self,
        candidates: list[dict[str, Any]],
        screening: dict[str, list[dict[str, Any]]],
        seed: int,
    ) -> dict[str, Any]:
        state = self.load_state()

        manifest = {
            "generation": state["current_generation"],
            "parent": state["primary_parent"],
            "seed": seed,
            "generated": len(candidates),
            "valid": len(screening["valid"]),
            "invalid": len(screening["invalid"]),
            "candidates": [
                {
                    "candidate": item["candidate"],
                    "parent": item["parent"],
                    "mutation": item["mutation"],
                    "signature": item["signature"],
                    "status": item["status"],
                }
                for item in candidates
            ],
        }

        self.result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = self.result_dir / "generation_manifest.json"

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

        return manifest

    def save_lineage(
        self,
        candidates: list[dict[str, Any]],
    ) -> None:
        self.lineage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        state = self.load_state()

        lineage = {
            "generation": state["current_generation"],
            "parent": state["primary_parent"],
            "children": [
                {
                    "candidate": candidate["candidate"],
                    "parent": candidate["parent"],
                    "signature": candidate["signature"],
                    "mutation": candidate["mutation"],
                }
                for candidate in candidates
            ],
        }

        path = (
            self.lineage_dir
            / f"generation_{state['current_generation']}.json"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                lineage,
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

    def dry_run(
        self,
        count: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        candidates = self.generate_generation(
            count=count,
            seed=seed,
        )

        technical = self.technical_screen(
            candidates
        )

        implementation = self.implementation_screen(
            technical["valid"]
        )

        self.save_candidates(
            candidates
        )

        actual_seed = (
            seed
            if seed is not None
            else 20260926
        )

        manifest = self.create_manifest(
            candidates,
            technical,
            actual_seed,
        )

        self.save_lineage(
            candidates
        )

        return {
            "mode": "DRY_RUN",
            "generation": manifest["generation"],
            "parent": manifest["parent"],
            "generated": len(candidates),
            "technical_valid": len(
                technical["valid"]
            ),
            "technical_invalid": len(
                technical["invalid"]
            ),
            "implementation_supported": len(
                implementation["supported"]
            ),
            "implementation_unsupported": len(
                implementation["unsupported"]
            ),
        }

    def selftest(self) -> bool:
        required = [
            self.state_file,
            self.search_file,
            self.registry_file,
        ]

        for path in required:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file missing: {path}"
                )

        state = self.load_state()

        if "current_generation" not in state:
            raise ValueError(
                "evo_state.json missing current_generation"
            )

        if "primary_parent" not in state:
            raise ValueError(
                "evo_state.json missing primary_parent"
            )

        parent = self.prepare_parent()

        signature = architecture_signature(
            parent
        )

        if not signature:
            raise ValueError(
                "Architecture signature generation failed"
            )

        print(
            "EVOLUTION ENGINE SELFTEST: PASSED"
        )

        return True


def main() -> None:
    engine = EvolutionEngine()

    engine.selftest()

    result = engine.dry_run(
        count=3,
        seed=20260926,
    )

    print("=" * 70)
    print("NOVA-EVO EVOLUTION ENGINE")
    print("=" * 70)
    print(f"Mode         : {result['mode']}")
    print(f"Generation   : {result['generation']}")
    print(f"Parent       : {result['parent']}")
    print(f"Generated    : {result['generated']}")
    print(
        f"Technical valid        : "
        f"{result['technical_valid']}"
    )
    print(
        f"Technical invalid      : "
        f"{result['technical_invalid']}"
    )
    print(
        f"Implementation supported   : "
        f"{result['implementation_supported']}"
    )
    print(
        f"Implementation unsupported : "
        f"{result['implementation_unsupported']}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
