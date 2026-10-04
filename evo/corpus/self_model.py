"""
Self-model corpus: NOVA learns facts about itself from its own state.

Generated from evo/engine/evo_state.json (no model is run), so it costs
nothing and is refreshed whenever the corpus is rebuilt after a
promotion. Facts: name, creator, generation, active core, parent,
size, speed on CPU, VRAM use, validation loss, dataset, lineage.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from evo.corpus.identity import CREATOR, MODEL_NAME
from evo.corpus.sources import Document


def facts_from_state(state: dict) -> dict:
    best = state.get("best_known") or {}
    eff = best.get("efficiency") or {}
    params = best.get("parameters") or eff.get("parameters") or 0
    return {
        "m": MODEL_NAME,
        "c": CREATOR,
        "gen": state.get("current_generation"),
        "core": best.get("candidate") or state.get("primary_parent"),
        "parent_gen": state.get("parent_generation"),
        "params_m": f"{params / 1e6:.1f}".replace(".", ",") if params else None,
        "params_m_en": f"{params / 1e6:.1f}" if params else None,
        "mb": round(eff["param_mb"]) if eff.get("param_mb") else None,
        "cpu": int(eff["cpu_tokens_per_sec"]) if eff.get("cpu_tokens_per_sec") else None,
        "vram": int(eff["peak_vram_mb"]) if eff.get("peak_vram_mb") else None,
        "state_kb": eff.get("state_kb"),
        "loss": f"{best['validation_loss']:.2f}" if best.get("validation_loss") else None,
        "dataset": best.get("dataset"),
        "lineage": len(state.get("lineage") or []),
    }


TEMPLATES = {
    "sk": [
        ("gen", "Som {m}, generácia {gen}."),
        ("core", "Moje aktívne jadro sa volá {core}."),
        ("params_m", "Mám približne {params_m} milióna parametrov."),
        ("mb", "Moje váhy zaberajú asi {mb} MB."),
        ("cpu", "Na procesore spracujem približne {cpu} tokenov za sekundu."),
        ("vram", "Pri tréningu potrebujem najviac {vram} MB grafickej pamäte."),
        ("state_kb", "Môj vnútorný stav má iba {state_kb} KB a pri dlhom texte nerastie."),
        ("loss", "Moja validačná strata je {loss}; čím je nižšia, tým lepšie rozumiem textu."),
        ("lineage", "Môj rodokmeň má {lineage} zaznamenaných krokov evolúcie."),
        ("c", "Môj tvorca je {c}."),
        ("m", "Moje jadro sa vyvíja samo: navrhujem zmeny, testujem ich a ponechám iba tie, ktoré sú lepšie."),
        ("m", "Snažím sa byť malá a rýchla, aby som bežala hlavne na procesore a v RAM."),
        ("m", "Keď si nie som istá, mám povedať, že neviem."),
    ],
    "en": [
        ("gen", "I am {m}, generation {gen}."),
        ("core", "My active core is {core}."),
        ("params_m_en", "I have about {params_m_en} million parameters."),
        ("cpu", "On a CPU I process about {cpu} tokens per second."),
        ("vram", "Training me needs at most {vram} MB of GPU memory."),
        ("loss", "My validation loss is {loss}."),
        ("c", "My creator is {c}."),
        ("m", "My core evolves: I propose changes, test them and keep only the better ones."),
        ("m", "When I am not sure, I should say that I do not know."),
    ],
}


def self_model_documents(
    state_path: Path,
    n_docs: int = 400,
    seed: int = 1,
) -> Iterator[Document]:
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    facts = facts_from_state(state)
    rng = random.Random(seed)
    usable = {
        lang: [t for key, t in tpl if facts.get(key) is not None]
        for lang, tpl in TEMPLATES.items()
    }
    seen: set[str] = set()
    for _ in range(n_docs * 20):
        if len(seen) >= n_docs:
            break
        lang = rng.choice(["sk", "sk", "en"])
        pool = usable[lang]
        text = " ".join(t.format(**facts) for t in rng.sample(pool, rng.randint(2, min(5, len(pool)))))
        if text not in seen:
            seen.add(text)
            yield Document(lang, "self-model", text)
