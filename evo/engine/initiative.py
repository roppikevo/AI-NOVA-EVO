"""
Initiative: the core proposes - and it is measured whether its proposals are better than the programmed ones
(the Creator's addition 15.10: the core must gradually take over the initiative).

For every weakness a proposal with six parts is written: what it cannot do; what may be the cause; what it wants
to try; what it expects; how that will be checked; what it changes if it fails.

Who proposes is always recorded, and it is never passed off as something else:
  programmed        the director's fixed rules (the first untried way in the order of past wins)
  core-structured   the CORE chooses the cause and the way: it is asked structured questions about its own weakness
                    (the evidence from its own errors is in the question) and its answer is the option it finds most
                    likely (calibrated, like the exams). The help (the questions) is recorded; the choice is the core's.
  (teacher)         a teacher model proposing - reserved; it would be recorded as such and evaluated separately

The director lets them take turns on the problems (A/B), so that the register can show, per author, how often the
proposal was supported and how much it gained per hour. That comparison is the measure of the core's initiative.
"""

from __future__ import annotations

from typing import Any

# cause -> the way of practising that tests it (and the cause's words for the core's question)
CAUSES = {
    "practice": ("chýbali mi cvičenia tohto druhu", "I lacked practice of this kind"),
    "longer": ("cvičení bolo príliš málo", "there was too little practice"),
    "gentle": ("nové učenie mi vytláča staršie schopnosti", "new learning pushes out what I could do"),
    "focus-errors": ("moje chyby majú vzorec, ktorý treba cvičiť cielene", "my errors have a pattern that needs targeted practice"),
}


def question(problem: dict, diag: dict, lang: str = "sk") -> str:
    """The structured question: the core's own evidence about the weakness, then 'what is the cause?'."""
    last = problem.get("last", {})
    acc = last.get("accuracy", 0.0)
    tried = ", ".join(problem.get("tried", [])) or "nič"
    pat = (diag.get("patterns") or [None])[0]
    pat_txt = (f" Najviac chýb robím pri {pat['feature']} = {pat['value']} ({pat['error_rate']:.0%} chýb, inde {pat['elsewhere']:.0%})."
               if pat else " Chyby nemajú jasný vzorec.")
    conf = diag.get("confidently_wrong")
    conf_txt = (f" Pri {conf:.0%} chýb som si bola istá." if conf is not None else "")
    failed = "; ".join(f"{f['strategy']} ({f.get('gain')})" for f in problem.get("failed_hypotheses", [])[-3:]) or "nič"
    return (f"Som NOVA. Úloha: {problem['description']}. Moja presnosť je {acc:.0%}.{pat_txt}{conf_txt} "
            f"Už som skúšala: {tried}. Nevyšlo: {failed}. Prečo to nedokážem? Príčina:")


def core_choice(model, tok, problem: dict, diag: dict, allowed: list[str], device: str = "cpu") -> list[tuple[str, float]]:
    """The core's ranking of the causes it may still test: log P(cause | its own evidence) minus log P(cause | the
    bare question) - what its evidence makes likelier, not what it would say anyway."""
    from evo.learning.skills import _option_logprobs

    opts = [s for s in allowed if s in CAUSES]
    if not opts:
        return []
    texts = [CAUSES[s][0] for s in opts]
    with_evidence = _option_logprobs(model, tok, [(question(problem, diag) + " ", "sk")], [texts], device)[0]
    bare = _option_logprobs(model, tok, [("Prečo to nedokážem? Príčina: ", "sk")], [texts], device)[0]
    return sorted(((s, round(w[0] - b0[0], 3)) for s, w, b0 in zip(opts, with_evidence, bare)), key=lambda x: -x[1])


def proposal(problem: dict, metric: str, strategy: str, hyp: dict, author: str, alternatives: list[str],
             help_used: str | None = None) -> dict[str, Any]:
    """The six parts, written for the register."""
    last = problem.get("last", {})
    nxt = next((a for a in alternatives if a != strategy), None)
    return {
        "author": author,
        "help": help_used,
        "cannot_do": f"{problem['description']}: {last.get('accuracy', 0):.2f} (cieľ {'0,90' if problem.get('class', 'mastery') == 'mastery' else 'potvrdené zlepšenie'})",
        "cause": hyp.get("cause") if author == "programmed" else f"{CAUSES[strategy][0]} (vybrala jadro) – {hyp.get('cause', '')}",
        "approach": strategy,
        "expected": hyp.get("prediction"),
        "verification": "výberová aj rezervovaná potvrdzovacia vzorka, párovo s Holmom; žiadna iná schopnosť významne horšia; text v tolerancii ústavy",
        "if_fails": (f"vyskúšať „{nxt}“ (ďalšia možnosť v poradí {author})" if nxt else "odložiť problém do registra a vrátiť sa pri novej príležitosti"),
    }


def authors_summary(reg: dict) -> dict[str, dict]:
    """Per author: experiments, supported, mean gain, gain per hour - is the core's initiative better?"""
    out: dict[str, dict] = {}
    for p in reg.get("problems", {}).values():
        for e in p.get("history", []):
            if e.get("crashed") or e.get("skipped") or "proposal" not in e:
                continue
            a = e["proposal"]["author"]
            s = out.setdefault(a, {"tries": 0, "supported": 0, "gain": 0.0, "hours": 0.0})
            s["tries"] += 1
            s["supported"] += int(bool(e.get("supported")))
            s["gain"] += e.get("gain") or 0.0
            s["hours"] += e.get("hours") or 0.0
    for s in out.values():
        s["mean_gain"] = round(s["gain"] / s["tries"], 3) if s["tries"] else 0.0
        s["gain_per_hour"] = round(s["gain"] / s["hours"], 3) if s["hours"] else 0.0
    return out
