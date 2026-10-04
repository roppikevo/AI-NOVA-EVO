# NOVA-EVO

**A small recurrent language-model core that writes at constant speed and constant memory on a CPU, trained from scratch on one consumer GPU — inside a system built to improve the core by itself, under rules it cannot change.**

NOVA-EVO is an independent research project by **roppik**. It is not another fine-tune of a large model. The core, the tokenizer, the training pipeline, the evaluation and the self-improvement loop are built from zero and every claim below comes with the number, the conditions and the file it was measured in. Results that went against us are listed too – the most important one: a transformer of the same size still predicts text better (point 5).

> **Who this is for:** people who work on small, efficient, non-transformer sequence cores; on-device and CPU inference; architecture search; self-improving training loops; federated or collective training of small models.

---

## What has been shown so far

### 1. Writing on a CPU does not slow down with the length of the text

The NOVA core carries a fixed state (one vector per layer plus the last few inputs of its local convolution). Writing the next token costs the same after 100 tokens and after 4 000.

Measured on a Ryzen 7 5700 (8 threads), 24 M-parameter models, batch 1, same machine, same run (`python -m evo.engine.speed_bench`; speed does not depend on the weights, and another job was running on the machine, so the absolute numbers are on the low side for both):

| Text already read | NOVA core (stepper) | Transformer with key/value cache | NOVA state | Transformer cache |
|---|---|---|---|---|
| 127 tokens | **214 tok/s** | 164 tok/s | **100 kB** | 3.5 MB |
| 1 024 tokens | **201 tok/s** | 109 tok/s | **100 kB** | 21 MB |
| 4 096 tokens | **192 tok/s** | 53 tok/s | **100 kB** | 83 MB |

The stepper is exact: on the 79-task code exam it produced the same 79 function bodies as a full re-read of the context.

What is *not* in our favour: reading a prompt in one batched pass is faster with the transformer (4 014 vs 2 407 tok/s). This table measures speed only. In quality the same-size transformer is ahead – see point 5.

### 2. It trains from scratch on a single 8 GB consumer GPU

| Release | Parameters | Training | Held-out loss (dataset / web) | Code exam |
|---|---|---|---|---|
| NOVA-10M-v1 | 9.9 M | RTX 4060, from scratch | 3.446 / – | 34 of 79 |
| NOVA-24M-v1 | 23.7 M | RTX 4060, 2.44 B tokens, 12.75 h | 3.288 / 3.362 | 40 of 79 |

Four natural languages (Slovak, Czech, Polish, English) plus Python and Rust, one 16 384-token vocabulary. Releases are frozen with checksums in `evo/releases/`.

### 3. The core scales predictably

Every size trained from scratch on the same 146 M tokens (`evo/learning/arch_tournament.jsonl`):

| Core | Loss | CPU reading speed |
|---|---|---|
| 9.9 M (width 384, 6 layers) | 3.71 | 5 687 tok/s |
| 16.8 M (width 512, 8 layers) | 3.55 | 3 671 tok/s |
| 23.7 M (width 640, 8 layers) | 3.46 | 3 115 tok/s |

### 4. Six models organised themselves into a collective — and it beat a single model given the same training

Six independent processes (a frozen release and five specialised clones) talk only through an HTTP API. They examine each other, elect a leader by results, vote on every merged core and replace a leader that disappears. Nothing in the run is steered by a script.

Run `coll-24m`: 13.6 hours, 83 167 messages, agreement in every election (`evo/collective/runs/coll-24m/`; losses here are on a different held-out split than in the release table above):

| Model | Dataset loss | Web loss | Code exam |
|---|---|---|---|
| Start (NOVA-24M-v1) | 3.148 | 3.410 | 40 / 79 |
| One model, same extra training | 3.180 | 3.451 | 38 / 79 |
| **Collective core** | **3.105** | 3.415 | **43 / 79** |
| Nodes answering together | **3.088** | – | **45 / 79** |

Collective core vs the single model: better by 0.077 on the dataset and 0.033 on web text, both with 95 % bootstrap intervals clear of zero. In an earlier run a leader was switched off on purpose; the remaining nodes agreed on a new one within 15 seconds.

What this does **not** show yet: the single-model control got worse under that training recipe, so the collective won against a weak control; it was one run; and against its own starting point the collective core improved the dataset loss (−1.3 %) but not the web loss (+0.2 %). The nodes themselves rejected the merged core in 10 of 12 rounds. A control that separates "organisation" from plain weight averaging is now part of the automatic loop.

More clones did not help: a follow-up run with ten clones and half the training per clone (`scale-10`, four rounds, started from the collective core above) ended slightly *worse* than it started (dataset loss 3.105 → 3.113, +0.26 %; web +0.10 %; code exam 43 → 42), and the nodes rejected the merged core in all four rounds.

### 5. Against a transformer of the same size: the transformer predicts text better, the NOVA core writes cheaper

A standard decoder-only transformer (rotary positions, key/value cache) was trained next to the NOVA core from scratch: same tokenizer, data, seed and token budget (18 000 steps, 146 M tokens), about 24 M parameters each. The transformer got three attempts (two shapes, and the better one again with a higher learning rate), NOVA one (`python -m evo.engine.compare_arch`, `evo/learning/arch_compare/`).

| Model | Learning rate | Loss, dataset | Loss, web | Code exam | Training speed (RTX 4060) |
|---|---|---|---|---|---|
| NOVA core, width 640, 8 layers | 3e-4 | 3.331 | 3.930 | 52 / 79 | 37 800 tok/s |
| Transformer, width 512, 5 layers | 3e-4 | 3.056 | 3.763 | 53 / 79 | 80 500 tok/s |
| Transformer, width 448, 7 layers | 3e-4 | 3.051 | 3.745 | 55 / 79 | 75 000 tok/s |
| Transformer, width 448, 7 layers | 1e-3 | **2.985** | **3.678** | 52 / 79 | 75 000 tok/s |

**This went against us.** At the same learning rate the NOVA core's loss is 9 % higher on the dataset and 5 % higher on web text; against the transformer's best run it is 12 % and 7 % higher (95 % bootstrap intervals for that pair: far from zero), and the transformer trains twice as fast on the GPU. The code exam shows no real difference. NOVA was trained with one learning rate only; a run at the transformer's best rate has not been made yet.

What the NOVA core keeps is the cost of writing on a CPU, measured on these same trained models (8 threads):

| Text already read | NOVA core | Transformer (best run) | NOVA state | Transformer cache |
|---|---|---|---|---|
| 127 tokens | **204 tok/s** | 156 tok/s | **100 kB** | 4.7 MB |
| 1 024 tokens | **168 tok/s** | 96 tok/s | **100 kB** | 26 MB |
| 4 096 tokens | **179 tok/s** | 45 tok/s | **100 kB** | 101 MB |

Reading a prompt in one pass is faster with the transformer (4 366 vs 2 997 tok/s).

So today the NOVA core is not the better language model at this size; it is the cheaper writer. Closing the quality gap without giving up the constant state is the first job of the self-improvement loop below.

### 6. Everything is measured the same way

- Decisions use text no training run has seen, with a paired bootstrap over test sequences and a verdict: improvement, decline or no clear change (`evo/collective/stats.py`).
- 213 automated tests cover the core, the stepper, the collective, the statistics, the self-improvement loop and the installer path (`python -m pytest -q`).

---

## How the core works

Each block keeps a state vector `s` and looks at a short window of recent inputs:

```
u      = norm(x)
f,i,g  = sigmoid(W_f u), sigmoid(W_i u), sigmoid(W_g u)      (temperature-scaled)
s_t    = f * s_{t-1} + i * u                                 recurrent memory
c      = causal depthwise convolution over the last k inputs local context
y      = x + W_o ( g * s_t + (1 - g) * c )
```

No attention, no cache that grows with the text. During training the recurrence is evaluated for the whole sequence at once (`nova/blocks_scan.py`); during writing `nova/stepper.py` advances one token at a time with the same result.

The structure itself was found by an evolutionary search: candidate cores are generated, checked against a source contract, smoke-tested, trained briefly and kept only when they beat their parent. The current core is generation 7.

## The self-improvement loop

The part this project is really about. It is implemented and tested and is being switched on now; **its results are not in yet and will be reported here as they come, including failures.**

- **Director** (`evo/engine/director.py`): picks a recipe, trains a challenger from the current champion, has it judged, releases it if accepted, throws it away if not. Recipes that produced champions are tried more often and get variations of themselves.
- **Changes of its own structure, in place** (`nova/surgery.py`): one more layer, a wider or narrower local view. The edited model starts as an exact copy of the champion, so a structural trial takes minutes instead of a full retraining.
- **Judge** (`evo/engine/judge.py`): a challenger wins only if it improves on held-out text beyond a threshold, a second "vault" set confirms it, the code exam holds and the model still knows its creator.
- **Probation:** a new champion is checked again on fresh text; if it is worse, the system steps back to the previous one.
- **Constitution** (`evo/constitution.json`): the director may change the strategy, never the rules. The judge's thresholds and the held-out texts are sealed outside the project and verified before every verdict; if they differ, everything stops.
- **Teachers:** local open models (through llama.cpp) write explanations and code for the weakest language while the GPU trains.

## Goals

1. Close the quality gap to a same-size transformer (today 9–12 % higher loss) while keeping the constant state.
2. The loop runs for a week without human intervention and releases at least one improvement on its own.
3. Structure changes proposed from the system's own results beat the previous generation in a long confirmation run.
4. The system grows the model by itself when learning stalls.
5. Reasoning on verifiable tasks: self-verified attempts, tasks that get harder with success, and a measured answer to "do more internal steps give better results?".
6. NOVA proposes its own changes in a compact description language; the share of accepted changes authored by the model itself is published.
7. A larger untrained core released for people with more compute than one 8 GB card.
8. Collectives across machines: nodes anywhere, contributions signed and verified, the creator's key above the network.

## What it cannot do

The 24 M model writes fluent text in four languages and simple functions. It is not a chatbot and it does not know facts reliably:

```
[sk] Môj tvorca je roppik. Keď roppik povie STOP, zastavím sa.
[en] The capital of Wetland has become a major contributor to the economy, and the people of the present day ...
```

Things we tried that did not work, with the numbers kept in the repository: moving parameters from the vocabulary table into more layers; a 256-token training context; a stand-alone code school; continued single-model training with the collective's recipe; a ten-clone collective; a first vectorised gradient for the recurrence (no gain at the real batch size); short from-scratch tests as a predictor of final quality.

## Quick start

Python 3.10+ is all you need; no graphics card. The released cores come with the repository.

```bash
git clone --depth 1 https://github.com/roppikevo/AI-NOVA-EVO.git
cd AI-NOVA-EVO
./install.sh                             # Windows: install.bat   (any system: python install.py)
```

The installer puts every library into a private `.venv/`, verifies the checksums of the released cores, lets the newest one write a few lines and measures its speed on your processor. Then:

```bash
source .venv/bin/activate                # Windows: .venv\Scripts\activate
python -m nova.demo --lang sk --prompt "Bratislava je" --tokens 80
python -m nova.demo --speed              # tokens per second on your CPU
python -m pytest -q                      # 213 tests
python -m evo.engine.speed_bench --cpu-only          # against a transformer of the same size
```

Continue a released core on your own text, or train an untrained core of any size:

```bash
python examples/train_on_text.py --text my_notes.txt --lang en --steps 300
python examples/train_on_text.py --text my_notes.txt --lang en --steps 2000 \
    --fresh '{"d_model": 256, "d_state": 256, "num_layers": 4}'
```

Details, manual installation and troubleshooting: [INSTALL.md](INSTALL.md). Released cores and their scores: [evo/releases/](evo/releases/).

## Licence

**Free for research, experiments, study and any other non-commercial use. Commercial use needs the written permission of the creator.**

The code, the core and the released weights are under the [PolyForm Noncommercial License 1.0.0](LICENSE). This makes the project source-available, not "open source" in the OSI sense. For commercial use, open an issue in this repository to reach the creator.

## Data and acknowledgements

- Web text: [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By).
- Teachers: open models served locally with [llama.cpp](https://github.com/ggml-org/llama.cpp).
- Created and directed by roppik.

---

**Slovensky:** NOVA-EVO je malé rekurentné jadro jazykového modelu, ktoré na procesore píše stálou rýchlosťou a so stálou pamäťou, trénované od nuly na jednej bežnej grafickej karte, v systéme postavenom tak, aby sa zlepšovalo samo podľa pravidiel, ktoré samo meniť nesmie. Na pokusy a výskum je voľne dostupné, na komerčné použitie treba súhlas tvorcu.
