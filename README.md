# NOVA-EVO

**A small recurrent language-model core that writes at constant speed and constant memory on a CPU, trained from scratch on one consumer GPU — inside a system built to improve the core by itself, under rules it cannot change.**

NOVA-EVO is an independent research project by **roppik**. It is not another fine-tune of a large model. The core, the tokenizer, the training pipeline, the evaluation and the self-improvement loop are built from zero and every claim below comes with the number, the conditions and the file it was measured in. Results that went against us are listed too – the most important one: our generation-7 core lost clearly to a transformer of the same size. The generation-8 core that came out of a tournament of candidates has closed most of that gap – level on web text, 2 % behind on the dataset – but it is not ahead (point 5). Trained in full, it replaced generation 7 as the champion: the judge accepted it with +6.1 % on held-out text at half the state. The self-improvement loop then released one improvement on its own and wasted the following 34 hours repeating itself; what went wrong and what was changed is under "The self-improvement loop".

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

What is *not* in our favour: reading a prompt in one batched pass is faster with the transformer (4 014 vs 2 407 tok/s). This table measures speed only, for the generation-7 core of the releases so far. Quality against a transformer, and the generation-8 core, are in point 5.

### 2. It trains from scratch on a single 8 GB consumer GPU

| Release | Parameters | Training | Held-out loss (dataset / web) | Code exam |
|---|---|---|---|---|
| NOVA-10M-v1 | 9.9 M | RTX 4060, from scratch | 3.446 / – | 34 of 79 |
| NOVA-24M-v1 | 23.7 M | RTX 4060, 2.44 B tokens, 12.75 h | 3.288 / 3.362 | 40 of 79 |
| NOVA-24M-v2 | 23.7 M | the above + one round of the collective (point 4) | 3.250 / 3.366 | 43 of 79 |
| NOVA8-24M-v1 | 24.2 M | generation 8, RTX 4060, from scratch, 300 000 steps, 12.2 h | 3.006 / 3.170 | 53 of 79 |
| NOVA8-24M-v2 | 24.2 M | the above + average of five checkpoints (the director's own step) | 2.961 / 3.156 | 53 of 79 |

Four natural languages (Slovak, Czech, Polish, English) plus Python and Rust, one 16 384-token vocabulary. Releases are frozen with checksums in `evo/releases/`. (Frozen files are not rewritten, so two slips stay in them: `MODEL.json` of the releases up to NOVA8-24M-v2 counts the token table twice in `parameters`, and NOVA8-24M-v2 carries the label of an older core. The table here and the index in `evo/releases/` have the right numbers.)

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

### 5. Against a transformer of the same size: generation 7 lost clearly; generation 8 is level on web text and 2 % behind on the dataset, with 56 kB of state

A standard decoder-only transformer (rotary positions, key/value cache) was trained next to the NOVA core from scratch: same tokenizer, data, seed and token budget (18 000 steps, 146 M tokens), about 24 M parameters each (`python -m evo.engine.compare_arch`, `evo/learning/arch_compare/`). Both sides got three learning rates.

| Model | Learning rate | Loss, dataset | Loss, web | Code exam | Training speed (RTX 4060) |
|---|---|---|---|---|---|
| NOVA generation 7, width 640, 8 layers | 3e-4 | 3.331 | 3.930 | 52 / 79 | 37 800 tok/s |
| | 1e-3 | 3.220 | 3.809 | 41 / 79 | |
| | 2e-3 | 3.203 | 3.781 | 45 / 79 | |
| Transformer, width 512, 5 layers | 3e-4 | 3.056 | 3.763 | 53 / 79 | 80 500 tok/s |
| Transformer, width 448, 7 layers | 3e-4 | 3.051 | 3.745 | 55 / 79 | 75 000 tok/s |
| | 1e-3 | 2.985 | **3.678** | 52 / 79 | |
| | 2e-3 | **2.983** | 3.687 | 59 / 79 | |

**This went against us.** Best run against best run, the generation-7 core's loss is 7.4 % higher on the dataset (95 % bootstrap interval of the difference: 0.20 to 0.24) and 2.6–2.8 % higher on web text, and the transformer trains twice as fast. During one night the self-improvement loop then tried nine changes to that core (more web text, teachers, one more layer, a wider and a narrower local view, a collective); the judge rejected all nine. Generation 7 was at its ceiling.

**Generation 8** was therefore built as a tournament: one block design, several memory mixers whose state has a fixed size, every candidate trained the same way (about 24 M parameters, 18 000 steps, learning rate 1e-3, one run each).

| Candidate | What its seven blocks hold | Loss, dataset | Loss, web | Web read as running text | Code exam | Training speed | State |
|---|---|---|---|---|---|---|---|
| lru | a gated linear recurrent unit | 3.137 | 3.729 | – | 36 / 79 | 38 500 tok/s | 49 kB |
| mlp | the generation-7 memory | 3.124 | 3.708 | 3.631 | 45 / 79 | 60 900 tok/s | 61 kB |
| slot | recurrent unit + slot memory | 3.096 | 3.719 | 3.768 | 51 / 79 | 61 000 tok/s | 49 kB |
| hash | recurrent unit + hash-table memory | 3.098 | 3.716 | 4.255 | 53 / 79 | 34 200 tok/s | 196 kB |
| win | recurrent unit + attention over the last 32 tokens | 3.039 | 3.688 | 3.751 | 52 / 79 | 66 600 tok/s | 252 kB |
| nwin16 | generation-7 memory + attention over the last 16 tokens | 3.103 | 3.699 | 3.623 | 50 / 79 | 65 100 tok/s | 149 kB |
| nwin3 | generation-7 memory + three window blocks | 3.031 | 3.678 | 3.599 | 49 / 79 | 67 500 tok/s | 361 kB |
| nwin | generation-7 memory + attention over the last 32 tokens | 3.028 | 3.674 | 3.595 | 55 / 79 | 62 400 tok/s | 261 kB |
| nsw | generation-7 memory + slots + window | **3.016** | **3.668** | **3.588** | 52 / 79 | 63 300 tok/s | 257 kB |
| **nslot** | **generation-7 memory + slot memory, no window** | 3.045 | 3.680 | 3.597 | 54 / 79 | 53 900 tok/s | **56 kB** |

Every block also has a gated non-linear layer, which generation 7 did not have. All candidates except lru were trained with the compiled forward pass (`torch.compile`, same arithmetic).

What the tournament says:

- **The winner is `nslot`**: purely recurrent, no attention over past tokens at all, 56 kB of state – half of generation 7. Against the best transformer run its loss is 2.0 % higher on the dataset; on web text there is no clear difference (3.680 against 3.678 and 3.687). Against the best generation-7 run it is 4.9 % and 2.7 % lower.
- A small window of the last 32 tokens buys a little more (`nsw`: 1.1 % behind the transformer on the dataset, 0.3 % ahead on web text) for a state of 257 kB that still does not grow. `nslot` is within 1 % of it, so the purely recurrent core was chosen; `nsw` stays as the second line.
- The old generation-7 memory beat the new recurrent unit once both had the non-linear layer, and candidates built on it read **running text** better without being trained for it: with the state carried from row to row, `nslot` goes from 3.680 to 3.597 (−2.3 %; most of that gain sits in the first tokens of each row). Candidates built on the recurrent unit got worse when their state was carried.
- Training on running text pays when the batches are mixed (a third of every batch is running text with the state carried, the rest starts cold): see the next table. Whole batches of consecutive rows cost 1.5–2.2 % on single rows and were dropped.

**Running text.** The same tokens predicted with different amounts of context (480 rows of held-out web text, positions inside the rows; `python -m evo.engine.long_context --rows 120 --every 8`):

| Model | Each row alone | Re-reading the last 127 tokens for every token | Two rows in one pass | State carried through the whole text |
|---|---|---|---|---|
| Transformer, learning rate 1e-3 | 3.643 | **3.540** | 4.157 | – |
| Transformer, learning rate 2e-3 | 3.666 | 3.556 | 3.814 | – |
| NOVA generation 7 | 3.746 | 3.689 | 3.687 | 3.687 |
| `nslot` | 3.663 | 3.590 | 3.584 | 3.646 |
| `nslot` trained with mixed batches | 3.673 | 3.586 | 3.578 | **3.568** |
| `nsw` | 3.645 | 3.569 | 3.564 | 3.689 |

- With its 56 kB state carried through the text, `nslot` trained with mixed batches reaches 3.568: 2.1–2.7 % better than the transformer reading each row alone, and better than `nslot` re-reading its own last 127 tokens. The price is 0.3 % on single rows.
- The transformer is still ahead when it may re-read the last 127 tokens for every token it predicts (3.540–3.556, 0.3–0.8 % better than the carried state). That costs a full pass over 127 tokens per token; a cache of the last 127 keys and values (about 4.7 MB) would be the cheap form and was not measured. Given more than its training length in one pass, the transformer breaks (+4 % and +14 %).
- Without that training `nslot` gains only 0.5 % from the carried state inside a row, and `nsw` loses 1.2 %.

What is **not** in our favour, or not known yet:

- One run per candidate and one learning rate, against three for the transformer and generation 7. The spread between runs was not measured when the tournament ran (the loop is measuring it now: the same candidate with three seeds); the code exam alone swings between 36 and 59 with no pattern. A finer reading of the same exam – points for valid Python and for every single test passed, `evo/learning/fine_exam.py` – puts the best transformer run first with 82.0 of 100, `nslot` at 78.6 and generation 7 at 72.6.
- The transformer still trains faster (75 000 tok/s without compiling; compiling gave a transformer block another 1.24× in a block-level test).
- Writing on a CPU, `nslot` reaches 162 tok/s at 127 tokens and 138 tok/s after 4 096 tokens (8 threads) – constant state, but slower than generation 7 with its fused stepper (204 and 179) and about level with the transformer on short texts (156).
- On running text a transformer that re-reads a sliding window is still slightly ahead (table above). The 101 MB in the table below is a transformer that keeps everything it has read; one that keeps only the last 127 tokens needs about 4.7 MB.
- These are 18 000-step runs. The long run of `nslot` (300 000 steps, mixed batches of running text, 12.2 hours on the RTX 4060) is done and the judge accepted it against the generation-7 champion: +6.11 % on the held-out sets, code exam 43 → 53 of 79, state 100 kB → 56 kB (released as NOVA8-24M-v1). **No transformer has been trained that long here**, so whether the short-run picture against a transformer holds in a long run is not known.

The cost of writing on a CPU for the two models of the first table (8 threads):

| Text already read | NOVA generation 7 | Transformer (best run) | NOVA state | Transformer cache |
|---|---|---|---|---|
| 127 tokens | **204 tok/s** | 156 tok/s | **100 kB** | 4.7 MB |
| 1 024 tokens | **168 tok/s** | 96 tok/s | **100 kB** | 26 MB |
| 4 096 tokens | **179 tok/s** | 45 tok/s | **100 kB** | 101 MB |

Reading a prompt in one pass is faster with the transformer (4 366 tok/s; generation 7: 2 997, `nslot`: about 3 500).

So today: at 24 M parameters and a short training run the generation-8 core is as good as a same-size transformer on web text and close on the dataset, with a state of 56 kB. It is not better. Trained in full it is clearly better than our own generation 7; the same long run for a transformer is missing.

### 6. Everything is measured the same way

- Decisions use text no training run has seen, with a paired bootstrap over test sequences and a verdict: improvement, decline or no clear change (`evo/collective/stats.py`).
- 317 automated tests cover the core, the stepper, the collective, the statistics, the self-improvement loop and the installer path (`python -m pytest -q`).

---

## How the core works

**Generation 7** (the releases so far). Each block keeps a state vector `s` and looks at a short window of recent inputs:

```
u      = norm(x)
f,i,g  = sigmoid(W_f u), sigmoid(W_i u), sigmoid(W_g u)      (temperature-scaled)
s_t    = f * s_{t-1} + i * u                                 recurrent memory
c      = causal depthwise convolution over the last k inputs local context
y      = x + W_o ( g * s_t + (1 - g) * c )
```

No attention, no cache that grows with the text. During training the recurrence is evaluated for the whole sequence at once (`nova/blocks_scan.py`); during writing `nova/stepper.py` advances one token at a time with the same result. This structure was found by an evolutionary search: candidate cores are generated, checked against a source contract, smoke-tested, trained briefly and kept only when they beat their parent.

**Generation 8** (`nova/core8.py`, in its long training now). A block is a memory mixer followed by a gated non-linear layer:

```
x = x + mixer(norm(x))
n = norm(x)
x = x + W_o ( gelu(W_a n) * (W_b n) )
```

The winning core `nslot` alternates two mixers, N S N S N S N, at width 448 (24.2 M parameters):

```
N   the generation-7 memory above: running average blended with the local convolution
S   a table of 16 slots x 112 numbers
    write:  share = softmax(address(u)) * sigmoid(gate(u))
            table = (1 - share) * table + share * value(u)       every token, into the slots its content points to
    read:   y = W_o ( softmax(query(u) . table) table )          by comparing a query with what the slots hold
```

Its whole state – one vector and four past inputs per N block, one table per S block – is 56 kB and has the same size after any length of text. The recurrences are computed in closed form over the whole sequence during training; one token at a time gives the same numbers (tested). What the slots do in the trained short-run core (`python -m evo.engine.slot_probe`, held-out web text): with the three slot blocks switched off the loss rises by 4.0 %; reading every slot equally costs 3.7 %, so their worth is the choice by content, not one more average. The write gate is open for only 6–12 % of the tokens, a token writes into three or four slots, and a slot keeps what it holds for a median of 50–100 tokens. About half of the 16 slots are in real use and a query spreads over about ten of them – room for improvement that has not been used yet. What gets written (`--contents`): the slots sort tokens by kind – English function words, Slavic words, brackets and quotes, numbers, capitals, word endings, line breaks and language tags. A slot tells 1.0–1.4 bits about the token written into it, under 0.1 bit about the language and nothing about the position.

Other mixers in the file (a gated linear recurrent unit, a hash-table memory, a matrix memory, attention over a fixed window of recent tokens) were candidates in the tournament of point 5.

## The self-improvement loop

The part this project is really about. It is implemented, tested and running. **Results so far, in order:**

1. The judge accepted the collective core of point 4 (NOVA-24M-v2, +0.6 % on held-out text). In the following night the director tried nine more changes to the generation-7 core by itself and the judge rejected every one (each came out 0.1 % to 1.4 % worse, or failed).
2. Generation 8 came from a tournament set up by hand, not from a proposal of the loop. The loop trained the winner for 300 000 steps and judged it like any other challenger: **accepted, +6.11 %** (NOVA8-24M-v1).
3. The director's first own step on it – averaging five checkpoints from the end of that run – was accepted with +0.98 % (NOVA8-24M-v2, the current champion).
4. **Then the loop wasted 34 hours.** 54 attempts, nothing that lasted. The same nine recipes went round about four times with practically the same result each time (eight of them between −0.87 % and +0.19 %; the judge needs +0.3 %), because the director remembered which recipes had produced champions but not which had been rejected on the champion in front of it. The ninth, a collective, came out at +0.29 % twice and at +0.30 % twice: the two that got through by a hair were released as v3 and v4 and failed probation on fresh text a few hours later. The system stepped back both times, as designed – but both releases had already been published here and had to be withdrawn.

What was changed after that (2026-10-07): the director keeps a memory of every recipe per champion and does not repeat what was rejected, stepped back or impossible; a release is published only after its probation; and when nothing untried is left, the director measures instead of repeating (below). Whether this makes the loop productive is not shown yet. Failures will keep being reported here.

- **Director** (`evo/engine/director.py`): picks a recipe, trains a challenger from the current champion, has it judged, releases it if accepted, throws it away if not. Recipes that produced champions are tried more often and get variations of themselves; a recipe that failed on a champion is not tried on it again (recipes that depend on newly collected text may return after three days).
- **Its own tournaments** (`evo/engine/explore.py`): with no untried recipe left, the director does what was done by hand for generation 8. It first trains the champion's architecture again with other seeds, to know how much two runs of the same experiment differ; then every untried candidate from a registry gets one short run. A candidate goes on to a full training run only if its state does not grow, its gain is at least twice that spread (and at least 0.3 %), and its code exam is not clearly below what the champion's architecture reaches between seeds. A candidate that is as good at a clearly smaller state or faster training is noted as the cheaper build, but not trained in full on that alone: the judge accepts only a gain in quality, so those twelve hours could not end in a release. With nothing left it waits for a new hypothesis instead of repeating itself. The candidates in the registry are still written by a human.
- **Identity test** (`evo/engine/identity.py`), before the judge: the state must have the same size after 128, 1 024 and 4 096 tokens, the model must not look into the future, and one token at a time must give the same numbers as a whole sequence. The current champion passes with 56 kB and no attention over stored tokens; a candidate with a fixed window of recent tokens passes as a hybrid (257 kB); the transformer of point 5 fails (its cache grows by 25 kB per token).
- **Ledger** (`evo/ledger.jsonl`): every tournament run, attempt and generation with its conditions, cost and verdict – what the director consults before trying something again.
- **Lessons** (`evo/engine/lessons.py`): the summary on top of that memory. Repeats of one attempt are folded into one experiment and used to measure how much the judge's numbers move by chance (the code exam too). Each recipe gets a reading with the number of experiments and champions behind it; each setting (share of web text, learning rate, steps, share of code) gets a comparison that is marked as confounded as long as the values came with different recipes. With no untried recipe left, the director tests such a lesson itself: one setting of a recipe it has already judged is moved, nothing else, so the verdict is about that setting alone. A few per champion. This is statistics over its own attempts, not new ideas.
- **Changes of its own structure, in place** (`nova/surgery.py`): one more layer, a wider or narrower local view. The edited model starts as an exact copy of the champion, so a structural trial takes minutes instead of a full retraining.
- **Judge** (`evo/engine/judge.py`): a challenger wins only if it improves on held-out text beyond a threshold, a second "vault" set confirms it, the code exam holds and the model still knows its creator.
- **Probation:** a new champion is checked again on fresh text; if it is worse, the system steps back to the previous one.
- **Constitution** (`evo/constitution.json`): the director may change the strategy, never the rules. The judge's thresholds and the held-out texts are sealed outside the project and verified before every verdict; if they differ, everything stops.
- **Teachers:** local open models (through llama.cpp) write explanations and code for the weakest language while the GPU trains.
- **What the model is told about itself** (`evo/corpus/self_v2.py`): a small set of training texts with its creator's authority first, then that getting better – confirmed by the judge – is its first task, then measured facts about its current core. This is text the model learns to write, not a mechanism: the drive to improve is in the director, which never idles while something untried is left.

## Goals

1. Close the remaining gap to a same-size transformer while keeping the constant state (generation 7: 7.4 % higher loss on the dataset and 2.6–2.8 % on web text; generation 8 in short runs: 2.0 % and level). The long run confirmed generation 8 against generation 7; an equally long transformer run is still to be done.
2. The loop runs for a week without human intervention and releases at least one improvement on its own. (So far: one improvement released on its own, then 34 hours of repeated attempts that needed a fix by hand.)
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

Things we tried that did not work, with the numbers kept in the repository: moving parameters from the vocabulary table into more layers; a 256-token training context; a stand-alone code school; continued single-model training with the collective's recipe; a ten-clone collective; a first vectorised gradient for the recurrence (no gain at the real batch size); short from-scratch tests as a predictor of final quality; nine changes the director tried on the generation-7 core in its first night; a gated linear recurrent unit in place of the old memory; a hash-table memory (no better than the slots, half the training speed, four times the state); training on whole batches of running text (single rows got 1.5–2.2 % worse); 54 attempts of the director on the generation-8 champion in 34 hours (nine recipes repeated about four times, two collective releases stepped back after probation).

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
python -m pytest -q                      # 263 tests
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
