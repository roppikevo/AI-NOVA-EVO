# Installing NOVA-EVO

You need **Python 3.10 or newer** and about 1.5 GB of free disk space. No graphics card is required: the released cores run on a processor. Git is optional (you can download the ZIP from GitHub instead).

Slovak version: [Slovensky](#slovensky) at the end.

## 1. Get the project

```bash
git clone --depth 1 https://github.com/roppikevo/AI-NOVA-EVO.git
cd AI-NOVA-EVO
```

The released cores come with it, in `evo/releases/` (about 100 MB together).

## 2. Run the installer

| System | Command |
|---|---|
| Linux, macOS | `./install.sh` |
| Windows | double-click `install.bat`, or `py install.py` in a terminal |
| any | `python install.py` |

The installer:

1. checks the Python version,
2. creates a private environment in `.venv/` – nothing is installed system-wide,
3. installs the libraries from `requirements.txt`: PyTorch, NumPy, tokenizers, pytest (the small CPU build of PyTorch unless an NVIDIA card is found),
4. verifies the checksums of the released cores,
5. lets the newest core write a few lines and measures its speed on your processor.

Options: `--cpu` (CPU build of PyTorch even with an NVIDIA card), `--gpu` (default build), `--tests` (also run the full test suite, about a minute), `--corpus` (also the libraries for building a training corpus), `--no-demo`. Running it again is safe; finished steps are skipped.

Expected end of the output (your numbers will differ):

```
NOVA-24M-v1: 23.7 M parameters, 6 files verified, not shipped here: nova_model_fp32.pt

[sk] Bratislava je ...
[en] The river ...
[py] def times_4(x):
    """Return x multiplied by 4."""
    return x * 4

writing on this CPU: 2xx tokens/s with 8 threads; the state the core carries: 100 kB, the same after any length of text
```

### By hand instead

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # or plain: pip install torch
pip install -r requirements.txt
python -m nova.demo
```

## 3. Use a released core

Activate the environment first (`source .venv/bin/activate`, on Windows `.venv\Scripts\activate`).

```bash
python -m nova.demo --list                                         # released cores and their scores
python -m nova.demo --lang sk --prompt "Bratislava je" --tokens 80
python -m nova.demo --lang py --prompt "def times_4(x):" --temperature 0
python -m nova.demo --release NOVA-10M-v1 --lang en --prompt "The river"
python -m nova.demo --speed                                        # tokens per second on your processor
python -m pytest -q                                                # the test suite
```

Languages: `sk`, `cs`, `pl`, `en`, `py` (Python), `rs` (Rust). `\n` inside a prompt is a new line.

These are small cores: the text is fluent but not factual, and the 24 M core solves 40 of the 79 tasks of its code exam – expect wrong functions too.

In your own code:

```python
from nova import demo
from nova.generate import generate

model, tok, info = demo.load(demo.releases()[-1])        # newest release; checksums are verified first
print(generate(model, tok, "Bratislava je", "sk", max_new_tokens=60, temperature=0.7))
```

What a release folder holds:

| File | What it is |
|---|---|
| `nova_model.pt` | the weights in 16-bit floats (call `.float()` before training) |
| `tokenizer.json` | the 16 384-token vocabulary shared by all six languages |
| `MODEL.json` | settings of the core, scores at release time, source commit |
| `SHA256SUMS` | checksums; `nova_model_fp32.pt` is listed but too large for a git repository |
| `blocks_scan.py`, `model_scan.py`, `config.py` | the exact source of the core at release time (a generation-8 release holds `core8.py` instead) |

## 4. Experiment with the core

Continue a released core on your own text, or train an untrained core of any size – one self-contained script, CPU or GPU:

```bash
python examples/train_on_text.py --text my_notes.txt --lang en --steps 300
python examples/train_on_text.py --text my_notes.txt --lang en --steps 2000 \
    --fresh '{"d_model": 256, "d_state": 256, "num_layers": 4}'
```

It holds out a tenth of the text, prints the held-out loss before and after and saves `my_core.pt`.

Build an untrained core of any size in code:

```python
from evo.engine.architecture_factory import build_model

core = build_model({"vocab_size": 16384, "d_model": 896, "d_state": 896, "num_layers": 12,
                    "conv_kernel": 5, "forget_bias": 1.125, "learnable_initial_state": False})
print(sum(p.numel() for p in core.parameters()))
```

Compare the core with a transformer of the same size on your machine: `python -m evo.engine.speed_bench --cpu-only`.

## 5. The full system (training lines, collective, self-improvement loop)

The released cores were trained by the pipeline in `evo/` on the author's server (one RTX 4060, 8 GB). That pipeline expects a built corpus in `data/` and its own state files, which are not part of the repository; its tools are documented at the top of each file:

| Tool | Purpose |
|---|---|
| `evo/corpus/build_text_v1.py`, `evo/corpus/bulk_web.py` | build the corpus (Wikipedia, FineWeb-2, FineWeb-Edu, Python and Rust standard libraries); needs `python install.py --corpus` |
| `evo/engine/train_line.py` | train a core of a given size from scratch, resumable |
| `evo/engine/release.py` | freeze a core as a release with checksums |
| `evo/collective/` | several cores that examine each other, elect a leader and vote on a merged core |
| `evo/engine/director.py`, `judge.py`, `constitution.py`, `nova/surgery.py` | the self-improvement loop and its sealed rules |

A step-by-step guide for running the full pipeline on another machine is not written yet. If you want to try, open an issue.

## Problems

| Message | What to do |
|---|---|
| `Python 3.10 or newer is needed` | install a newer Python from python.org (Windows, macOS) or your package manager |
| `could not create .venv` | Debian/Ubuntu: `sudo apt install python3-venv` |
| PyTorch does not download | check the network; the installer falls back from the CPU build to the default one; otherwise follow pytorch.org/get-started and run the installer again |
| `checksum mismatch` | the download is damaged – clone again |
| `no release with weights` | you have the code without the weights (for example a partial download) – clone the repository again |

To remove everything, delete the project folder; nothing was installed outside it.

## Licence

Free for research, experiments, study and other non-commercial use ([PolyForm Noncommercial 1.0.0](LICENSE)). Commercial use needs the written permission of the creator – open an issue in the repository.

---

## Slovensky

Treba **Python 3.10 alebo novší** a asi 1,5 GB miesta na disku. Grafická karta nie je potrebná.

```bash
git clone --depth 1 https://github.com/roppikevo/AI-NOVA-EVO.git
cd AI-NOVA-EVO
./install.sh            # Windows: install.bat alebo  py install.py
```

Inštalátor vytvorí súkromné prostredie `.venv/` (do systému nič neinštaluje), nainštaluje všetky knižnice z `requirements.txt` (PyTorch, NumPy, tokenizers, pytest), overí kontrolné súčty vydaných jadier a nechá najnovšie jadro napísať pár viet a zmerať rýchlosť na vašom procesore. Dá sa spustiť opakovane.

Potom:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m nova.demo --lang sk --prompt "Bratislava je" --tokens 80
python -m nova.demo --speed
python examples/train_on_text.py --text moj_text.txt --lang sk --steps 300
```

Jadrá sú voľne dostupné na pokusy, výskum a štúdium. Na komerčné použitie treba písomný súhlas tvorcu – napíšte cez „issue“ v repozitári.
