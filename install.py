#!/usr/bin/env python3
"""
NOVA-EVO installer. Needs only Python 3.10+; everything else goes into a private folder (.venv).

    python install.py              # Linux, macOS, Windows
    python install.py --cpu        # force the small CPU-only build of PyTorch
    python install.py --gpu        # force the default PyTorch build (CUDA on Linux)
    python install.py --tests      # also run the full test suite (about a minute)
    python install.py --corpus     # also the libraries for building a training corpus

What it does:
  1. checks the Python version
  2. creates .venv/ next to this file (nothing is installed system-wide)
  3. installs PyTorch (CPU build unless an NVIDIA card is found) and requirements.txt
  4. verifies the checksums of the released cores in evo/releases/
  5. lets the newest core write a few lines and measures its speed on your processor

Run it again any time; finished steps are skipped.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def say(text: str) -> None:
    print(f"\n== {text}", flush=True)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str], **kw) -> int:
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], cwd=ROOT, **kw)


def has(py: Path, module: str) -> bool:
    return subprocess.call([str(py), "-c", f"import {module}"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def want_gpu_build(args: argparse.Namespace) -> bool:
    """The default build on Linux carries CUDA (over 2 GB); take it only if there is a card to use it."""
    if args.cpu:
        return False
    if args.gpu:
        return True
    return shutil.which("nvidia-smi") is not None


def make_venv(venv: Path) -> Path:
    py = venv_python(venv)
    if py.exists():
        print(f"   {venv.name}/ already exists")
        return py
    code = run([sys.executable, "-m", "venv", venv])
    if code != 0 or not py.exists():
        hint = "sudo apt install python3-venv" if sys.platform.startswith("linux") else "reinstall Python from python.org with pip included"
        raise SystemExit(f"could not create {venv} - the 'venv' module is missing; try: {hint}")
    return py


def install_torch(py: Path, args: argparse.Namespace) -> None:
    if has(py, "torch"):
        print("   PyTorch is already installed")
        return
    cmd = [py, "-m", "pip", "install", "torch"]
    if want_gpu_build(args):
        print("   NVIDIA card found (or --gpu): installing the default build")
        if os.name == "nt":
            print("   note: on Windows the default build is CPU-only; for CUDA use the command from https://pytorch.org/get-started/locally/")
    elif sys.platform == "darwin":
        print("   macOS: installing the default build")
    else:
        print("   installing the CPU build (about 200 MB); use --gpu for the CUDA build")
        cmd += ["--index-url", CPU_INDEX]
    if run(cmd) != 0:
        if "--index-url" in cmd:
            print("   the CPU build could not be downloaded; trying the default build from PyPI (bigger)")
            if run(cmd[:cmd.index("--index-url")]) == 0:
                return
        raise SystemExit("PyTorch did not install - check the network and https://pytorch.org/get-started/locally/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", action="store_true", help="CPU-only PyTorch even if an NVIDIA card is present")
    ap.add_argument("--gpu", action="store_true", help="default PyTorch build even if no NVIDIA card is detected")
    ap.add_argument("--tests", action="store_true", help="run the full test suite at the end")
    ap.add_argument("--corpus", action="store_true", help="also install requirements-corpus.txt")
    ap.add_argument("--venv", default=".venv", help="folder for the private environment")
    ap.add_argument("--no-demo", action="store_true", help="skip the first run of the core")
    args = ap.parse_args(argv)

    say(f"1/5 Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed (this is {platform.python_version()}). "
                         "Get it from https://www.python.org/downloads/")

    say("2/5 private environment")
    py = make_venv(ROOT / args.venv)
    run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])

    say("3/5 libraries")
    install_torch(py, args)
    if run([py, "-m", "pip", "install", "--quiet", "-r", "requirements.txt"]) != 0:
        raise SystemExit("installing requirements.txt failed")
    if args.corpus and run([py, "-m", "pip", "install", "--quiet", "-r", "requirements-corpus.txt"]) != 0:
        raise SystemExit("installing requirements-corpus.txt failed")
    missing = [m for m in ("torch", "numpy", "tokenizers", "pytest") if not has(py, m)]
    if missing:
        raise SystemExit(f"still missing after the install: {', '.join(missing)}")
    subprocess.call([str(py), "-c", "import torch, numpy, tokenizers; print('   torch', torch.__version__, '| numpy', numpy.__version__, "
                     "'| tokenizers', tokenizers.__version__, '| CUDA', torch.cuda.is_available())"], cwd=ROOT)

    say("4/5 released cores")
    code = run([py, "-m", "nova.demo", "--list"])
    if code != 0:
        print("   no released core in evo/releases/ - the code is installed, but there is nothing to run yet (see INSTALL.md)")

    if code == 0 and not args.no_demo:
        say("5/5 first run")
        if run([py, "-m", "nova.demo"]) != 0:
            raise SystemExit("the core did not run - please open an issue with the lines above")

    if args.tests:
        say("tests")
        if run([py, "-m", "pytest", "-q"]) != 0:
            raise SystemExit("some tests failed - please open an issue with the lines above")

    activate = f"{args.venv}\\Scripts\\activate" if os.name == "nt" else f"source {args.venv}/bin/activate"
    say("done")
    print(f"   {activate}")
    print('   python -m nova.demo --lang en --prompt "The river"')
    print("   python -m nova.demo --speed")
    print("   more in INSTALL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
