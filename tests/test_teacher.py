"""A new, bigger core can learn from an earlier one as well as from the text (evo.engine.long_train.teacher_loss)."""

import json

import numpy as np
import torch
import torch.nn.functional as F

from evo.engine.architecture_factory import build_model
from evo.engine.long_train import load_teacher, long_train, teacher_loss

SMALL = {"arch": "nova8", "vocab_size": 60, "d_model": 16, "pattern": "NS", "mlp_hidden": 24, "heads": 2, "slots": 4}
BIG = {"arch": "nova8", "vocab_size": 60, "d_model": 32, "pattern": "NSN", "mlp_hidden": 48, "heads": 4, "slots": 4}


def test_the_loss_is_the_cross_entropy_against_the_teachers_likeliest_tokens():
    torch.manual_seed(0)
    teacher, student = build_model(SMALL).eval(), build_model(BIG)
    x = torch.randint(0, 60, (3, 9))
    logits = student(x)[0] if isinstance(student(x), (tuple, list)) else student(x)
    full = teacher_loss(teacher, x, logits, topk=60)                       # all tokens: plain soft cross-entropy
    with torch.no_grad():
        t = teacher(x)
        t = (t[0] if isinstance(t, (tuple, list)) else t).float()
    want = -(F.softmax(t, -1) * F.log_softmax(logits.float(), -1)).sum(-1).mean()
    assert torch.allclose(full, want, atol=1e-5)
    top = teacher_loss(teacher, x, logits, topk=8)
    assert torch.isfinite(top) and top.requires_grad and top.item() != full.item()
    same = teacher_loss(teacher, x, (teacher(x)[0] if isinstance(teacher(x), (tuple, list)) else teacher(x)).detach(), topk=60)
    assert same < full                                                      # agreeing with the teacher costs least


def test_training_with_a_teacher_pulls_the_student_towards_it_and_leaves_the_teacher_alone(tmp_path):
    torch.manual_seed(0)
    teacher = build_model(SMALL)
    torch.save({"config": SMALL, "model_state_dict": teacher.state_dict()}, tmp_path / "teacher.pt")
    loaded = load_teacher(tmp_path / "teacher.pt")
    before = {k: v.clone() for k, v in loaded.state_dict().items()}
    assert load_teacher("") is None
    train = np.random.default_rng(1).integers(12, 60, size=(64, 16)).astype(np.int32)
    x = torch.from_numpy(train[:8, :-1]).long()

    def gap(model):
        out = model(x)
        with torch.no_grad():
            return float(teacher_loss(loaded.cpu(), x, out[0] if isinstance(out, (tuple, list)) else out, 60))

    def run(name, **kw):
        torch.manual_seed(5)
        m = build_model(BIG)
        rep = long_train(m, train, train[:8], steps=40, out_dir=tmp_path / name, meta={"tag": name}, batch_size=8, warmup=1, lr=3e-3,
                         eval_every=40, save_every=100, progress=tmp_path / f"{name}.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None, **kw)
        assert rep["steps_done"] == 40
        return gap(m.cpu())

    alone = run("alone")
    taught = run("taught", teacher=loaded, teacher_weight=1.0, teacher_until=400)
    assert taught < alone                                                   # closer to what the teacher would write
    assert all(torch.equal(v, loaded.cpu().state_dict()[k]) for k, v in before.items())       # the teacher itself never changes
    assert not any(p.requires_grad for p in loaded.parameters())
    faded = run("faded", teacher=loaded, teacher_weight=1.0, teacher_until=0)                  # no teacher step: plain training
    assert abs(faded - alone) < 1e-6


def test_mixed_batches_and_a_teacher_go_together(tmp_path):
    torch.manual_seed(0)
    teacher = build_model(SMALL)
    bulk = np.resize(np.random.default_rng(0).integers(12, 60, size=37), 400 * 16).reshape(400, 16).astype(np.int32)
    train = np.random.default_rng(1).integers(12, 60, size=(40, 16)).astype(np.int32)
    rep = long_train(build_model(BIG), train, train[:8], steps=24, out_dir=tmp_path, meta={"tag": "t"}, batch_size=8, warmup=1,
                     eval_every=12, save_every=100, bulk=bulk, bulk_frac=0.5, carry=4, carry_share=0.5, teacher=teacher, teacher_until=12,
                     progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert rep["steps_done"] == 24 and np.isfinite(rows[-1]["val_loss"])
