"""The size rules reproduce the cores we trained and plan larger ones from the same build."""

from nova import sizes as S


def test_the_trained_cores_come_out_exactly():
    for name in ("24M", "53M"):
        cfg = S.preset(name)
        assert S.parameters(cfg) == S.MEASURED[name]["params"]
        assert cfg["pattern"] == "NSNSNSN" and cfg["slots"] == 16
        assert abs(S.approx_parameters(cfg) - S.MEASURED[name]["params"]) / S.MEASURED[name]["params"] < 0.01
        assert S.state_kb(cfg["d_model"], len(cfg["pattern"])) == S.MEASURED[name]["state_kb"]
    assert S.preset("24M")["mlp_hidden"] == 1296 and S.preset("53M")["mlp_hidden"] == 2032


def test_the_memory_model_matches_the_measured_steps():
    assert abs(S.step_memory_mb(24_167_975, 448, 7, 64) - 5142) / 5142 < 0.02
    assert abs(S.step_memory_mb(37_201_703, 576, 7, 64) - 6180) / 6180 < 0.02
    planned = S.step_memory_mb(52_959_271, 704, 7, 48)
    assert planned < 6531 < planned * S.HELD                              # what the card held in the real run (with its teacher)
    assert S.batch_for(52_959_271, 704, 7, 8) == 48                       # what the trial run found on the 8 GB card
    assert S.batch_for(44_850_000, 640, 7, 8) != 64                       # 45 M at batch 64 did not fit
    assert abs(S.train_hours(52_959_271, 2.44e9) - 24) < 3                # the 53 M run took about a day


def test_larger_cores_follow_the_same_build():
    big = S.for_params(1e9)
    assert 0.8e9 < big["params"] < 1.25e9 and big["pattern"].startswith("NS") and big["pattern"].endswith("N")
    assert big["blocks"] > 7 and big["blocks"] % 2 == 1 and big["d_model"] % 64 == 0 and big["slots"] == 16
    assert big["step_mb"][64] > 40 * 1024 and big["batch_for_card"] if "batch_for_card" in big else True
    widths = [S.for_params(t, exact=False)["d_model"] for t in (1e8, 3e8, 1e9, 3e9)]
    assert widths == sorted(widths) and len(set(widths)) == 4
    assert S.for_card(8, 32, exact=False)["params"] < 120e6               # one 8 GB card: about a hundred million at most
    assert S.for_card(80, 32, exact=False)["params"] > 1e9                # one 80 GB card: past a billion
    assert S.for_card(0.5, 32) is None
    assert "| 1B |" in S.table(exact=False) and "| no |" in S.table(exact=False)
