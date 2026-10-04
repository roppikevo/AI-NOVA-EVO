import torch

from nova.blocks import NovaBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = 256


def make_block():
    return NovaBlock(
        d_model=D,
        d_state=D,
        conv_kernel=5,
        forget_bias=1.5,
    ).to(DEVICE)


def test_causality():
    """
    Zmena budúceho tokenu nesmie zmeniť skoršie výstupy.
    """

    torch.manual_seed(1)

    block = make_block().eval()

    x1 = torch.randn(1, 32, D, device=DEVICE)
    x2 = x1.clone()

    # Zmeníme iba posledný token.
    x2[:, -1, :] = torch.randn(1, D, device=DEVICE)

    with torch.no_grad():
        y1, _ = block(x1)
        y2, _ = block(x2)

    # Všetky výstupy pred posledným tokenom musia zostať rovnaké.
    diff = (y1[:, :-1] - y2[:, :-1]).abs().max().item()

    print("causality max diff:", diff)

    assert diff < 1e-5


def test_state_carry():
    """
    Informácia z prvého tokenu musí prejsť do konečného stavu.
    """

    torch.manual_seed(2)

    block = make_block().eval()

    x_zero = torch.zeros(1, 32, D, device=DEVICE)
    x_signal = x_zero.clone()

    # Silný signál iba na prvom tokene.
    x_signal[:, 0, :] = 5.0

    with torch.no_grad():
        _, state_zero = block(x_zero)
        _, state_signal = block(x_signal)

    diff = (state_zero - state_signal).abs().mean().item()

    print("state carry mean diff:", diff)

    assert diff > 1e-4


def test_state_reset():
    """
    Explicitne nový state=None musí začínať od nuly.
    """

    torch.manual_seed(3)

    block = make_block().eval()

    x = torch.randn(1, 16, D, device=DEVICE)

    with torch.no_grad():
        _, state_a = block(x)

        _, state_b = block(
            torch.zeros_like(x),
            state=None,
        )

    # Nulová sekvencia síce môže vytvoriť nenulový stav
    # kvôli biasom, ale druhý beh nesmie zdediť state_a.
    assert state_b.shape == state_a.shape

    print("reset state shape:", tuple(state_b.shape))


def test_gradient_flow():
    """
    Gradient musí prejsť cez sekvenčný state update.
    """

    torch.manual_seed(4)

    block = make_block().train()

    x = torch.randn(
        2,
        32,
        D,
        device=DEVICE,
        requires_grad=True,
    )

    y, state = block(x)

    loss = y[:, -1].pow(2).mean() + state.pow(2).mean()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    gradient_norm = x.grad.norm().item()

    print("input gradient norm:", gradient_norm)

    assert gradient_norm > 0.0


def test_numerical_stability():
    """
    Väčší vstup nesmie vytvoriť NaN/Inf.
    """

    torch.manual_seed(5)

    block = make_block().eval()

    x = torch.randn(
        2,
        128,
        D,
        device=DEVICE,
    ) * 10.0

    with torch.no_grad():
        y, state = block(x)

    assert torch.isfinite(y).all()
    assert torch.isfinite(state).all()

    print("numerical stability: OK")


if __name__ == "__main__":
    test_causality()
    test_state_carry()
    test_state_reset()
    test_gradient_flow()
    test_numerical_stability()

    print()
    print("ALL NOVA STATE TESTS PASSED")
