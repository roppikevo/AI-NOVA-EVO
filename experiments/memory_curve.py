import torch

from nova.blocks import NovaBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D = 256
SIGNAL = 5.0

# Vzdialenosť od signálu.
DISTANCES = [1, 4, 8, 16, 32, 64, 128, 256]

# Budeme porovnávať rôzne počiatočné biasy forget gate.
FORGET_BIASES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def measure_memory(forget_bias: float, distance: int) -> float:
    torch.manual_seed(1234)

    block = NovaBlock(
        d_model=D,
        d_state=D,
        conv_kernel=5,
        forget_bias=forget_bias,
    ).to(DEVICE).eval()

    # Dĺžka sekvencie musí byť väčšia ako vzdialenosť signálu.
    seq_len = distance + 1

    x_zero = torch.zeros(
        1,
        seq_len,
        D,
        device=DEVICE,
    )

    x_signal = x_zero.clone()

    # Signál vložíme na prvý token.
    x_signal[:, 0, :] = SIGNAL

    with torch.no_grad():
        _, state_zero = block(x_zero)
        _, state_signal = block(x_signal)

    # Rozdiel medzi stavom bez signálu a so signálom.
    difference = (
        state_signal - state_zero
    ).abs().mean().item()

    return difference


def main():
    print("=" * 72)
    print("NOVA-0 MEMORY CURVE")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Signal: {SIGNAL}")
    print()

    print(
        "forget_bias".ljust(14),
        end=""
    )

    for distance in DISTANCES:
        print(
            f"{distance:>10}",
            end=""
        )

    print()
    print("-" * 72)

    for bias in FORGET_BIASES:
        print(
            f"{bias:<14.1f}",
            end=""
        )

        for distance in DISTANCES:
            value = measure_memory(
                forget_bias=bias,
                distance=distance,
            )

            print(
                f"{value:10.6f}",
                end=""
            )

        print()

    print()
    print("=" * 72)
    print("Interpretation:")
    print("Higher value = more signal retained in final state.")
    print("=" * 72)


if __name__ == "__main__":
    main()
