import torch
import torch.nn as nn

from nova.config import CONFIG
from nova.model import NovaModel
from nova.blocks_reference import NovaBlock as ReferenceNovaBlock

from custom_model_benchmark import CustomNovaModel


DEVICE = "cuda"

torch.manual_seed(20260925)
torch.cuda.manual_seed_all(20260925)


def main():

    print("=" * 70)
    print("NOVA CUSTOM SCAN — LAYER-BY-LAYER GRADIENT TEST")
    print("=" * 70)

    reference = NovaModel(CONFIG).to(DEVICE)
    custom = CustomNovaModel(CONFIG).to(DEVICE)

    custom.load_state_dict(reference.state_dict())

    input_ids = torch.randint(
        0,
        CONFIG.vocab_size,
        (2, 64),
        device=DEVICE,
    )

    targets = torch.randint(
        0,
        CONFIG.vocab_size,
        (2, 64),
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Capture block outputs
    # --------------------------------------------------------

    ref_outputs = []
    custom_outputs = []

    ref_hooks = []
    custom_hooks = []

    def make_hook(storage):

        def hook(module, inp, out):

            storage.append(
                out[0]
            )

        return hook

    for block in reference.blocks:
        ref_hooks.append(
            block.register_forward_hook(
                make_hook(ref_outputs)
            )
        )

    for block in custom.blocks:
        custom_hooks.append(
            block.register_forward_hook(
                make_hook(custom_outputs)
            )
        )

    reference.zero_grad(set_to_none=True)
    custom.zero_grad(set_to_none=True)

    ref_logits, _ = reference(input_ids)
    custom_logits, _ = custom(input_ids)

    ref_loss = nn.functional.cross_entropy(
        ref_logits.reshape(-1, ref_logits.size(-1)),
        targets.reshape(-1),
    )

    custom_loss = nn.functional.cross_entropy(
        custom_logits.reshape(-1, custom_logits.size(-1)),
        targets.reshape(-1),
    )

    ref_loss.backward()
    custom_loss.backward()

    print()
    print("FORWARD DIFFERENCE PER BLOCK")
    print("-" * 70)

    for i, (a, b) in enumerate(
        zip(ref_outputs, custom_outputs)
    ):

        diff = (
            a.detach() - b.detach()
        ).abs()

        print(
            f"block {i}: "
            f"max={diff.max().item():.10e} "
            f"mean={diff.mean().item():.10e}"
        )

    print()
    print("GRADIENT DIFFERENCE PER BLOCK")
    print("-" * 70)

    for i, (ref_block, custom_block) in enumerate(
        zip(reference.blocks, custom.blocks)
    ):

        max_diff = 0.0
        mean_diff = 0.0
        count = 0

        for (
            (name_r, param_r),
            (name_c, param_c),
        ) in zip(
            ref_block.named_parameters(),
            custom_block.named_parameters(),
        ):

            if param_r.grad is None or param_c.grad is None:
                continue

            diff = (
                param_r.grad
                - param_c.grad
            ).abs()

            max_diff = max(
                max_diff,
                diff.max().item()
            )

            mean_diff += diff.mean().item()
            count += 1

        mean_diff /= max(count, 1)

        print(
            f"block {i}: "
            f"max={max_diff:.10e} "
            f"mean={mean_diff:.10e}"
        )

    print()
    print("EMBEDDING GRADIENT")

    diff = (
        reference.embedding.weight.grad
        - custom.embedding.weight.grad
    ).abs()

    print(
        "max :",
        diff.max().item()
    )

    print(
        "mean:",
        diff.mean().item()
    )

    for h in ref_hooks:
        h.remove()

    for h in custom_hooks:
        h.remove()


if __name__ == "__main__":
    main()
