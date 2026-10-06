# Released cores

Every release is frozen: the files never change and `SHA256SUMS` lists their checksums. Losses are measured on text no training run has seen (lower is better); the code exam has 79 tasks.

| Release | Parameters | Frozen | Loss, dataset | Loss, web | Code exam | Weights |
|---|---|---|---|---|---|---|
| [NOVA-10M-v1](NOVA-10M-v1/) | 9.9 M | 2026-10-02 | 3.446 | – | 34 / 79 | 32 MB |
| [NOVA-24M-v1](NOVA-24M-v1/) | 23.7 M | 2026-10-03 | 3.2877 | 3.3619 | 40 / 79 | 68 MB |
| [NOVA-24M-v2](NOVA-24M-v2/) | 23.7 M | 2026-10-04 | 3.2498 | 3.3658 | 43 / 79 | 68 MB |
| [NOVA8-24M-v1](NOVA8-24M-v1/) | 24.2 M | 2026-10-06 | 3.0061 | 3.1696 | 53 / 79 | 63 MB |
| [NOVA8-24M-v2](NOVA8-24M-v2/) | 24.2 M | 2026-10-06 | 2.9609 | 3.1562 | 53 / 79 | 63 MB |
| [NOVA8-24M-v3](NOVA8-24M-v3/) | 24.2 M | 2026-10-06 | 2.9435 | 3.1567 | 52 / 79 | 63 MB |

Use one: `python -m nova.demo --release <name> --lang en --prompt "The river"` (see [INSTALL.md](../../INSTALL.md)).

`nova_model.pt` holds the weights as 16-bit floats; the full-precision file named in `SHA256SUMS` is too large for a git repository.

Licence: free for research, experiments and other non-commercial use; commercial use needs the creator's permission ([LICENSE](../../LICENSE)).
