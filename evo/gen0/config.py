from dataclasses import dataclass


@dataclass
class NovaConfig:
    # Vocabulary / representation
    vocab_size: int = 16384
    d_model: int = 256
    d_state: int = 256

    # NOVA depth
    num_layers: int = 6

    # Local interaction
    conv_kernel: int = 5

    # Initial state
    learnable_initial_state: bool = False

    # Forget gate initialization
    forget_bias: float = 1.5

    # Training / sequence
    max_seq_len: int = 512

    # Special tokens
    pad_token_id: int = 0

    # Numerical settings
    dtype: str = "float32"


CONFIG = NovaConfig()
