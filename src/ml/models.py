"""Model definitions for ML experiments."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Basic placeholder model configuration."""

    input_dim: int
    output_dim: int


def build_model(config: ModelConfig) -> dict[str, int]:
    """Build a placeholder model descriptor."""
    return {
        "input_dim": config.input_dim,
        "output_dim": config.output_dim,
    }
