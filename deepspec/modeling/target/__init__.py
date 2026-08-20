from .common import TargetForwardResult
from .deepseek_v4_cp import (
    install_deepseek_v4_ring_context_parallel,
    install_target_context_parallel,
    ring_left_context,
)
from .online import DeepseekV4OnlineTarget


def install_qwen3_6_ring_context_parallel(target_model):
    """Lazily load optional PyTorch native CP support for Qwen3.6."""

    from .qwen3_6_cp import (
        install_qwen3_6_ring_context_parallel as install,
    )

    return install(target_model)

__all__ = [
    "TargetForwardResult",
    "DeepseekV4OnlineTarget",
    "install_deepseek_v4_ring_context_parallel",
    "install_target_context_parallel",
    "install_qwen3_6_ring_context_parallel",
    "ring_left_context",
]
