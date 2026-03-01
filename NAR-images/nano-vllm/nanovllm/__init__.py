from nanovllm.sampling_params import SamplingParams, NARSamplingParams
from nanovllm.config import Config, NARConfig, create_proximity_mask

# LLM/NARLLM require optional model deps (e.g. Qwen3Config in transformers).
try:  # pragma: no cover - optional import
    from nanovllm.llm import LLM, NARLLM
except Exception as exc:  # pylint: disable=broad-except
    _LLM_IMPORT_ERROR = exc

    class _MissingLLM:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LLM/NARLLM unavailable. Install compatible transformers "
                "or use NARPagedLLM for NAR image generation."
            ) from _LLM_IMPORT_ERROR

    LLM = _MissingLLM  # type: ignore[assignment]
    NARLLM = _MissingLLM  # type: ignore[assignment]

# New NAR backend that uses paged KV + FlexAttention.
from nanovllm.inference.nar_vllm import NARPagedLLM, NARPagedConfig, NARContinuousBatcher
