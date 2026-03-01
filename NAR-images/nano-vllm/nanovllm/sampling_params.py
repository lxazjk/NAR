from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = 0
    top_p: float = 1.0
    cfg_scale: float = 1.0
    cfg_interval: int = -1

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"


@dataclass
class NARSamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    cfg_scale: float = 1.0
    cfg_interval: int = -1
    block_size: int = 256
    cls_token_num: int = 1
    model_type: str = 'c2i'
    
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert self.model_type in ['c2i', 't2i'], "model_type must be 'c2i' or 't2i'"
