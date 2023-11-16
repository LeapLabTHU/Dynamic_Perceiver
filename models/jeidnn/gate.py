from enum import Enum
from typing import Tuple
from torch import nn
from torch import Tensor


class GateType(Enum):
    UNCERTAINTY = 'unc'
    CODE = 'code'
    CODE_AND_UNC = 'code_and_unc'
    IDENTITY = 'identity'


class Gate(nn.Module):
    """Abstract class for gating"""

    def __init__(self):
        super(Gate, self).__init__()
        

    def forward(self, input: Tensor) -> Tensor:
        pass

    def inference_forward(self, input: Tensor,
                          previous_mask: Tensor) -> Tuple[Tensor, Tensor]:
        pass
