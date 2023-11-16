from typing import Tuple
import torch
from torch import Tensor
from models.jeidnn.gate import Gate, GateType
from models.jeidnn.metrics_utils import compute_detached_uncertainty_metrics

class LearnableUncGate(Gate):
    def __init__(self, classifier_id):
        super(Gate, self).__init__()
        self.classifier_id = classifier_id
        self.gate_type = GateType.UNCERTAINTY
        self.dim = 4
        self.linear = torch.nn.Linear(self.dim, 1)

    def forward(self, logits: Tensor) -> (Tensor):
        p_maxes, entropies, _, margins, entropy_pows = compute_detached_uncertainty_metrics(logits, None)
        p_maxes = torch.tensor(p_maxes)[:, None]
        entropies = torch.tensor(entropies)[:, None]
        margins = torch.tensor(margins)[:, None]
        entropy_pows = torch.tensor(entropy_pows)[:, None]
        uncertainty_metrics = torch.cat((p_maxes, entropies, margins, entropy_pows), dim = 1)
        uncertainty_metrics = uncertainty_metrics.to(logits.device)
        return self.linear(uncertainty_metrics)

    def inference_forward(self, input: Tensor, previous_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns 2 equal-size tensors, the prediction tensor and a tensor containing the indices of predictions
        :param input: The softmax logits of the classifier
        """
        input = torch.mul(
            torch.logical_not(previous_mask).to('cuda').float()[:, None],
            input
        )
        max_probs = input.max(dim = 1)
        idx_preds_above_threshold = torch.flatten((max_probs.values > self.threshold).nonzero())
        confident_preds = torch.index_select(input, 0, idx_preds_above_threshold)
        mask = torch.zeros(input.shape[0], dtype=torch.bool)
        mask[idx_preds_above_threshold] = True
        return confident_preds, mask # 1 means early exit, 0 means propagate downstream

    def get_flops(self, num_classes):
        # compute flops for preprocssing of input and then for linear layer.
        p_max_flops = num_classes # comparison across the logits
        margin_flops = num_classes + 1 # compare top1 with top2
        entropy_flops = num_classes * 2 # compute entropy p log p then sum those values
        entropy_pow_flops = num_classes * 5 # 1 for raising to power, 1 for computing normalizing denom, 1 for scaling each pow then 2 for entropy computation
        linear_flops = self.dim + 1 # dim + bias
        return p_max_flops + margin_flops + entropy_flops + entropy_pow_flops + linear_flops
