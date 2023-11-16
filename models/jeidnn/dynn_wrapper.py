import torch
import numpy as np
from models.jeidnn.learnable_uncertainty_gate import LearnableUncGate
from enum import Enum

class TrainingPhase(Enum):
    CLASSIFIER = 1
    GATE = 2
    WARMUP = 3

class DynnWrapper(torch.nn.Module):

    def __init__(self, net, args):
        super().__init__()
        self.net = net
        self.args = args
        self.CE_IC_tradeoff = args.ce_ic_tradeoff
        self.classifiers = list(filter(lambda x: 'classifier' in x[0], list(net.named_children()))) # tuples (name, module)
        self.intermediate_heads = list(map(lambda x: x[1], self.classifiers[:-1]))
        self.init_gates()
        self.freeze_backbone_except_classifiers_and_gates()
        self.set_cost_per_exit()

    def init_gates(self):
        self.gates = torch.nn.ModuleList([
            LearnableUncGate(classifier[0]) for classifier in self.classifiers[:-1]])

    def freeze_backbone_except_classifiers_and_gates(self):
        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        total_num_parameters = sum([np.prod(p.size()) for p in model_parameters])
        # set everything to not trainable.
        for param in self.net.parameters():
            param.requires_grad = False

        for classifier in self.intermediate_heads: # keep last classifier frozen
            for param in classifier.parameters():
                param.requires_grad = True

        for gate in self.gates:
            for param in gate.parameters():
                param.requires_grad = True

        trainable_parameters = filter(lambda p: p.requires_grad,
                                      self.net.parameters())
        num_trainable_params = sum(
            [np.prod(p.size()) for p in trainable_parameters])
        print('Successfully froze network: from {} to {} trainable params.'.format(
            total_num_parameters, num_trainable_params))

    def set_cost_per_exit(self, mult_add_at_exits: list[float] = [0.097629792, 0.181669488, 0.205988448, 0.264681648]):
        normalized_cost = torch.tensor(mult_add_at_exits) / mult_add_at_exits[-1]
        self.mult_add_at_exits = (torch.tensor(mult_add_at_exits) * 1e9).tolist()
        self.normalized_cost_per_exit = normalized_cost.tolist()

    def are_all_classifiers_frozen(self):
        for inter_head in self.intermediate_heads:
            for param in inter_head.parameters():
                if param.requires_grad:
                    return False
        return True
    def get_gate_prediction(self, l, current_logits):
        return self.gates[l](current_logits)

    def forward(self, x):
        return list(self.net.forward(x)) # returns 4 values y_early3, y_att, y_cnn, y_merge in a list
