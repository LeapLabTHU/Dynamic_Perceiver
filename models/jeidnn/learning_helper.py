import torch
from models.jeidnn.dynn_wrapper import TrainingPhase
from torch import nn
import numpy as np
from models.jeidnn.classifier_training_helper import LossContributionMode, ClassifierTrainingHelper
from models.jeidnn.gate_training_helper import GateTrainingHelper
from models.jeidnn.gate_training_helper import GateObjective

criterion = nn.CrossEntropyLoss()

class LearningHelper:
    def __init__(self, net, optimizer, args, device, classifier_criterion_func = nn.CrossEntropyLoss) -> None:
        self.net = net
        self.optimizer = optimizer
        self._init_classifier_training_helper(device, classifier_criterion_func)
        self._init_gate_training_helper(device)

    def _init_classifier_training_helper(self, device, classifier_criterion_func) -> None:
        self.loss_contribution_mode = LossContributionMode.BOOSTED
        self.classifier_training_helper = ClassifierTrainingHelper(self.net, self.loss_contribution_mode, len(self.net.gates), device, classifier_criterion_func)
    
    def _init_gate_training_helper(self, device) -> None:
        self.gate_training_helper = GateTrainingHelper(self.net, GateObjective.CrossEntropy, len(self.net.gates), device)

    def get_surrogate_loss(self, inputs, targets, training_phase=None):
        if self.net.training:
            self.optimizer.zero_grad()
            if training_phase == TrainingPhase.CLASSIFIER:
                return self.classifier_training_helper.get_loss(inputs, targets)
            elif training_phase == TrainingPhase.GATE:
                return self.gate_training_helper.get_loss(inputs, targets)
        else:
            with torch.no_grad():
                classifier_loss, things_of_interest = self.classifier_training_helper.get_loss(inputs, targets)
                gate_loss, things_of_interest_gate = self.gate_training_helper.get_loss(inputs, targets)
                loss = (gate_loss + classifier_loss) / 2
                things_of_interest.update(things_of_interest_gate)
                return loss, things_of_interest


    def get_warmup_loss(self, inputs, targets, criterion = nn.CrossEntropyLoss()):
        self.optimizer.zero_grad()
        logits= self.net(inputs)
        final_logits = logits[-1]
        loss = criterion(final_logits, targets)  # the grad_fn of this loss should be None if frozen
        num_gates = len(self.net.gates) + 1
        for l, intermediate_logit in enumerate(logits[:-1]):
            intermediate_loss = criterion(intermediate_logit, targets)
            loss += (num_gates - l) * intermediate_loss # we scale the gradient by G-l => early gates have bigger gradient
        things_of_interest = {
            'intermediate_logits': logits[:-1],
            'final_logits': final_logits}
        return loss, things_of_interest

def freeze_backbone(network, excluded_submodules: list[str]):
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    total_num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    # set everything to not trainable.
    for param in network.module.parameters():
        param.requires_grad = False

    for submodule_attr_name in excluded_submodules:  # Unfreeze excluded submodules to be trained.
        for submodule in getattr(network.module, submodule_attr_name):
            for param in submodule.parameters():
                param.requires_grad = True

    trainable_parameters = filter(lambda p: p.requires_grad,
                                  network.parameters())
    num_trainable_params = sum(
        [np.prod(p.size()) for p in trainable_parameters])
    print('Successfully froze network: from {} to {} trainable params.'.format(
        total_num_parameters, num_trainable_params))


def switch_training_phase(current_phase):
    if current_phase == TrainingPhase.GATE:
        return TrainingPhase.CLASSIFIER
    elif current_phase == TrainingPhase.CLASSIFIER:
        return TrainingPhase.GATE
    elif current_phase == TrainingPhase.WARMUP:
        return TrainingPhase.GATE
