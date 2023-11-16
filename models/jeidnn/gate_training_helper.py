
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from enum import Enum

class GateObjective(Enum):
    CrossEntropy = "ce"
    ZeroOne = "zeroOne"
    Prob = "prob"

class InvalidLossContributionModeException(Exception):
    pass

class GateTrainingHelper:
    def __init__(self, net: nn.Module, gate_objective: GateObjective, G: int, device) -> None:
        self.net = net
        self.device  = device
        self.set_ratios([1 for _ in range(G)])
        
        self.gate_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.gate_objective = gate_objective
        if self.gate_objective == GateObjective.CrossEntropy:
            self.predictor_criterion = nn.CrossEntropyLoss(reduction='none') # Measures Cross entropy loss
        elif self.gate_objective == GateObjective.ZeroOne:
            self.predictor_criterion = self.zeroOneLoss # Measures the accuracy
        elif self.gate_objective == GateObjective.Prob:
            self.predictor_criterion = self.prob_when_correct # Measures prob of the correct class if accurate, else returns 0
    
    def zeroOneLoss(self, logits, targets):
        _, predicted = logits.max(1)
        correct = predicted.eq(targets)
        return -correct.flatten()

    def prob_when_correct(self, logits, targets):
        probs = torch.nn.functional.softmax(logits, dim=1) # get the probs
        p_max, _ = torch.topk(probs, 1) # get p max
        _, predicted = logits.max(1) # get the prediction
        correct = predicted.eq(targets)[:,None]

        prob_when_correct = correct * p_max # hadamard product, p when the prediciton is correct, else 0
        
        return -prob_when_correct.flatten()

    def set_ratios(self, pos_weights):
        self.pos_weights = torch.Tensor(pos_weights).to(self.device)
        
 

    def get_loss(self, inputs: torch.Tensor, targets: torch.tensor):
        logits = self.net.forward(inputs)
        final_logits = logits[-1]
        intermediate_losses = []
        gate_logits = []
        intermediate_logits = []
        
        optimal_exit_count_per_gate = dict.fromkeys(range(len(self.net.intermediate_heads)), 0) # counts number of times a gate was selected as the optimal gate for exiting
        
        for l, current_logits in enumerate(logits[:-1]):
            intermediate_logits.append(current_logits)
            current_gate_logits = self.net.get_gate_prediction(l, current_logits)
            gate_logits.append(current_gate_logits)
            pred_loss = self.predictor_criterion(current_logits, targets)
            ic_loss = self.net.normalized_cost_per_exit[l]
            level_loss = pred_loss + self.net.CE_IC_tradeoff * ic_loss
            level_loss = level_loss[:, None]
            intermediate_losses.append(level_loss)
        # add the final head as an exit to optimize for
        final_pred_loss = self.predictor_criterion(final_logits, targets)
        final_ic_loss = 1
        final_level_loss = final_pred_loss + self.net.CE_IC_tradeoff * final_ic_loss
        final_level_loss = final_level_loss[:, None]
        intermediate_losses.append(final_level_loss)
        
        gate_target = torch.argmin(torch.cat(intermediate_losses, dim = 1), dim = 1) # For each sample in batch, which gate should exit
        for gate_level in optimal_exit_count_per_gate.keys():
            count_exit_at_level = torch.sum(gate_target == gate_level).item()
            optimal_exit_count_per_gate[gate_level] += count_exit_at_level
        things_of_interest = {'exit_count_optimal_gate': optimal_exit_count_per_gate}
        gate_target_one_hot = torch.nn.functional.one_hot(gate_target, len(self.net.intermediate_heads) + 1)
        gate_logits = torch.cat(gate_logits, dim=1)
        loss, correct_exit_count = self._get_exit_subsequent_loss(gate_target_one_hot, gate_logits)
        things_of_interest = things_of_interest | {'intermediate_logits': intermediate_logits, 'final_logits':final_logits, 'correct_exit_count': correct_exit_count}
        return loss, things_of_interest
    
    def _get_exit_subsequent_loss(self, gate_target_one_hot, gate_logits):
        correct_exit_count = 0
        gate_target_one_hot = gate_target_one_hot[:,:-1] # remove exit since there is not associated gate
        hot_encode_subsequent = gate_target_one_hot.cumsum(dim=1)
        gate_loss = self.gate_criterion(gate_logits.flatten(), hot_encode_subsequent.double().flatten())

        # one 1/0 ratio per gate computed on a validation set
        ones_loss_multiplier = (hot_encode_subsequent.double() * self.pos_weights).flatten() # balances ones
        zeros_loss_multiplier = torch.logical_not(hot_encode_subsequent).double().flatten()
        multiplier = ones_loss_multiplier + zeros_loss_multiplier

        gate_loss = torch.mean(gate_loss * multiplier)
        # compute gate accuracies
        actual_exits_binary = torch.nn.functional.sigmoid(gate_logits) >= 0.5
        correct_exit_count += accuracy_score(actual_exits_binary.flatten().cpu(), hot_encode_subsequent.double().flatten().cpu(), normalize=False)
        
        return gate_loss, correct_exit_count