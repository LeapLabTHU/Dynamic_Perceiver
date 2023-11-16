'''Train DYNN from checkpoint of trained backbone'''
import itertools
import os
import torch
import mlflow
from models.jeidnn.collect_metric_iter import aggregate_metrics, process_things
from models.jeidnn.utils import get_path_to_project_root#, split_dataloader_in_n
from models.jeidnn.learning_helper import LearningHelper, switch_training_phase
from models.jeidnn.log_helper import log_aggregate_metrics_mlflow
from models.jeidnn.utils import aggregate_dicts
from models.jeidnn.dynn_wrapper import TrainingPhase
import numpy as np
import pickle as pk


def train_single_epoch(args, helper: LearningHelper, device, train_loader, epoch, training_phase,
          bilevel_batch_count=20):
    print('\nEpoch: %d' % epoch)
    helper.net.train()

    metrics_dict = {}
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)

        if training_phase == TrainingPhase.WARMUP:
            #  we compute the warmup loss
            loss, things_of_interest = helper.get_warmup_loss(inputs, targets)
        else:
            if batch_idx % bilevel_batch_count == 0:
                if helper.net.are_all_classifiers_frozen(): # no need to train classifiers anymore
                    training_phase = TrainingPhase.GATE
                    print("All classifiers are frozen, setting training phase to gate")
                else:
                    metrics_dict = {}
                    training_phase = switch_training_phase(training_phase)
            loss, things_of_interest = helper.get_surrogate_loss(inputs, targets, training_phase)
        loss.backward()
        helper.optimizer.step()

        # obtain the metrics associated with the batch
        metrics_of_batch = process_things(things_of_interest, gates_count=len(helper.net.gates),
                                          targets=targets, batch_size=batch_size,
                                          cost_per_exit=helper.net.normalized_cost_per_exit)
        metrics_of_batch['loss'] = (loss.item(), batch_size)

        # keep track of the average metrics
        metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=len(helper.net.gates))

        # format the metric ready to be displayed
        log_dict = log_aggregate_metrics_mlflow(
                prefix_logger='train',
                metrics_dict=metrics_dict, gates_count=len(helper.net.gates))


        log_dict = log_dict
        mlflow.log_metrics(log_dict,
                            step=batch_idx +
                            (epoch * len(train_loader)))

      #  display_progress_bar('train', training_phase, step=batch_idx, total=len(train_loader), log_dict=log_dict)

    return metrics_dict



def evaluate(best_acc, args, helper: LearningHelper, device, init_loader, epoch, mode: str, experiment_name: str, store_results=False):
    helper.net.eval()
    metrics_dict = {}
    if mode == 'test': # we should split the data and combine at the end
        loaders = 12 #split_dataloader_in_n(init_loader, n=10)
    else:
        loaders = [init_loader]
    metrics_dicts = []
    log_dicts_of_trials = {}
    average_trials_log_dict = {}
    for loader in loaders:
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.size(0)

            loss, things_of_interest = helper.get_surrogate_loss(inputs, targets)

            # obtain the metrics associated with the batch
            metrics_of_batch = process_things(things_of_interest, gates_count=len(helper.net.gates),
                                              targets=targets, batch_size=batch_size,
                                              cost_per_exit=helper.net.mult_add_at_exits)
            metrics_of_batch['loss'] = (loss.item(), batch_size)


            # keep track of the average metrics
            metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=len(helper.net.gates))

            # format the metric ready to be displayed
            log_dict = log_aggregate_metrics_mlflow(
                    prefix_logger=mode,
                    metrics_dict=metrics_dict, gates_count=len(helper.net.gates))
           #display_progress_bar(prefix_logger=prefix_logger,training_phase=TrainingPhase.CLASSIFIER, step=batch_idx, total=len(loader), log_dict=log_dict)


        metrics_dicts.append(metrics_dict)
        for k, v in log_dict.items():
            aggregate_dicts(log_dicts_of_trials, k, v)
    for k,v in log_dicts_of_trials.items():
        average_trials_log_dict[k] = np.mean(v)

    gated_acc = average_trials_log_dict[mode+'/gated_acc']
    total_cost = average_trials_log_dict[mode+'/total_cost']
    total_cost_scaled = total_cost/1e9
    print(f'At epoch {epoch} \t Gated acc {gated_acc} \t Total cost {total_cost}')
    average_trials_log_dict[mode+'/test_acc']= gated_acc
    mlflow.log_metrics(average_trials_log_dict, step=epoch)
    # Save checkpoint.
    if gated_acc > best_acc and mode == 'val':
        print('SHOULD SAVE..')
        state = {
            'net': helper.net.state_dict(),
            'acc': gated_acc,
            'epoch': epoch,
        }
        checkpoint_path = os.path.join(get_path_to_project_root(), 'jeidnn_exploration_checkpoint')
        this_run_checkpoint_path = os.path.join(checkpoint_path, f'checkpoint_cifar100_{args.ce_ic_tradeoff}')
        if not os.path.isdir(this_run_checkpoint_path):
            os.mkdir(this_run_checkpoint_path)
        torch.save(
            state,
            os.path.join(this_run_checkpoint_path,f'ckpt_{gated_acc}.pth')
        )
        best_acc = gated_acc


    elif mode == 'test' and store_results:
        print('storing results....')
        with open(experiment_name+'_'+args.dataset+"_"+args.arch+"_"+str(args.ce_ic_tradeoff)+'_results.pk', 'wb') as file:
            pk.dump(log_dicts_of_trials, file)
    return metrics_dict, best_acc, log_dicts_of_trials

# Any action based on the validation set
def set_from_validation(learning_helper, val_metrics_dict, freeze_classifier_with_val=False, alpha_conf = 0.04, account_subsequent = False):
    exit_count_optimal_gate = val_metrics_dict['exit_count_optimal_gate'] # ({0: 0, 1: 0, 2: 0, 3: 0, 4: 6, 5: 72}, 128)
    total = exit_count_optimal_gate[1]
    # we fix the 1/0 ratios of gate tasks based on the optimal percent exit in the validation sets
    if not account_subsequent:
        pos_weights = []
        for gate, count in exit_count_optimal_gate[0].items():
            count = max(count, 0.1)
            pos_weight = (total-count) / count # #0/#1
            pos_weight = min(pos_weight, 5) # clip for stability
            pos_weights.append(pos_weight)
        learning_helper.gate_training_helper.set_ratios(pos_weights)
    else:
        pos_weights = []
        acc_ones = 0
        for gate, count in exit_count_optimal_gate[0].items():
            count = max(count, 0.1)
            with_previous = count + acc_ones
            acc_ones += count
            pos_weight = (total - with_previous) / with_previous # #0/#1
            pos_weight = min(pos_weight, 5) # clip for stability
            pos_weights.append(pos_weight)
        learning_helper.gate_training_helper.set_ratios(pos_weights)



    # ## compute the quantiles for the conformal intervals
    #
    # mixed_score, n = val_metrics_dict['gated_score']
    # scores_per_gate, n = val_metrics_dict['score_per_gate']
    # score_per_final_gate, n = val_metrics_dict['score_per_final_gate']
    #
    # all_score_per_gates, n = val_metrics_dict['all_score_per_gate']
    # all_final_score, n = val_metrics_dict['all_final_score']
    #
    # alpha_qhat_dict = compute_conf_threshold(mixed_score, scores_per_gate+[score_per_final_gate], all_score_per_gates+[all_final_score])
    #
    #
    # learning_helper.classifier_training_helper.set_conf_thresholds(alpha_qhat_dict)



def eval_baseline(args, helper: LearningHelper, val_loader, device, epoch, mode: str):
    helper.net.eval()
    metrics_dict = {}
    metrics_dicts = []
    log_dicts_of_trials = {}
    average_trials_log_dict = {}
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)

        loss, things_of_interest, _ = helper.get_warmup_loss(inputs, targets)

        # obtain the metrics associated with the batch
        metrics_of_batch = process_things(things_of_interest, gates_count=len(helper.net.gates),
                                          targets=targets, batch_size=batch_size,
                                          cost_per_exit=helper.net.mult_add_at_exits)
        metrics_of_batch['loss'] = (loss.item(), batch_size)


        # keep track of the average metrics
        metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=len(helper.net.gates))

        # format the metric ready to be displayed
        log_dict = log_aggregate_metrics_mlflow(
            prefix_logger=mode,
            metrics_dict=metrics_dict, gates_count=len(helper.net.gates))
        #display_progress_bar(prefix_logger=prefix_logger,training_phase=TrainingPhase.CLASSIFIER, step=batch_idx, total=len(loader), log_dict=log_dict)

    metrics_dicts.append(metrics_dict)
    for k, v in log_dict.items():
        aggregate_dicts(log_dicts_of_trials, k, v)
    for k,v in log_dicts_of_trials.items():
        average_trials_log_dict[k] = np.mean(v)

    # gated_acc = average_trials_log_dict[mode+'/gated_acc']
    # average_trials_log_dict[mode+'/test_acc']= gated_acc
    # mlflow.log_metrics(average_trials_log_dict, step=epoch)
    return metrics_dict, log_dicts_of_trials
