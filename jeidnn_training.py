# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

import torch.nn.functional as F

def train_one_epoch_earlyExit(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,
                    loss_cnn_factor=0.25,
                    loss_att_factor=0.25,
                    loss_merge_factor=0.5,
                    
                    with_kd=False,
                    T_kd=4.0,
                    alpha_kd=0.5,
                    criterion_distill=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # if data_iter_step >= 50:
        #     break
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                y_early3, y_att, y_cnn, y_merge = model(samples)
                loss_cnn, loss_att, loss_merge, loss_early3 = criterion(y_cnn, targets), criterion(y_att, targets), criterion(y_merge, targets) if criterion_distill is None else criterion_distill(samples, y_merge, targets), criterion(y_early3, targets)
        else: # full precision
            y_early3, y_att, y_cnn, y_merge,  = model(samples)
            loss_cnn, loss_att, loss_merge, loss_early3 = criterion(y_cnn, targets), criterion(y_att, targets),  criterion(y_merge, targets) if criterion_distill is None else criterion_distill(samples, y_merge, targets), criterion(y_early3, targets)

        loss = loss_cnn_factor*loss_cnn + loss_att_factor*(loss_att+loss_early3) + loss_merge_factor*loss_merge
        
        if with_kd:
            out_teacher = y_merge.detach()
        
            kd_loss = F.kl_div(F.log_softmax(y_early3/T_kd, dim=1),F.softmax(out_teacher/T_kd, dim=1), reduction='batchmean') * T_kd**2 + \
                    F.kl_div(F.log_softmax(y_att/T_kd, dim=1),F.softmax(out_teacher/T_kd, dim=1), reduction='batchmean') * T_kd**2 + \
                    F.kl_div(F.log_softmax(y_cnn/T_kd, dim=1),F.softmax(out_teacher/T_kd, dim=1), reduction='batchmean') * T_kd**2

            loss += alpha_kd * kd_loss
        
        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc_cnn = (y_cnn.max(-1)[-1] == targets).float().mean()
            class_acc_att = (y_att.max(-1)[-1] == targets).float().mean()
            class_acc_merge = (y_merge.max(-1)[-1] == targets).float().mean()
        else:
            class_acc_cnn, class_acc_att, class_acc_merge = None, None, None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc_merge)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)



        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        # print(str(metric_logger))
        # assert(0==1)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            # log_writer.update(class_acc_cnn=class_acc_cnn, head="loss")
            # log_writer.update(class_acc_att=class_acc_att, head="loss")
            log_writer.update(class_acc=class_acc_merge, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc_cnn:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc_cnn': class_acc_cnn}, commit=False)
            if class_acc_att:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc_att': class_acc_att}, commit=False)
            if class_acc_merge:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc_merge': class_acc_merge}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_open_epoch_jeidnn(model: torch.nn.Module, criterion: torch.nn.Module,
                            data_loader: Iterable, optimizer: torch.optim.Optimizer,
                            device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                            model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                            wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                            num_training_steps_per_epoch=None, update_freq=None, use_amp=False,
                            loss_cnn_factor=0.25,
                            loss_att_factor=0.25,
                            loss_merge_factor=0.5,

                            with_kd=False,
                            T_kd=4.0,
                            alpha_kd=0.5,
                            criterion_distill=None):

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_earlyExit(data_loader, model, device, use_amp=False,
                    loss_cnn_factor=0.25,
                    loss_att_factor=0.25,
                    loss_merge_factor=0.5):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                y_early3, y_att, y_cnn, y_merge,  = model(images)
                loss_cnn, loss_att, loss_merge, loss_early3 = criterion(y_cnn, target), criterion(y_att, target),  criterion(y_merge, target), criterion(y_early3, target)
        else:
            y_early3, y_att, y_cnn, y_merge,  = model(images)
            loss_cnn, loss_att, loss_merge, loss_early3 = criterion(y_cnn, target), criterion(y_att, target),  criterion(y_merge, target), criterion(y_early3, target)

        loss = loss_cnn_factor*loss_cnn + loss_att_factor*(loss_att+loss_early3) + loss_merge_factor*loss_merge

        acc1_cnn, acc5_cnn = accuracy(y_cnn, target, topk=(1, 5))
        acc1_att, acc5_att = accuracy(y_att, target, topk=(1, 5))
        acc1_early3, acc5_early3 = accuracy(y_early3, target, topk=(1, 5))
        acc1_merge, acc5_merge = accuracy(y_merge, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1_cnn'].update(acc1_cnn.item(), n=batch_size)
        metric_logger.meters['acc1_att'].update(acc1_att.item(), n=batch_size)
        metric_logger.meters['acc1_early3'].update(acc1_early3.item(), n=batch_size)
        metric_logger.meters['acc1_merge'].update(acc1_merge.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1_early3 {top1_early3.global_avg:.3f}  \
             Acc@1_merge {top1_merge.global_avg:.3f}  \
             Acc@1_att {top1_att.global_avg:.3f}  \
             Acc@1_cnn {top1_cnn.global_avg:.3f}  \
             loss {losses.global_avg:.3f}'
          .format(top1_cnn=metric_logger.acc1_cnn, 
          top1_att=metric_logger.acc1_att,
          top1_early3=metric_logger.acc1_early3,
          top1_merge=metric_logger.acc1_merge, 
          losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
