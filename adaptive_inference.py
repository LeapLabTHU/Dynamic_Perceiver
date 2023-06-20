import torch
import torch.nn as nn
import os
import math
import numpy as np


def dynamic_evaluate(model, test_loader, val_loader, filename, args):
    tester = Tester(model)
    # if os.path.exists(os.path.join(args.output_dir, 'logits_single.pth')):
    #     val_pred, val_target, test_pred, test_target = \
    #         torch.load(os.path.join(args.output_dir, 'logits_single.pth'))
    # else:
    val_pred, val_target = tester.calc_logit(val_loader, early_break=True)
    test_pred, test_target = tester.calc_logit(test_loader, early_break=False)
    # torch.save((val_pred, val_target, test_pred, test_target),
    #             os.path.join(args.output_dir, 'logits_single.pth'))

    # flops = torch.load(os.path.join(args.output_dir, 'flops.pth'))
    flops = np.loadtxt(f'{args.output_dir}/flops.txt')
    flops = [flops[i] for i in [0,1,2,3]]
    each_exit = False
    with open(os.path.join(args.output_dir, filename), 'w') as fout:
        probs_list = generate_distribution(each_exit=each_exit)
        for probs in probs_list:
            print('\n*****************')
            print(probs)
            acc_val, _, T = tester.dynamic_eval_find_threshold(val_pred, val_target, probs, flops)
            print(T)
            acc_test, exp_flops, acc_each_stage = tester.dynamic_eval_with_threshold(test_pred, test_target, flops, T)
            print('valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}M'.format(acc_val, acc_test, exp_flops))
            print('acc of each exit: {}'.format(acc_each_stage))
            fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))
    print('----------ALL DONE-----------')


def generate_distribution(each_exit=False):
    probs_list = []
    if each_exit:
        for i in range(4):
            probs = torch.zeros(4, dtype=torch.float)
            probs[i] = 1
            probs_list.append(probs)
    else:
        p_list = torch.zeros(34)
        for i in range(17):
            p_list[i] = (i + 4) / 20
            p_list[33 - i] = 20 / (i + 4)

        # y_early3, y_att, y_cnn, y_merge
        # 对应放缩比例
        k = [0.85, 1, 0.5, 1]
        for i in range(33):
            probs = torch.exp(torch.log(p_list[i]) * torch.range(1, 4))
            probs /= probs.sum()
            for j in range(3):
                probs[j] *= k[j]
                probs[j+1:4] = (1 - probs[0:j+1].sum()) * probs[j+1:4] / probs[j+1:4].sum()
            probs_list.append(probs)
    return probs_list


class Tester(object):
    def __init__(self, model):
        # self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader, early_break=False):
        self.model.eval()
        n_stage = 4
        logits = [[] for _ in range(n_stage)]
        targets = []
        # print('xxxxxxxxxxx111111')
        # print(len(dataloader))
        for i, (input, target) in enumerate(dataloader):
            # print(input.shape, target.shape)
            if early_break and i > 100:
                break
            targets.append(target)
            input = input.cuda()
            with torch.no_grad():
                y_early3, y_att, y_cnn, y_merge = self.model(input)
                output = [y_early3, y_att, y_cnn, y_merge]
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t)
            if i % 50 == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    def dynamic_eval_find_threshold(self, logits, targets, p, flops):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, T):
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, acc_rec / exp


if __name__ == '__main__':
    print(generate_distribution(each_exit=False))
