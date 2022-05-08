from abc import abstractmethod
from torch.nn.functional import binary_cross_entropy, l1_loss, nll_loss, \
    cross_entropy, softmin, kl_div
import torch
from torch import sigmoid, softmax, clamp, log_softmax, tensor, argmax, zeros, \
    sqrt, arange
from mmaction.models.builder import LOSSES
from mmaction.models.losses import BaseWeightedLoss
from .apn import get_correlated_progressions


@LOSSES.register_module()
class ApnBaseLoss(BaseWeightedLoss):
    def __init__(self, uncorrelated_progs='ignore'):
        super().__init__()
        self.uncorrelated_progs = uncorrelated_progs

    def _forward(self, output, progression_label, class_label):
        """Forward function.

        Args:
            output (torch.Tensor): The class score.
            progression_label (torch.Tensor): The ground truth label of progression.


        Returns:
            torch.Tensor: The returned mse loss.
        """
        if self.uncorrelated_progs == 'ignore':
            loss = self._forward_ignore(output, progression_label, class_label)
        elif self.uncorrelated_progs == 'set_zeros':
            loss = self._forward_zeros(output, progression_label, class_label)
        elif self.uncorrelated_progs == 'random':
            loss = self._forward_random(output, progression_label, class_label)
        else:
            raise ValueError(f"Unrecognized argument {self.uncorrelated_loss}")
        return loss

    @abstractmethod
    def _forward_ignore(self, output, progression_label, class_label):
        pass

    @abstractmethod
    def _forward_zeros(self, output, progression_label, class_label):
        pass

    @abstractmethod
    def _forward_random(self, output, progression_label, class_label):
        pass


@LOSSES.register_module()
class ApnMAELoss(ApnBaseLoss):
    """Mean Absolute Error Loss"""

    def _forward_ignore(self, output, progression_label, class_label):
        indexed_output = clamp(get_correlated_progressions(output, class_label),
                               max=1, min=0)
        loss = l1_loss(indexed_output, progression_label.squeeze(-1))
        return loss

    def _forward_zeros(self, output, progression_label, class_label):
        """Forward function.
        Args:
            output (torch.Tensor): The class score.
            progression_label (torch.Tensor): The ground truth label.
        Returns:
            torch.Tensor: The returned mse loss.
        """
        output = clamp(output, max=1, min=0)
        num_sample = output.shape[0]
        num_action = output.shape[1]

        batch_loss = 0
        for output_e, target_progression, target_class in zip(output,
                                                              progression_label,
                                                              class_label):
            target_output = zeros(num_action).cuda()
            target_output[target_class] = target_progression.float()
            loss_e = l1_loss(output_e, target_output, reduction='none')
            loss_e[target_class] *= (num_action - 1)
            batch_loss += loss_e.sum() / (num_action - 1)
        loss = batch_loss / num_sample
        return loss

    def _forward_random(self, output, progression_label, class_label):
        output = clamp(output, max=1, min=0)
        num_sample = output.shape[0]
        num_action = output.shape[1]
        indexed_output = clamp(get_correlated_progressions(output, class_label),
                               max=1, min=0)
        correlated_loss = l1_loss(indexed_output, progression_label.squeeze(-1))
        uncorrelated_progs_sum = output.sum() - indexed_output.sum()
        random_sum_target = tensor((num_action - 1) * num_sample * 0.5, device=output.device)
        uncorrelated_loss = l1_loss(uncorrelated_progs_sum,
                                    random_sum_target) / (
                                    (num_action - 1) * num_sample)
        loss = correlated_loss + uncorrelated_loss
        return loss

    @staticmethod
    def test_output(output):
        output = clamp(output, max=1, min=0)
        progressions = output * 100
        progressions = progressions.long()
        return progressions


@LOSSES.register_module()
class ApnCELoss(ApnBaseLoss):

    def _forward_ignore(self, output, progression_label, class_label):
        """Forward function.

        Args:
            output (torch.Tensor): The class score.
            progression_label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The returned bce loss.
        """
        output = log_softmax(output, dim=-1)
        indexed_output = get_correlated_progressions(output, class_label)
        loss = nll_loss(indexed_output, progression_label.squeeze(-1))

        return loss

    def _forward_zeros(self, output, progression_label, class_label):
        num_sample = output.shape[0]
        num_action = output.shape[1]
        output = log_softmax(output, dim=-1)
        batch_loss = 0
        for output_e, target_progression, target_class in zip(output,
                                                              progression_label,
                                                              class_label):
            target_output = zeros(num_action, dtype=torch.long).cuda()
            target_output[target_class] = target_progression
            loss_e = nll_loss(output_e, target_output, reduction='none')
            loss_e[target_class] *= (num_action - 1)
            batch_loss += loss_e.sum() / (num_action - 1)
        loss = batch_loss / num_sample

        return loss

    def _forward_random(self, output, progression_label, class_label):
        num_sample = output.shape[0]
        num_action = output.shape[1]
        num_stage = output.shape[2]
        correlated_loss = self._forward_ignore(output, progression_label,
                                               class_label)
        correlated_prog_sum = self.val_output(output, class_label,
                                              return_tensor=True).sum()
        uncorrelated_progs_sum = self.test_output(output,
                                                  return_tensor=True).sum() - correlated_prog_sum
        random_sum_target = tensor(
            (num_action - 1) * num_sample * ((num_stage - 1) / 2), device=output.device)
        uncorrelated_loss = l1_loss(uncorrelated_progs_sum,
                                    random_sum_target) / tensor(
            (num_action - 1) * num_sample * (num_stage - 1))
        loss = correlated_loss + 1 * uncorrelated_loss

        return loss

    @staticmethod
    def test_output(output):
        num_sample = output.shape[0]
        num_action = output.shape[1]
        num_stage = output.shape[2]

        output = softmax(output, dim=-1)
        arg_prog = argmax(output, dim=-1)
        arg_prog = torch.true_divide((arg_prog * 100), (num_stage - 1))

        stage_cuboid = arange(0, num_stage, device=output.device).unsqueeze(
            0).unsqueeze(0).expand(num_sample,
                                   num_action,
                                   num_stage)
        exp_prog = (output * stage_cuboid).sum(dim=-1)
        exp_prog = exp_prog * 100 / (num_stage - 1)
        return exp_prog


@LOSSES.register_module()
class ApnBCELoss(ApnBaseLoss):
    """
    Binary Cross Entropy Loss of single output
    """

    def _forward_ignore(self, output, progression_label, class_label):
        indexed_output = get_correlated_progressions(output, class_label)
        loss = cross_entropy(indexed_output.view(-1, 2),
                             progression_label.view(-1).long())

        return loss

    def _forward_zeros(self, output, progression_label, class_label):
        num_ele = output.shape[0]
        num_action = output.shape[1]
        num_stage = output.shape[2]
        batch_loss = 0
        for output_e, target_progression, target_class in zip(output,
                                                              progression_label.long(),
                                                              class_label):
            target_output = zeros(num_action, num_stage,
                                  dtype=torch.int64).cuda()
            target_output[target_class, :] = target_progression
            loss_e = cross_entropy(output_e.view(-1, 2), target_output.view(-1),
                                   reduction='none').view(num_action,
                                                          num_stage).mean(
                dim=-1)
            loss_e[target_class] *= (num_action - 1)
            batch_loss += loss_e.sum() / (num_action - 1)
        loss = batch_loss / num_ele
        return loss

    def _forward_random(self, output, progression_label, class_label):
        num_sample = output.shape[0]
        num_action = output.shape[1]
        num_stage = output.shape[2]
        batch_correlated_loss = 0
        batch_uncorrelated_progs_sum = 0
        for output_e, target_progression, target_class in zip(output,
                                                              progression_label,
                                                              class_label):
            correlated_output = output_e[target_class.squeeze()]
            correlated_loss = cross_entropy(correlated_output,
                                            target_progression.long())
            correlated_prog_sum = sigmoid((softmax(correlated_output, -1)[:,
                                           1] - 0.5) * 10).sum()  ## mimic the step function: >0.5 then 1 otherwise 0
            uncorrelated_progs_sum = sigmoid((softmax(output_e, -1)[:, :,
                                              1] - 0.5) * 10).sum() - correlated_prog_sum
            batch_uncorrelated_progs_sum += uncorrelated_progs_sum
            batch_correlated_loss += correlated_loss

        correlated_loss = batch_correlated_loss / num_sample
        random_sum_target = tensor(
            (num_action - 1) * num_sample * (num_stage / 2), device=output.device)
        uncorrelated_loss = l1_loss(batch_uncorrelated_progs_sum,
                                    random_sum_target)
        uncorrelated_loss = uncorrelated_loss / tensor(
            (num_action - 1) * num_sample * num_stage)
        loss = correlated_loss + 1 * uncorrelated_loss
        return loss

    @staticmethod
    def test_output(output):
        num_stage = output.shape[2]
        prob = softmax(output, dim=-1)
        progressions = torch.count_nonzero(prob[..., 1] > 0.5, dim=-1)
        progressions = progressions * 100 / num_stage
        return progressions


@LOSSES.register_module()
class ApnSORDLoss(ApnCELoss):

    def _forward_ignore(self, output, progression_label, class_label):
        num_samples = output.shape[0]
        num_stages = output.shape[2]

        output = log_softmax(output, dim=-1)
        indexed_output = get_correlated_progressions(output, class_label)

        stage_matrix = arange(0, num_stages,
                              device=progression_label.device).unsqueeze(
            0).expand(num_samples,
                      num_stages)
        distance = sqrt(abs(stage_matrix - progression_label).float())
        smoothed_label = softmin(distance, dim=-1)

        loss = kl_div(indexed_output, smoothed_label, reduction='batchmean')

        return loss

    def _forward_zeros(self, output, progression_label, class_label):
        num_sample = output.shape[0]
        num_action = output.shape[1]
        num_stage = output.shape[2]
        output = log_softmax(output, dim=-1)

        stage_matrix = arange(0, num_stage,
                              device=progression_label.device).unsqueeze(
            0).expand(num_action,
                      num_stage)
        batch_loss = 0
        for output_e, target_progression, target_class in zip(output,
                                                              progression_label,
                                                              class_label.squeeze(
                                                                  -1)):
            target_output = zeros(num_action, 1).cuda()
            target_output[target_class] = target_progression
            distance = sqrt(abs(stage_matrix - target_output).float())
            smoothed_label = softmin(distance, dim=-1)

            loss_e = kl_div(output_e, smoothed_label, reduction='none').sum(
                dim=-1)
            loss_e[target_class] *= (num_action - 1)
            batch_loss += loss_e.sum() / (num_action - 1)
        loss = batch_loss / num_sample
        return loss
    # random loss directly employ the one of CELOSS


@LOSSES.register_module()
class ApnCORALLoss(ApnBaseLoss):
    """Two head for prog and action class"""

    def _forward_ignore(self, output, progression_label, class_label):
        output = sigmoid(output)
        indexed_output = get_correlated_progressions(output, class_label)
        loss = binary_cross_entropy(indexed_output, progression_label)
        return loss

    def _forward_zeros(self, output, progression_label, class_label):
        num_sample = output.shape[0]
        num_action = output.shape[1]
        num_stage = output.shape[2]
        output = sigmoid(output)
        batch_loss = 0
        for output_e, target_progression, target_class in zip(output,
                                                              progression_label,
                                                              class_label):
            target_output = zeros(num_action, num_stage).cuda()
            target_output[target_class, :] = target_progression
            loss_e = binary_cross_entropy(output_e, target_output,
                                          reduction='none').mean(dim=-1)
            loss_e[target_class] *= (num_action - 1)
            batch_loss += loss_e.sum() / (num_action - 1)
        loss = batch_loss / num_sample
        return loss

    def _forward_random(self, output, progression_label, class_label):
        num_sample = output.shape[0]
        num_action = output.shape[1]
        num_stage = output.shape[2]
        output = sigmoid(output)
        batch_correlated_loss = 0
        batch_uncorrelated_progs_sum = 0
        for output_e, target_progression, target_class in zip(output,
                                                              progression_label,
                                                              class_label):
            correlated_output = output_e[target_class.squeeze()]
            correlated_loss = binary_cross_entropy(correlated_output,
                                                   target_progression)
            # mimic the step function: >0.5 then 1 otherwise 0
            correlated_prog = sigmoid((correlated_output - 0.5) * 10).sum()
            uncorrelated_progs_sum = sigmoid(
                (output_e - 0.5) * 10).sum() - correlated_prog

            batch_uncorrelated_progs_sum += uncorrelated_progs_sum
            batch_correlated_loss += correlated_loss

        correlated_loss = batch_correlated_loss / num_sample
        random_sum_target = tensor(
            (num_action - 1) * num_sample * (num_stage / 2), device=output.device)
        uncorrelated_loss = l1_loss(batch_uncorrelated_progs_sum,
                                    random_sum_target) / random_sum_target
        loss = correlated_loss + 1 * uncorrelated_loss
        return loss

    @staticmethod
    def test_output(output):
        num_stage = output.shape[2]
        output = sigmoid(output)
        progressions = torch.count_nonzero(output > 0.5, dim=-1)
        progressions = progressions * 100 / num_stage
        return progressions

# @LOSSES.register_module()
# class ApnTwoLoss(BaseWeightedLoss):
#     """Two head for prog and action class"""
#
#     def _forward(self, output, progression_label, class_label):
#         """Forward function.
#
#         Args:
#             output (torch.Tensor): The class score.
#             progression_label (torch.Tensor): The ground truth label of progression.
#
#
#         Returns:
#             torch.Tensor: The returned mse loss.
#         """
#         prog = log_softmax(output['prog'], dim=-1)
#         action = output['action']
#
#         num_samples = prog.shape[0]
#         num_stages = prog.shape[-1]
#
#         stage_matrix = arange(0, num_stages,
#                               device=progression_label.device).unsqueeze(
#             0).expand(num_samples,
#                       num_stages)
#         distance = sqrt(abs(stage_matrix - progression_label).float())
#         smoothed_label = softmin(distance, dim=-1)
#
#         prog_loss = kl_div(prog, smoothed_label, reduction='batchmean')
#
#         action_loss = cross_entropy(action, class_label.squeeze())
#
#         return {'loss_p': prog_loss, 'loss_a': action_loss}
#
#     def test_output(self, output):
#         prog = softmax(output['prog'], dim=-1)
#         action = softmax(output['action'], dim=-1).cpu().numpy()
#
#         num_sample = prog.shape[0]
#         num_stage = prog.shape[-1]
#
#         stage_matrix = arange(0, num_stage, device=prog.device).unsqueeze(
#             0).expand(num_sample, num_stage)
#
#         arg_prog = argmax(prog, dim=-1).cpu().numpy()
#         exp_prog = (prog * stage_matrix).sum(dim=-1).cpu().numpy()
#
#         return list(zip(arg_prog, exp_prog, action))
