from torch.nn.functional import binary_cross_entropy, l1_loss
from torch import sigmoid, tensor, zeros
from mmaction.models.builder import LOSSES
from mmaction.models.losses import BaseWeightedLoss
from .apn import get_correlated_progressions


@LOSSES.register_module()
class ApnCORALLoss(BaseWeightedLoss):
    """Two head for prog and action class"""

    def __init__(self, uncorrelated_progs='random'):
        super().__init__()
        self.uncorrelated_progs = uncorrelated_progs

    def _forward(self, output, progression_label, class_label):
        if self.uncorrelated_progs == 'ignore':
            loss = self._forward_ignore(output, progression_label, class_label)
        elif self.uncorrelated_progs == 'set_zeros':
            loss = self._forward_zeros(output, progression_label, class_label)
        elif self.uncorrelated_progs == 'random':
            loss = self._forward_random(output, progression_label, class_label)
        else:
            raise ValueError(f"Unrecognized argument {self.uncorrelated_loss}")
        return loss

    @staticmethod
    def _forward_ignore(output, progression_label, class_label):
        output = sigmoid(output)
        indexed_output = get_correlated_progressions(output, class_label)
        loss = binary_cross_entropy(indexed_output, progression_label)
        return loss

    @staticmethod
    def _forward_zeros(output, progression_label, class_label):
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

    @staticmethod
    def _forward_random(output, progression_label, class_label):
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
