import warnings
from typing import List

import torch
import torch.nn as nn
import torchtext.data as tt
from captum.attr import IntegratedGradients, InputXGradient, Saliency

from attribution.attribution import AttributionMixin
from tools.data_types import Input, ClassWeights


class GradientMixin(AttributionMixin):
    """
    Generic interface for gradient-based Captum attributors.
    """

    def __init__(self, model: nn.Module, dataset: tt.Dataset):
        super(GradientMixin, self).__init__(model, dataset)
        self.forward_func = self._forward_func

    def _forward_func(self, x_one_hot: torch.Tensor, lengths: torch.Tensor,
                      save_output: bool = False) -> torch.Tensor:
        y = self.model(x_one_hot, lengths, save_output=save_output)
        if len(y.shape) == 1:
            return y.unsqueeze(0)
        else:
            return y

    def attribute(self, x: Input, target: ClassWeights = None) -> List[float]:
        if target is None:
            target = int(torch.argmax(self.forward_func(*x)))

        elif isinstance(target, dict):
            scores = {c: self.attribute(x, target=c) for c in target}
            final_scores = []
            for c, score_list in scores.items():
                weighted_scores = [s * target[c] for s in score_list]
                if len(final_scores) == 0:
                    final_scores = weighted_scores
                else:
                    for i in range(len(final_scores)):
                        final_scores[i] += weighted_scores[i]

            return final_scores

        with warnings.catch_warnings():  # Ignore requires_grad warning
            warnings.simplefilter("ignore")
            kwargs = dict(target=target, additional_forward_args=x[1])
            rel = super(GradientMixin, self).attribute(x[0], **kwargs)

        return list(rel.sum(2).detach().numpy()[0])


class IGAttribution(GradientMixin, IntegratedGradients):
    """
    Integrated Gradients
    """
    name = "Integrated Gradients"


class GxIAttribution(GradientMixin, InputXGradient):
    """
    Gradient times Input
    """
    name = "Gradient * Input"


class SaliencyAttribution(GradientMixin, Saliency):
    """
    Saliency
    """
    name = "Saliency"
