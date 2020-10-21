import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, InputXGradient, Saliency

from attribution.attribution import AttributionMixin

Input = Tuple[torch.Tensor, torch.Tensor]


class GradientMixin(AttributionMixin):
    """
    Generic interface for gradient-based Captum attributors.
    """

    def __init__(self, model: nn.Module):
        super(GradientMixin, self).__init__(model)
        self.forward_func = self._forward_func

    def _forward_func(self, x_one_hot: torch.Tensor, lengths: torch.Tensor,
                      save_output: bool = False) -> torch.Tensor:
        y = self.model(x_one_hot, lengths, save_output=save_output)
        if len(y.shape) == 1:
            return y.unsqueeze(0)
        else:
            return y

    def attribute(self, x: Input, target: int = None, **kwargs) -> List[float]:
        if target is None:
            target = int(torch.argmax(self.forward_func(*x)))

        with warnings.catch_warnings():  # Ignore requires_grad warning
            warnings.simplefilter("ignore")
            kwargs["target"] = target
            kwargs["additional_forward_args"] = x[1]
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

    def attribute(self, x: Input, target: int = None) -> List[float]:
        return super(SaliencyAttribution, self).attribute(x, target=target,
                                                          abs=False)
