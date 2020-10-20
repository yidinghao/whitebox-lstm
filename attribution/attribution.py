from typing import List

import torch
from captum.attr import Occlusion
from torch import nn

from tools.data_types import Input


class AttributionMixin(object):
    """
    A custom interface for Captum's attribution classes.
    """

    def __new__(cls, *args, **kwargs):
        if cls is AttributionMixin:
            raise TypeError("Can't instantiate an AttributionMixin by itself")
        return object.__new__(cls)

    def __init__(self, model: nn.Module):
        """
        A wrapper around Captum's __init__ function, which supports
        saving the fields of a dataset.

        :param model: A model
        """
        super(AttributionMixin, self).__init__(model)
        self.model = model
        self.x_field = self.model.x_field
        self.y_field = self.model.y_field
        model.eval()

    def __call__(self, x_str: str, *args, **kwargs) -> List[float]:
        """
        A wrapper around Captum's attribute method.

        :param x_str: The input, represented as a token sequence
        :return: The attribution scores
        """
        return self.attribute(self.x_field.process([x_str]), *args, **kwargs)

    name = "Attribution Mixin"

    @classmethod
    def get_name(cls):
        return cls.name


class OcclusionAttribution(AttributionMixin, Occlusion):
    name = "Occlusion"

    def __init__(self, model: nn.Module):
        super(OcclusionAttribution, self).__init__(model)
        self.forward_func = self._forward_func

    def _forward_func(self, x_one_hot: torch.Tensor, lengths: torch.Tensor,
                      save_output: bool = False) -> torch.Tensor:
        y = self.model(x_one_hot, lengths, save_output=save_output)
        if len(y.shape) == 1:
            return y.unsqueeze(0)
        else:
            return y

    def attribute(self, x: Input, target: int = None) -> List[float]:
        if target is None:
            target = int(torch.argmax(self.forward_func(*x)))

        window = (1, x[0].shape[2]) if len(x[0].shape) == 3 else (1,)
        kwargs = dict(target=target, additional_forward_args=x[1])
        rel = super(OcclusionAttribution, self).attribute(x[0], window,
                                                          **kwargs)

        if len(rel.shape) == 3:
            return list(rel.sum(2).detach().numpy()[0])
        else:
            return list(rel.squeeze().detach().numpy())
