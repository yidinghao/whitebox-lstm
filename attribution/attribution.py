from typing import List

import torch
from captum.attr import Occlusion
from torch import nn
from torchtext import data as tt

from tools.data_types import Input, ClassWeights


class AttributionMixin(object):
    """
    A custom interface for Captum's attribution classes.
    """

    def __new__(cls, *args, **kwargs):
        if cls is AttributionMixin:
            raise TypeError("Can't instantiate an AttributionMixin by itself")
        return object.__new__(cls)

    def __init__(self, model: nn.Module, dataset: tt.Dataset):
        """
        A wrapper around Captum's __init__ function, which supports
        saving the fields of a dataset.

        :param model: A model
        :param dataset: A dataset
        """
        super(AttributionMixin, self).__init__(model)
        self.model = model
        fields = dataset.fields
        self.x_field = fields["x"] if "x" in fields else fields["text"]
        self.y_field = fields["y"] if "y" in fields else fields["label"]
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

    def __init__(self, model: nn.Module, dataset: tt.Dataset):
        super(OcclusionAttribution, self).__init__(model, dataset)
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

        if len(x[0].shape) == 3:
            window = (1, x[0].shape[2])
        else:
            window = (1,)
        kwargs = dict(target=target, additional_forward_args=x[1])
        rel = super(OcclusionAttribution, self).attribute(x[0], window,
                                                          **kwargs)

        if len(rel.shape) == 3:
            return list(rel.sum(2).detach().numpy()[0])
        else:
            return list(rel.squeeze().detach().numpy())
