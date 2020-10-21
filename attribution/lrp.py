from typing import List, Tuple

import numpy as np
import torch
from captum.attr import Attribution

from attribution.attribution import AttributionMixin

Input = Tuple[torch.Tensor, torch.Tensor]


class LRPAttribution(AttributionMixin, Attribution):
    """
    LRP attribution
    """

    name = "LRP"

    def attribute(self, x: Input, target: int = None, eps: float = 0.001) -> \
            List[float]:
        """
        Computes LRP scores for an input.

        :param x: An input
        :param target: Target weights
        :param eps: Stabilizer
        :return: LRP scores for the input
        """
        lrp_scores = self.forward_func.lrp(x, target=target, eps=eps)
        return list(np.sum(lrp_scores, 1))
