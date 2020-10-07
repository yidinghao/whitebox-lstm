"""
This module contains a wrapper around nn.Linear to use as a decoder for
an LSTM classifier.
"""
import numpy as np
import torch
import torch.nn as nn

import models.lrp_modules as lrpnn


class RNNLinearDecoder(nn.Module, lrpnn.LRPModule):
    """
    A simple linear decoder.
    """

    def __init__(self, rnn_output_size: int, logit_size: int,
                 bidirectional: bool = False):
        """
        Initializes a linear decoder.

        :param rnn_output_size: The hidden size of the RNN/LSTM
        :param logit_size: The size of the decoder's output
        :param bidirectional: Whether the LSTM is bidirectional
        """
        super(RNNLinearDecoder, self).__init__()
        self._output_size = logit_size
        if bidirectional:
            self._input_size = 2 * rnn_output_size
        else:
            self._input_size = rnn_output_size

        self._linear = lrpnn.LRPLinear(self._input_size, self._output_size)

    def forward(self, h_t: torch.Tensor):
        return self._linear(h_t)

    def lrp_forward(self, h_t: np.ndarray):
        self._linear.lrp_forward(h_t)

    def lrp_backward(self, rel_y: np.ndarray,
                     eps: float = 0.001) -> np.ndarray:
        return self._linear.lrp_backward(rel_y, eps=eps)

    @property
    def lrp_output(self):
        return self._linear.lrp_output
