"""
This is an architecture for a standard unidirectional LSTM classifier.
"""

import numpy as np
import torch
from torch import nn
from torch.nn.utils import rnn as rnnutils
from torchtext import data as tt

from models import lrp_modules as lrpnn
from models.decoder import RNNLinearDecoder
from tools.data_types import Input


class LSTMClassifier(nn.Module):
    """
    A basic RNN model with no embedding layer. This model consists of a
    unidirectional LSTM along with a linear decoding layer.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        """
        Constructor for an LSTMClassifier.

        :param input_size: The size of model inputs
        :param output_size: The size of model outputs
        :param hidden_size: The size of the hidden state and cell state
            vectors
        """
        super(LSTMClassifier, self).__init__()

        self._lstm = lrpnn.LRPLSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   batch_first=True)

        self._decoder = RNNLinearDecoder(hidden_size, output_size)

        # Individual layer outputs
        self._lstm_out = None
        self._lstm_out_last = None
        self._lin_out = None

    @classmethod
    def from_dataset(cls, dataset: tt.Dataset, hidden_size: int,
                     num_layers: int = 1):
        """
        Constructs an LSTMClassifier using parameters extracted from a
        Dataset object.

        :param dataset: A dataset
        :param hidden_size: The number of hidden units in the LSTM
        :param num_layers: The number of layers in the LSTM

        :return: An LSTMClassifier with input and output sizes given by
            the sizes of the input and output vocab of the Dataset,
            respectively
        """
        input_size = len(dataset.fields["x"].vocab)
        output_size = len(dataset.fields["y"].vocab)
        return cls(input_size, output_size, hidden_size, num_layers=num_layers)

    def forward(self, x_one_hot: torch.Tensor, lengths: torch.Tensor,
                save_output: bool = False) -> torch.Tensor:
        """
        The standard PyTorch forward pass. The LSTM takes packed inputs.

        :param x_one_hot: A batch of inputs
        :param lengths: The lengths of the inputs
        :param save_output: If True, the output will be saved
        :return: The model output
        """
        if len(x_one_hot.shape) == 2:
            x_one_hot = x_one_hot.unsqueeze(0)

        x_packed = rnnutils.pack_padded_sequence(x_one_hot, lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)

        lstm_out_packed, (lstm_out_last, _) = self._lstm(x_packed)
        y_hat = self._decoder(lstm_out_last[-1])

        if save_output:
            lstm_out, _ = rnnutils.pad_packed_sequence(lstm_out_packed,
                                                       batch_first=True)
            self._lstm_out = lstm_out
            self._lstm_out_last = lstm_out_last
            self._lin_out = y_hat

        return y_hat

    """ LRP """

    def lrp(self, x: Input, target: int = None, eps: float = 0.001) -> \
            np.ndarray:
        """
        Computes relevance scores for an input using LRP.

        :param x: A single input, as a batch of size 1

        :param target: Relevance will be propagated from each index of
             the output appearing in this set. If this set is empty,
             then relevance will be propagated from the index with the
             highest logit score

        :param eps: The LRP stabilizer

        :return: The relevance scores of all components of the input
        """
        x_one_hot = x[0].squeeze(0)
        self._lstm.lrp_forward(x_one_hot.detach().numpy())
        self._decoder.lrp_forward(self._lstm.lrp_output)

        # Initialize relevance
        if target is None:
            target = np.argmax(self._decoder.lrp_output, 0)

        rel_y = np.zeros(len(self._decoder.lrp_output))
        rel_y[target] = self._decoder.lrp_output[target]

        # Backward pass
        hidden_size = self._lstm.hidden_size
        rel_h_last = self._decoder.lrp_backward(rel_y, eps=eps)

        rel_h = np.zeros((len(x_one_hot), hidden_size))
        rel_h[-1] = rel_h_last

        return self._lstm.lrp_backward(rel_h, eps=eps)
