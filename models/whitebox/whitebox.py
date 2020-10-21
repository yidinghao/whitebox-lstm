from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torchtext.data as tt

from models.lstm import LSTMClassifier

Weights = Tuple[np.ndarray, np.ndarray, np.ndarray]


class WhiteBoxRNN(LSTMClassifier, ABC):
    """
    Abstract class for implementing a white-box network. A white-box
    network is defined by implementing _forget_gate, _cell_state_update,
    _input_gate, _output_gate, and _decoder_weights.
    """

    def __init__(self, x_field: tt.Field, y_field: tt.Field, hidden_size: int,
                 m: float = 50.):
        """
        WhiteBoxRNN constructor.

        :param x_field: A field used to tokenize model inputs
        :param y_field: A field used to decode model outputs
        :param hidden_size: The hidden size of the LSTM
        :param m: The constant used to saturate gates
        """
        x_size = len(x_field.vocab)
        y_size = len(y_field.vocab)
        super(WhiteBoxRNN, self).__init__(x_size, y_size, hidden_size)

        self.m = m
        self._hidden_size = hidden_size
        self.x_stoi = x_field.vocab.stoi
        self.y_stoi = y_field.vocab.stoi

        self.x_field = x_field
        self.y_field = y_field

        self._set_params()

    def _set_params(self):
        wfh, wfx, bf = self._forget_gate
        wch, wcx, bc = self._cell_state_update
        wih, wix, bi = self._input_gate
        woh, wox, bo = self._output_gate
        w, b = self._decoder_weights

        wh_tensor = torch.tensor(np.concatenate([wih, wfh, wch, woh]))
        wx_tensor = torch.tensor(np.concatenate([wix, wfx, wcx, wox]))
        bh_tensor = torch.tensor(np.concatenate([bi, bf, bc, bo]))

        state_dict = self.state_dict().copy()
        state_dict["_lstm.weight_hh_l0"] = wh_tensor
        state_dict["_lstm.weight_ih_l0"] = wx_tensor
        state_dict["_lstm.bias_hh_l0"] = bh_tensor
        state_dict["_lstm.bias_ih_l0"] = torch.zeros(4 * self._hidden_size)
        state_dict["_decoder._linear.weight"] = torch.tensor(w)
        state_dict["_decoder._linear.bias"] = torch.tensor(b)

        self.load_state_dict(state_dict)

    @property
    def _forget_gate(self) -> Weights:
        """
        Defines the weights for the forget gate. By default, it is set
        to all 1s.

        :return: W_hh, W_ih, and the bias
        """
        return np.zeros((self._hidden_size, self._hidden_size)), \
               np.zeros((self._hidden_size, len(self.x_stoi))), \
               self.m * np.ones(self._hidden_size)

    @property
    @abstractmethod
    def _cell_state_update(self) -> Weights:
        """
        Defines the weights for determining the value added to the cell
        state at each time step.

        :return: W_hh, W_ih, and the bias
        """
        raise NotImplementedError("Cell state update undefined")

    @property
    def _input_gate(self) -> Weights:
        """
        Defines the weights for the input gate. By default, it is set to
        all 1s.

        :return: W_hh, W_ih, and the bias
        """
        return np.zeros((self._hidden_size, self._hidden_size)), \
               np.zeros((self._hidden_size, len(self.x_stoi))), \
               self.m * np.ones(self._hidden_size)

    @property
    def _output_gate(self) -> Weights:
        """
        Defines the weights for the output gate. By default, it is set
        to all 1s.

        :return: W_hh, W_ih, and the bias
        """
        return np.zeros((self._hidden_size, self._hidden_size)), \
               np.zeros((self._hidden_size, len(self.x_stoi))), \
               self.m * np.ones(self._hidden_size)

    @property
    @abstractmethod
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Defines the weights of the linear decoder.

        :return: The weight matrix and the bias
        """
        raise NotImplementedError("Decoder undefined")
