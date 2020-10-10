from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torchtext.data as tt

from models.lstm import LSTMClassifier

Weights = Tuple[np.ndarray, np.ndarray, np.ndarray]


class WhiteBoxRNN(LSTMClassifier, ABC):
    """
    Abstract class for a white-box network.
    """

    def __init__(self, dataset: tt.Dataset, hidden_size: int, m: float = 50.,
                 set_params: bool = True):
        x_size = len(dataset.fields["x"].vocab)
        y_size = len(dataset.fields["y"].vocab)
        super(WhiteBoxRNN, self).__init__(x_size, y_size, hidden_size)

        self._m = m
        self._hidden_size = hidden_size
        self._x_stoi = dataset.fields["x"].vocab.stoi
        self._y_stoi = dataset.fields["y"].vocab.stoi

        self.x_field = dataset.fields["x"]
        self.y_field = dataset.fields["y"]

        if set_params:
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
        By default, set to all 1s

        :return:
        """
        return (np.zeros((self._hidden_size, self._hidden_size)),
                np.zeros((self._hidden_size, len(self._x_stoi))),
                self._m * np.ones(self._hidden_size))

    @property
    @abstractmethod
    def _cell_state_update(self) -> Weights:
        raise NotImplementedError("Cell state update undefined")

    @property
    def _input_gate(self) -> Weights:
        """
        By default, set to all 1s

        :return: The input gate
        """
        return (np.zeros((self._hidden_size, self._hidden_size)),
                np.zeros((self._hidden_size, len(self._x_stoi))),
                self._m * np.ones(self._hidden_size))

    @property
    def _output_gate(self) -> Weights:
        """
        By default, set to all 1s

        :return: The output gate
        """
        return (np.zeros((self._hidden_size, self._hidden_size)),
                np.zeros((self._hidden_size, len(self._x_stoi))),
                self._m * np.ones(self._hidden_size))

    @property
    @abstractmethod
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Decoder undefined")
