from typing import Tuple

import numpy as np

from data.loaders import load_counter
from models.whitebox.whitebox import WhiteBoxRNN, Weights


class CounterRNN(WhiteBoxRNN):
    """
    White-box LSTM for the counting task.
    """

    def __init__(self, m: float = 50., u: float = .5):
        """
        Constructs an LSTM for the counting task.

        :param m: The constant used to saturate the gates
        :param u: The value used to increment the counter
        """
        self.m2 = m if u is None else u
        dataset = load_counter(blank=True)
        super(CounterRNN, self).__init__(dataset, 2, m=m)

    @property
    def _cell_state_update(self) -> Weights:
        wch = np.zeros((self._hidden_size, self._hidden_size))
        bc = np.zeros(self._hidden_size)
        wcx = np.zeros((self._hidden_size, len(self._x_stoi)))
        wcx[0, self._x_stoi["a"]] = self.m2
        wcx[0, self._x_stoi["b"]] = -self.m2

        return wch, wcx, bc

    @property
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        w = np.zeros((len(self._y_stoi), self._hidden_size))
        w[self._y_stoi["True"], 0] = 1.

        b = np.zeros((len(self._y_stoi)))
        b[self._y_stoi["False"]] = np.tanh(np.tanh(self.m2)) / 2

        return w, b
