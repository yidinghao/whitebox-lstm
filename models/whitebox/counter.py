from typing import Tuple

import numpy as np

from datasets.loaders import load_fields
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
        self.u = u
        fields = load_fields("datasets/counter_fields.p")
        super(CounterRNN, self).__init__(*fields, 2, m=m)

    @property
    def _cell_state_update(self) -> Weights:
        wch = np.zeros((self._hidden_size, self._hidden_size))
        bc = np.zeros(self._hidden_size)
        wcx = np.zeros((self._hidden_size, len(self.x_stoi)))
        wcx[0, self.x_stoi["a"]] = self.u
        wcx[0, self.x_stoi["b"]] = -self.u

        return wch, wcx, bc

    @property
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        w = np.zeros((len(self.y_stoi), self._hidden_size))
        w[self.y_stoi["True"], 0] = 1.

        b = np.zeros((len(self.y_stoi)))
        b[self.y_stoi["False"]] = np.tanh(np.tanh(self.u)) / 2

        return w, b
