from typing import Tuple

import numpy as np
from torchtext import data as tt

from datasets.loaders import load_fields
from models.whitebox.whitebox import WhiteBoxRNN, Weights
from tools.fsa import FSA, sp_fsa

sp_x_field, sp_y_field = load_fields("datasets/sp_fields.p")


class SPRNN(WhiteBoxRNN):
    """
    Counter-based white-box LSTM for the SP task.
    """

    def __init__(self, m: float = 50., u: float = .7):
        """
        Constructs the counter-based SP network.

        :param m: The constant used to saturate the gates
        :param u: The value used to increment the counter
        """
        self.u = u
        super(SPRNN, self).__init__(sp_x_field, sp_y_field, 7, m=m)

    @property
    def _cell_state_update(self) -> Weights:
        wch = np.zeros((self._hidden_size, self._hidden_size))
        bc = np.zeros(self._hidden_size)
        wcx = np.zeros((self._hidden_size, len(self.x_stoi)))
        for i, s in enumerate(["a", "b", "c", "d"]):
            wcx[i, self.x_stoi[s]] = self.u
            if s != "a":
                wcx[i + 3, self.x_stoi[s]] = self.u

        return wch, wcx, bc

    @property
    def _input_gate(self) -> Weights:
        wix = np.zeros((self._hidden_size, len(self.x_stoi)))
        wih = np.zeros((self._hidden_size, self._hidden_size))
        wih[4, 0] = 2 * self.m
        wih[5, 1] = 2 * self.m
        wih[5, 3] = 2 * self.m
        wih[6, 2] = 2 * self.m

        bi = np.empty(self._hidden_size)
        bi[:4] = self.m
        bi[4:] = -self.m

        return wih, wix, bi

    @property
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        w = np.zeros((len(self.y_stoi), self._hidden_size))
        w[self.y_stoi["False"], 4:] = 1.

        b = np.zeros((len(self.y_stoi)))
        b[self.y_stoi["True"]] = np.tanh(np.tanh(self.u)) / 20

        return w, b


class FSARNN(WhiteBoxRNN):
    """
    A white-box LSTM that implements an FSA. By default, an FSA for the
    SP task will be implemented. However, any arbitrary FSA can be used.
    """

    def __init__(self, fsa: FSA = sp_fsa, x_field: tt.Field = sp_x_field,
                 y_field: tt.Field = sp_y_field, m: float = 50.,
                 u: float = 1.):
        """
        Constructor for an FSA-based SP network.

        :param fsa: The FSA that will be implemented
        :param x_field: The field used to tokenize inputs
        :param y_field: The field used to decode outputs
        :param m: The constant used to saturate the gates
        :param u: A scaling factor for one-hot vectors
        """
        self.fsa = fsa
        self.u = u
        hidden_size = len(fsa.states) * len(x_field.vocab)
        super(FSARNN, self).__init__(x_field, y_field, hidden_size, m=m)

    @property
    def _forget_gate(self) -> Weights:
        """
        Set to all 0s

        :return:
        """
        return (np.zeros((self._hidden_size, self._hidden_size)),
                np.zeros((self._hidden_size, len(self.x_stoi))),
                -self.m * np.ones(self._hidden_size))

    @property
    def _cell_state_update(self) -> Weights:
        wch = np.zeros((self._hidden_size, self._hidden_size))
        bc = np.zeros(self._hidden_size)
        wcx = self.u * np.tile(np.identity(len(self.x_stoi)),
                               (len(self.fsa.states), 1))

        return wch, wcx, bc

    def _get_cell_position(self, state: int, symbol: str) -> int:
        return state * len(self.x_stoi) + self.x_stoi[symbol]

    @property
    def _input_gate(self) -> Weights:
        wix = np.zeros((self._hidden_size, len(self.x_stoi)))

        # First define bias: what happens when h_{t - 1} == 0
        bi = -self.m * np.ones(self._hidden_size)
        for p, a, q in self.fsa.transitions:
            if p == 0:
                bi[self._get_cell_position(q, a)] = self.m

        # Then initialize weight based on bias
        wih = np.empty((self._hidden_size, len(self.fsa.states)))
        wih[:] = -self.m - np.expand_dims(bi, axis=1)
        for p, a, q in self.fsa.transitions:
            r = self._get_cell_position(q, a)
            wih[r, p] = self.m - bi[r]

        wih = wih.repeat(len(self.x_stoi), axis=1)

        return wih, wix, bi

    @property
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        w = np.zeros((len(self.y_stoi), len(self.fsa.states)))
        w[self.y_stoi["False"], -1] = 1.
        w[self.y_stoi["True"], :-1] = 1.
        w = w.repeat(len(self.x_stoi), axis=1)

        b = np.zeros((len(self.y_stoi)))

        return w, b
