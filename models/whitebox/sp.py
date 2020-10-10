from typing import Tuple

import numpy as np

from data.loaders import load_sp
from models.whitebox.whitebox import WhiteBoxRNN, Weights
from tools.fsa import FSA, sp_fsa


class SPRNN(WhiteBoxRNN):
    """
    Counter-based white-box LSTM for the SP task. The grammar is ab, bc,
    cd, dc.
    """

    def __init__(self, m: float = 50., u: float = .6):
        """
        Constructs the counter-based SP network.

        :param m: The constant used to saturate the gates
        :param u: The value used to increment the counter
        """
        self.u = m if u is None else u
        dataset = load_sp(blank=True)
        super(SPRNN, self).__init__(dataset, 7, m=m)

    @property
    def _cell_state_update(self) -> Weights:
        wch = np.zeros((self._hidden_size, self._hidden_size))
        bc = np.zeros(self._hidden_size)
        wcx = np.zeros((self._hidden_size, len(self._x_stoi)))
        for i, s in enumerate(["a", "b", "c", "d"]):
            wcx[i, self._x_stoi[s]] = self.u
            if s != "a":
                wcx[i + 3, self._x_stoi[s]] = self.u

        return wch, wcx, bc

    @property
    def _input_gate(self) -> Weights:
        wix = np.zeros((self._hidden_size, len(self._x_stoi)))
        wih = np.zeros((self._hidden_size, self._hidden_size))
        wih[4, 0] = 2 * self._m
        wih[5, 1] = 2 * self._m
        wih[5, 3] = 2 * self._m
        wih[6, 2] = 2 * self._m

        bi = np.empty(self._hidden_size)
        bi[:4] = self._m
        bi[4:] = -self._m

        return wih, wix, bi

    @property
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        w = np.zeros((len(self._y_stoi), self._hidden_size))
        w[self._y_stoi["False"], 4:] = 1.

        b = np.zeros((len(self._y_stoi)))
        b[self._y_stoi["True"]] = np.tanh(np.tanh(self.u)) / 20

        return w, b


class FSARNN(WhiteBoxRNN):
    """
    Handmade LSTM based on an FSA.
    """

    def __init__(self, fsa: FSA = sp_fsa, m: float = 50., u: float = None):
        """
        Constructs the FSA-based SP network.

        :param fsa: A finite-state automaton for the SP task
        :param m: The constant used to saturate the gates
        :param u: A scaling factor for one-hot vectors
        """
        self.fsa = fsa
        self.u = m if u is None else u
        dataset = load_sp(blank=True)
        hidden_size = len(fsa.states) * len(dataset.fields["x"].vocab)
        super(FSARNN, self).__init__(dataset, hidden_size, m=m)

    @property
    def _forget_gate(self) -> Weights:
        """
        Set to all 0s

        :return:
        """
        return (np.zeros((self._hidden_size, self._hidden_size)),
                np.zeros((self._hidden_size, len(self._x_stoi))),
                -self._m * np.ones(self._hidden_size))

    @property
    def _cell_state_update(self) -> Weights:
        wch = np.zeros((self._hidden_size, self._hidden_size))
        bc = np.zeros(self._hidden_size)
        wcx = self.u * np.tile(np.identity(len(self._x_stoi)),
                               (len(self.fsa.states), 1))

        return wch, wcx, bc

    def _get_cell_position(self, state: int, symbol: str) -> int:
        return state * len(self._x_stoi) + self._x_stoi[symbol]

    @property
    def _input_gate(self) -> Weights:
        wix = np.zeros((self._hidden_size, len(self._x_stoi)))

        # First define bias: what happens when h_{t - 1} == 0
        bi = -self._m * np.ones(self._hidden_size)
        for p, a, q in self.fsa.transitions:
            if p == 0:
                bi[self._get_cell_position(q, a)] = self._m

        # Then initialize weight based on bias
        wih = np.empty((self._hidden_size, len(self.fsa.states)))
        wih[:] = -self._m - np.expand_dims(bi, axis=1)
        for p, a, q in self.fsa.transitions:
            r = self._get_cell_position(q, a)
            wih[r, p] = self._m - bi[r]

        wih = wih.repeat(len(self._x_stoi), axis=1)

        return wih, wix, bi

    @property
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        w = np.zeros((len(self._y_stoi), len(self.fsa.states)))
        w[self._y_stoi["False"], -1] = 1.
        w[self._y_stoi["True"], :-1] = 1.
        w = w.repeat(len(self._x_stoi), axis=1)

        b = np.zeros((len(self._y_stoi)))

        return w, b
