from typing import Tuple

import numpy as np

from datasets.loaders import load_fields
from models.whitebox.whitebox import WhiteBoxRNN

Weights = Tuple[np.ndarray, np.ndarray, np.ndarray]


class BracketRNN(WhiteBoxRNN):
    """
    PDA-based LSTM for the bracket prediction task.
    """

    def __init__(self, stack_size: int, m: float = 50.):
        """
        Constructs the PDA-based LSTM for the bracket prediction task.

        :param stack_size: The maximum size of the bounded stack
        :param m: The value used to saturate the gates
        """
        self._stack_size = stack_size
        fields = load_fields("datasets/bracket_fields.p")
        super(BracketRNN, self).__init__(*fields, 2 * stack_size + 2, m=m)

    @property
    def _forget_gate(self) -> Weights:
        n = self._stack_size

        wfh = np.zeros((self._hidden_size, self._hidden_size))
        wfh[:n, n:2 * n] = self._a_up
        wfh[n:2 * n, n:2 * n] = self._a
        wfh *= (-2. * self._m)

        wfx = np.zeros((self._hidden_size, len(self._x_stoi)))
        wfx[:2 * n, self._x_stoi["("]] = np.ones(2 * n)
        wfx[:2 * n, self._x_stoi["["]] = np.ones(2 * n)
        wfx *= (2. * self._m)

        bf = np.ones(self._hidden_size)
        bf[:2 * n] *= self._m
        bf[2 * n:] *= -self._m

        return wfh, wfx, bf

    @property
    def _cell_state_update(self) -> Weights:
        n = self._stack_size

        wch = np.zeros((self._hidden_size, self._hidden_size))
        wch[:n, -2] = np.ones(n)
        wch[-2, :n] = np.array([2. ** (i + 1.) for i in range(n)])
        wch[-1, n:2 * n] = -np.tanh(1.) * np.ones(n)
        wch *= self._m / np.tanh(1.)

        wcx = np.zeros((self._hidden_size, len(self._x_stoi)))
        wcx[n:2 * n, self._x_stoi["("]] = np.ones(n)
        wcx[n:2 * n, self._x_stoi["["]] = np.ones(n)
        wcx[-2, self._x_stoi["("]] = 2. ** (n + 1)
        wcx[-2, self._x_stoi["["]] = -(2. ** (n + 1))
        wcx[-1, self._x_stoi["("]] = -2.
        wcx[-1, self._x_stoi["["]] = -2.
        wcx *= self._m

        bc = np.zeros(self._hidden_size)
        bc[-1] = 1.5 * self._m

        return wch, wcx, bc

    @property
    def _input_gate(self) -> Weights:
        n = self._stack_size

        wih = np.zeros((self._hidden_size, self._hidden_size))
        wih[:n, n:2 * n] = self._a
        wih[n:2 * n, n:2 * n] = self._a_down
        wih *= (2. * self._m)

        wix = np.zeros((self._hidden_size, len(self._x_stoi)))
        wix[:n, self._x_stoi[")"]] = 2. * np.ones(n)
        wix[:n, self._x_stoi["]"]] = 2. * np.ones(n)
        wix[n, self._x_stoi["("]] = -2.
        wix[n, self._x_stoi["["]] = -2.
        wix *= -self._m

        bi = np.ones(self._hidden_size)
        bi[-2:] = np.array([-1., -1.])
        bi *= -self._m

        return wih, wix, bi

    @property
    def _output_gate(self) -> Weights:
        woh = np.zeros((self._hidden_size, self._hidden_size))
        wox = np.zeros((self._hidden_size, len(self._x_stoi)))
        bo = self._m * np.ones(self._hidden_size)
        return woh, wox, bo

    @property
    def _decoder_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        w = np.zeros((len(self._y_stoi), self._hidden_size))
        w[:, -1] = np.ones(len(self._y_stoi))
        w[self._y_stoi[")"], -2:] = np.array([1., 0.])
        w[self._y_stoi["]"], -2:] = np.array([-1., 0.])

        b = np.zeros((len(self._y_stoi)))

        return w, b

    @property
    def _a_up(self) -> np.ndarray:
        return np.concatenate([self._a[1:], np.zeros((1, self._stack_size))])

    @property
    def _a_down(self) -> np.ndarray:
        return np.concatenate([np.zeros((1, self._stack_size)), self._a[:-1]])

    @property
    def _a(self) -> np.ndarray:
        a = np.identity(self._stack_size)
        a[:-1, 1:] -= np.identity(self._stack_size - 1)
        return a
