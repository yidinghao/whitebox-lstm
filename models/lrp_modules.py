"""
This module implements PyTorch Modules that support LRP. This is done by
re-implementing the forward pass in NumPy and then implementing a custom
backward pass using LRP rules.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.special import expit
from torch import nn

from attribution.src.lrp_lstm import lrp_linear

# A Trace is a tuple that holds computations from the LSTM forward pass
Trace = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
              np.ndarray, np.ndarray, np.ndarray]


class LRPModule(ABC):
    """
    Interface for Modules with LRP.
    """

    @abstractmethod
    def lrp_forward(self, x: np.ndarray):
        raise NotImplementedError("Please implement the LRP forward pass")

    @abstractmethod
    def lrp_backward(self, y_rel: np.ndarray):
        raise NotImplementedError("Please implement the LRP backward pass")

    @property
    @abstractmethod
    def lrp_output(self):
        raise NotImplementedError("Please implement lrp_output")


class LRPLSTM(nn.LSTM, LRPModule):
    """
    LSTM module with LRP.
    """

    def __init__(self, *args, **kwargs):
        """
        This class keeps the traces of its forward passes as state.
        """
        super(LRPLSTM, self).__init__(*args, **kwargs)
        self.traces = [None] * self.num_layers
        self.traces_rev = [None] * self.num_layers
        self._layer_inputs = [None] * self.num_layers
        self._rel_h = None
        self._rel_c = None

    """ LRP Forward and Backward Passes """

    def lrp_forward(self, x: np.ndarray):
        """
        Computes the LSTM forward pass using NumPy operations and saves
        a trace of the computation to self.traces and self.traces_rev.

        :param x: An input to the LSTM (seq_len, input_size)

        :return: None
        """
        curr_input = x
        for l in range(self.num_layers):
            self._lrp_forward(curr_input, l, 0)
            if self.bidirectional:
                self._lrp_forward(np.flip(curr_input, 0), l, 1)
                h_rev = np.flip(self.traces_rev[l][0], 0)
                curr_input = np.concatenate((self.traces[l][0], h_rev), 1)
            else:
                curr_input = self.traces[l][0]

    def lrp_backward(self, rel_y: np.ndarray,
                     eps: float = 0.001) -> np.ndarray:
        """
        Computes the LRP backward pass using NumPy operations based on
        the most recent call of self.lrp_forward.

        :param rel_y: The relevance of the LSTM output (seq_len,
            hidden_size)

        :param eps: The LRP stabilizer

        :return: The relevance of the input from the most recent call of
            self.lrp_forward
        """
        if self.bidirectional:
            curr_rel = rel_y[:, :self.hidden_size]
            curr_rel_rev = np.flip(rel_y[:, self.hidden_size:], 0)
        else:
            curr_rel = rel_y
            curr_rel_rev = None

        for l in reversed(range(self.num_layers)):
            rel_x = self._lrp_backward(curr_rel, l, 0, eps=eps)
            if self.bidirectional:
                rel_x_rev = self._lrp_backward(curr_rel_rev, l, 1, eps=eps)
                rel_x += np.flip(rel_x_rev, 0)

            if self.bidirectional and l > 0:
                curr_rel = rel_x[:, :self.hidden_size]
                curr_rel_rev = np.flip(rel_x[:, self.hidden_size:], 0)
            else:
                curr_rel = rel_x

        return curr_rel

    @property
    def lrp_output(self):
        if self.bidirectional:
            return np.concatenate((self.traces[-1][0][-1],
                                   self.traces_rev[-1][0][-1]))
        else:
            return self.traces[-1][0][-1]

    """ Helper Functions """

    def _lrp_backward(self, rel_y: np.ndarray, layer: int, direction: int,
                      eps: float = 0.001) -> np.ndarray:
        """
        Performs a backward pass using numpy operations for one layer.

        :param rel_y: The relevance flowing to this layer

        :param layer: The layer to perform the backward pass for

        :param direction: The direction to perform the backward pass for

        :return: The relevance of the layer inputs
        """
        if direction == 1:
            x = np.flip(self._layer_inputs[layer], 0)
            h, c, i, f, g, g_pre, w_ig, w_hg = self.traces_rev[layer]
        else:
            x = self._layer_inputs[layer]
            h, c, i, f, g, g_pre, w_ig, w_hg = self.traces[layer]

        # Initialize
        rel_h = np.zeros((h.shape[0] + 1, h.shape[1]))
        rel_c = np.zeros((c.shape[0] + 1, c.shape[1]))
        rel_g = np.zeros(g.shape)
        rel_x = np.zeros(x.shape)

        # Backward pass
        rel_h[1:] = rel_y
        for t in reversed(range(len(x))):
            rel_c[t + 1] += rel_h[t + 1]
            rel_c[t] = lrp_linear(f[t] * c[t - 1], c[t], rel_c[t + 1], eps=eps)
            rel_g[t] = lrp_linear(i[t] * g[t], c[t], rel_c[t + 1], eps=eps)
            rel_x[t] = lrp_linear(x[t], g_pre[t], rel_g[t], w=w_ig.T, eps=eps)

            h_prev = np.zeros(self.hidden_size) if t == 0 else h[t - 1]
            rel_h[t] += lrp_linear(h_prev, g_pre[t], rel_g[t], w=w_hg.T,
                                   eps=eps)

        self._rel_h = rel_h
        self._rel_c = rel_c
        return rel_x

    def _lrp_forward(self, x: np.ndarray, layer: int, direction: int):
        """
        Performs a forward pass using numpy operations for one layer.

        :param x: An input of shape (seq_len, input_size)

        :param layer: The layer to perform the forward pass for

        :param direction: The direction to perform the forward pass for

        :return: None, but saves a trace of the forward pass, consisting
            of the weights used and the values of the gates, to
            self._traces or self._traces_rev
        """
        kwargs = {"layer": layer, "direction": direction}
        w_ii, w_if, w_ig, w_io = self._params_numpy("weight_ih", **kwargs)
        w_hi, w_hf, w_hg, w_ho = self._params_numpy("weight_hh", **kwargs)
        b_ii, b_if, b_ig, b_io = self._params_numpy("bias_ih", **kwargs)
        b_hi, b_hf, b_hg, b_ho = self._params_numpy("bias_hh", **kwargs)

        # Initialize
        h = np.zeros((len(x), self.hidden_size))
        i = np.zeros((len(x), self.hidden_size))
        f = np.zeros((len(x), self.hidden_size))
        g_pre = np.zeros((len(x), self.hidden_size))
        g = np.zeros((len(x), self.hidden_size))
        o = np.zeros((len(x), self.hidden_size))
        c = np.zeros((len(x), self.hidden_size))

        # Forward pass
        for t in range(len(x)):
            h_prev = np.zeros(self.hidden_size) if t == 0 else h[t - 1]
            c_prev = np.zeros(self.hidden_size) if t == 0 else c[t - 1]

            i[t] = expit(w_ii @ x[t] + b_ii + w_hi @ h_prev + b_hi)
            f[t] = expit(w_if @ x[t] + b_if + w_hf @ h_prev + b_hf)
            g_pre[t] = w_ig @ x[t] + b_ig + w_hg @ h_prev + b_hg
            g[t] = np.tanh(g_pre[t])
            o[t] = expit(w_io @ x[t] + b_io + w_ho @ h_prev + b_ho)
            c[t] = f[t] * c_prev + i[t] * g[t]
            h[t] = o[t] * np.tanh(c[t])

        # Save trace to state
        if direction == 1:
            self.traces_rev[layer] = h, c, i, f, g, g_pre, w_ig, w_hg
        else:
            self._layer_inputs[layer] = x
            self.traces[layer] = h, c, i, f, g, g_pre, w_ig, w_hg

    """ Helper Function """

    def _params_numpy(self, prefix: str, layer: int, direction: int = 0) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves weight matrices or bias vectors for a particular layer
        and direction.

        :param prefix: "weight_ih" for input weights, "weight_hh" for
            hidden state weights, "bias_ih" for input biases, or
            "bias_hh" for hidden state biases

        :param layer: The layer to retrieve weights for

        :param direction: The direction to retrieve weights for
        :return:
        """
        n = self.hidden_size
        p = prefix + "_l" + str(layer)
        if direction == 1:
            p += "_reverse"

        w = getattr(self, p).detach().numpy()
        return tuple(w[n * i:n * (i + 1)] for i in range(4))


class LRPLinear(nn.Linear, LRPModule):
    """
    Linear module with LRP.
    """

    def __init__(self, *args, **kwargs):
        super(LRPLinear, self).__init__(*args, **kwargs)
        self._input = None
        self._output = None
        self._w = None

    def lrp_forward(self, x: np.ndarray):
        self._w = self.weight.detach().numpy()
        self._input = x
        self._output = self._w @ x.T + self.bias.detach().numpy()

    def lrp_backward(self, rel_y: np.ndarray,
                     eps: float = 0.001) -> np.ndarray:
        w = self._w
        return lrp_linear(self._input, w @ self._input.T, rel_y, w.T, eps=eps)

    @property
    def lrp_output(self):
        return self._output
