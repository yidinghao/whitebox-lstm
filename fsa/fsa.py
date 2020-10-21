import itertools
from typing import List, Set, Tuple

Transition = Tuple[int, str, int]


class FSA(object):
    """
    An FSA. States are numbers, 0 is the start state.
    """

    def __init__(self, *transitions: List[Transition], states: Set[int] = None,
                 start_state: int = 0, accept_states: Set[int] = None,
                 alphabet: Set[str] = None):
        """
        FSA constructor.

        :param transitions: The set of transitions
        :param start_state: The start state
        :param accept_states: The set of accept states. If None, then
            all states are accept states, and a sink state will be added
        :param alphabet: The alphabet
        """
        self.transitions = set(transitions)

        if states is None:
            self.states = {t[0] for t in self.transitions}
            self.states.update({t[2] for t in self.transitions})
        else:
            self.states = states

        if alphabet is None:
            self.alphabet = {t[1] for t in self.transitions}
        else:
            self.alphabet = alphabet

        if accept_states is None:  # Add a sink state
            self.accept_states = self.states.copy()
            sink_state = max(self.states) + 1
            self.states.add(sink_state)
            self.reject_states = {sink_state}

            avail_trans = set(itertools.product(self.states, self.alphabet))
            avail_trans.difference_update({t[:2] for t in self.transitions})
            self.transitions.update((q, a, sink_state) for q, a in avail_trans)
        else:
            self.accept_states = accept_states
            self.reject_states = self.states.difference(accept_states)

    @classmethod
    def from_xfst_string(cls, xfst_output: str, alphabet: Set[str]):
        """
        Instantiates an FSA from xfst output, obtained from print net.

        :param xfst_output: The output of xfst
        :param alphabet: The alphabet
        :return: An instantiated FSA
        """
        transitions = dict()
        states = set()
        lines = xfst_output.split("\n")
        for line in lines:
            if ":" not in line:
                continue

            p, out_arrows = line.strip()[:-1].split(": ")
            for arrow in out_arrows.split(", "):
                a, q = arrow.split(" -> ")
                p_int = int(p[2:])
                q_int = int(q[2:])
                states.add(p_int)
                states.add(q_int)
                if a != "?":
                    transitions[(p_int, a)] = q_int

        return cls(*{(p, a, q) for (p, a), q in transitions.items()},
                   alphabet=alphabet, states=states)


sp_fsa = FSA.from_xfst_string("""
    fs0: ? -> fs0, a -> fs1, b -> fs2, c -> fs3, d -> fs2.
    fs1: ? -> fs1, a -> fs1, c -> fs4, d -> fs5.
    fs2: ? -> fs2, a -> fs5, b -> fs2, d -> fs2.
    fs3: ? -> fs3, a -> fs4, b -> fs6, c -> fs3.
    fs4: ? -> fs4, a -> fs4, c -> fs4.
    fs5: ? -> fs5, a -> fs5, d -> fs5.
    fs6: ? -> fs6, a -> fs7, b -> fs6.
    fs7: ? -> fs7, a -> fs7.
""", {"a", "b", "c", "d"})


