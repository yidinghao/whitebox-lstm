"""
This module loads Torchtext fields that are used to tokenize and prepare
input batches for the white-box models.
"""
import pickle
from typing import Callable, List, Tuple

import torch
from torchtext import data as tt
from torchtext.vocab import Vocab

Tokenizer = Callable[[str], List[str]]


def add_postprocessing(field: tt.Field):
    """
    Adds postprocessing to a Field that converts batches to one-hots.

    :param field: The field to add postprocessing to
    :return: None
    """
    p = tt.Pipeline(convert_token=lambda x, v: [_to_one_hot(i, v) for i in x])
    field.postprocessing = p
    field.dtype = torch.float


def _to_one_hot(i: int, vocab: Vocab) -> List[int]:
    return [1 if j == i else 0 for j in range(len(vocab))]


def load_fields(filename: str) -> Tuple[tt.Field, tt.Field]:
    """
    Loads the fields for a formal language task from a pickled object.

    :param filename: The filename of the pickled fields
    :return: The input and output fields
    """
    with open(filename, "rb") as f:
        fields = pickle.load(f)
        add_postprocessing(fields["x"])

    return fields["x"], fields["y"]
