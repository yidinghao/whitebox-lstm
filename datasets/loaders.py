"""
This module loads Torchtext fields that are used to tokenize and prepare
input batches for the white-box models.
"""
import pickle
from typing import Callable, List

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


def load_dataset(task_name: str) -> tt.Dataset:
    """
    Loads the fields for a formal language task and puts them into a
    blank Dataset.

    :param task_name: counter, sp, or bracket
    :return: A blank dataset with the fields loaded
    """
    with open("datasets/{}_fields.p".format(task_name), "rb") as f:
        fields = pickle.load(f)
        add_postprocessing(fields["x"])
    return tt.Dataset([], fields)


def load_sp() -> tt.Dataset:
    return load_dataset("sp")


def load_counter() -> tt.Dataset:
    return load_dataset("counter")


def load_bracket() -> tt.Dataset:
    return load_dataset("bracket")
