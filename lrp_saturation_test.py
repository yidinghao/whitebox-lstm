"""
This is the script for running the LRP saturation test. It tries to
determine whether prefixes of counter task inputs with equal numbers of
as and bs receive LRP scores of 0. This test is run with varying levels
of model saturation.
"""
import csv

import numpy as np

from attribution.lrp import LRPAttribution
from models.whitebox.counter import CounterRNN


def _zero_point(x: str) -> int:
    ctr = 0
    for ind, letter in enumerate(x):
        if letter == "a":
            ctr += 1
        else:
            ctr -= 1
        if ctr == 0:
            return ind
    return -1


def lrp_saturation_test(model: CounterRNN):
    lrp = LRPAttribution(model)
    with open("datasets/lrp_saturation_test_data.csv", "r") as f:
        reader = csv.reader(f)

        n_correct = 0
        n_zeros = 0
        zero_values = []
        for line in reader:
            y_ind = model.y_stoi[line[1]]

            # Test accuracy
            x_batch = model.x_field.process([line[0]])
            y_hat = model(*x_batch).squeeze()
            if y_ind == y_hat.argmax():
                n_correct += 1

            # Count percentage of zeros
            if abs(lrp(line[0], target=3)[0]) < 1e-5:
                n_zeros += 1

            # Record average value of zero counter
            zero_ind = _zero_point(line[0])
            c_t = model.lstm.traces[0][1][:, 0]
            zero_values.append(c_t[zero_ind])

    print("m =", model.m)
    print("Zeros: {} out of 1000".format(n_zeros))
    print("Correct: {} out of 100".format(n_correct))
    print("Avg. Zero Counter:", np.mean(zero_values), end="\n\n")


if __name__ == "__main__":
    for m in range(8, 25, 2):
        lrp_saturation_test(CounterRNN(m=m / 2))
