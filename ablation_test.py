"""
This is the script for running the SP task ablation test.
"""
import random
from typing import Dict, List, Set, Union

import numpy as np

from attribution import *
from fsa import sp_fsa
from models.whitebox import WhiteBoxRNN, SPRNN, FSARNN


def _check_subseqs(letters: Union[Set[str], str], suffix: str) -> bool:
    if "a" in letters and suffix == "b":
        return False
    elif ("b" in letters or "d" in letters) and suffix == "c":
        return False
    elif "c" in letters and suffix == "d":
        return False
    return True


def no_subseqs(x_tokens: Union[List[str], str]) -> bool:
    """
    Checks to see whether a string lacks the subsequences ab, bc, cd,
    and dc.

    :param x_tokens: A string
    :return: True iff x_tokens does not have any subsequences
    """
    letters = set()
    for letter in x_tokens:
        if not _check_subseqs(letters, letter):
            return False
        letters.add(letter)
    return True


def attr_ablation(x: str, attr: AttributionMixin) -> int:
    """
    Removes the symbol with the highest attribution score from a string
    until all ab, bc, cd, and dc subsequences are destroyed.

    :param x: An input
    :param attr: An attribution method instance
    :return: The number of symbols removed from x
    """
    x_tokens = list(x)
    while len(x_tokens) > 1:
        score_argmax = np.argmax(attr("".join(x_tokens)))
        del x_tokens[score_argmax]
        if no_subseqs(x_tokens):
            break

    return len(x) - len(x_tokens)


def random_ablation(x: str) -> int:
    """
    Randomly removes symbols from a string until all ab, bc, cd, and dc
    subsequences are destroyed.

    :param x: An input
    :return: The number of symbols removed from x
    """
    x_tokens = list(x)
    num_ablations = 0
    while len(x_tokens) > 1:
        num_ablations += 1
        ind = random.randint(0, len(x_tokens) - 1)
        del x_tokens[ind]
        if no_subseqs(x_tokens):
            return num_ablations


fsa_dict = dict(a=dict(), b=dict(), c=dict(), d=dict())
for q, letter, r in sp_fsa.transitions:
    assert q not in fsa_dict[letter]
    fsa_dict[letter][q] = r


def optimal_ablation(x: str) -> int:
    """
    Finds the minimum number of symbols that need to be removed from a
    string in order to destroy all ab, bc, cd, and dc subsequences. This
    is achieved using a dynamic programming algorithm where the chart is
    indexed by the string and the states of an FSA that detects
    subsequences.

    :param x: An input
    :return: The number of symbols removed from x
    """
    assert len(x) > 1
    global fsa_dict
    chart = {q: 0 for q in sp_fsa.states}

    for letter in x:
        new_chart = {q: 0 for q in sp_fsa.states}
        for r in new_chart:
            max_subseq_len = 0
            for q in chart:
                if (q == 0 or chart[q] > 0) and fsa_dict[letter][q] == r:
                    subseq_len = chart[q] + 1
                    max_subseq_len = max(max_subseq_len, subseq_len)
            new_chart[r] = max(max_subseq_len, chart[r])

        chart = new_chart

    max_subseq_len = max(chart[q] for q in sp_fsa.accept_states)
    return len(x) - max_subseq_len


def ablation_test(model: WhiteBoxRNN) -> Dict[str, List[float]]:
    """
    Runs the ablation test for one of the two SP networks.

    :param model: An SP network
    :return: The results of the ablation test
    """
    occ = OcclusionAttribution(model)
    sal = SaliencyAttribution(model)
    gi = GxIAttribution(model)
    ig = IGAttribution(model)
    lrp = LRPAttribution(model)
    attrs = [occ, sal, gi, ig, lrp]

    results = {a.name: [] for a in attrs}
    with open("datasets/ablation_test_data.txt", "r") as f:
        for line in f:
            x = line.strip()
            print(x)
            print("Length:", len(x))

            for a in attrs:
                score = attr_ablation(x, a)
                percentage = score / len(x) * 100
                results[a.name].append(percentage)
                print("{}: {} ({:.1f}%)".format(a.name, score, percentage))
            print()

    return results


def print_results(results: Dict[str, List[float]]):
    for a, scores in results.items():
        mean = np.mean(scores)
        stdev = np.std(scores)
        print("{}: mean {:.1f}%, stdev {:.1f}%".format(a, mean, stdev))
    print()


if __name__ == "__main__":
    # Test attribution methods for both models
    print("Testing counter-based SP network\n")
    counter_results = ablation_test(SPRNN())
    print_results(counter_results)

    print("Testing FSA-based SP network\n")
    fsa_results = ablation_test(FSARNN(u=.5))
    print_results(counter_results)

    # Optimal and random baselines
    baseline_results = dict(Optimal=[], Random=[])
    print("Testing optimal and random baselines\n")
    with open("datasets/ablation_test_data.txt", "r") as f:
        for line in f:
            x = line.strip()
            print(x)
            print("Length:", len(x))

            optimal_score = optimal_ablation(x)
            optimal_percentage = optimal_score / len(x) * 100
            print("Optimal: {} ({:.1f}%)".format(optimal_score,
                                                 optimal_percentage))

            random_score = random_ablation(x)
            random_percentage = random_score / len(x) * 100
            print("Random: {} ({:.1f}%)\n".format(random_score,
                                                  random_percentage))

            baseline_results["Optimal"].append(optimal_percentage)
            baseline_results["Random"].append(random_percentage)

    print_results(baseline_results)

    # Summary
    print("Summary\n\nCounter:")
    print_results(counter_results)

    print("FSA:")
    print_results(fsa_results)

    print("Baselines:")
    print_results(baseline_results)
