# -*- coding: utf-8 -*-
#
# The bias-aware EG must not depend on the order in which the network lists its nodes.
# Bug fixed 2026-11-23: update_biased_boolean_network wrote each node's new bias into its
# targets inside the loop (a sequential update), so after a few rounds the weights depended on
# node order. Two copies of the same Cell Collective model (Cell Cycle Transcription, in CANA's
# order and in BioDivine's) gave different bias-EG(1).
#
import random

import pytest

from cana.boolean_network import BooleanNetwork
from cana.datasets.bio import load_cell_collective_model
from cana.random_boolean_network import regular_boolean_network


def permuted(bn, seed):
    """The same network with its nodes listed in a random order (each node keeps its rule)."""
    old = list(range(bn.Nnodes))
    new_of = dict(zip(old, random.Random(seed).sample(old, len(old))))
    logic = {new_of[i]: {"name": bn.logic[i]["name"], "in": [new_of[j] for j in bn.logic[i]["in"]],
                         "out": list(bn.logic[i]["out"])}
             for i in old}
    return BooleanNetwork.from_dict(dict(sorted(logic.items())))


def weights(bn, rounds):
    names = [n.name for n in bn.nodes]
    g = bn.biased_effective_graph(max_iter=rounds, threshold=0.0)
    return {(names[u], names[v]): w for u, v, w in g.edges(data="weight")}


def same(a, b):
    return a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-12 for k in a)


@pytest.mark.parametrize("rounds", [1, 2, 5])
def test_cell_cycle_transcription_permuted(rounds):
    bn = load_cell_collective_model("Cell Cycle Transcription by Coupled CDK and Network Oscillators")
    ref = weights(bn, rounds)
    for seed in range(5):
        assert same(weights(permuted(bn, seed), rounds), ref)


@pytest.mark.parametrize("rounds", [1, 2])
def test_random_bn_permuted(rounds):
    random.seed(21260923)
    bn = regular_boolean_network(N=100, K=3, bias=0.5, bias_constraint="soft", keep_constants=True,
                                 remove_multiedges=True)
    ref = weights(bn, rounds)
    for seed in range(3):
        assert same(weights(permuted(bn, seed), rounds), ref)
