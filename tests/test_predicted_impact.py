# -*- coding: utf-8 -*-
#
# Tests for ``predicted_impact`` in ``dynamical_impact.py``: one reference per path rule.
#
#   at_most_t_edges        brute force over simple paths of <= t edges
#   inside_light_cone      BooleanNetwork.approx_dynamic_impact on this branch (rows 1 and 2)
#   global_strongest_path  brute force over all simple paths, gated by structural hops
#   pnas2021               approx_dynamic_impact of the 2021 release ec0be10, extracted from git
#   interaction graph      approx_dynamic_impact row 0, as the binary light cone (distance <= N)
#
# plus the three hand-built examples of the 2021 cutoff bug (merged-eg-project,
# reports/method-notes/eg-light-cone-2021-cutoff-bug-2026-10-01/README.org).
#
import os
import random
import subprocess
import textwrap

import networkx as nx
import numpy as np
import pytest

from cana.datasets.bio import BUDDING_YEAST, DROSOPHILA, THALIANA
from cana.dynamical_impact import PATH_RULES, predicted_impact
from cana.random_boolean_network import regular_boolean_network

BIO = {"thaliana": THALIANA, "drosophila": DROSOPHILA, "yeast": BUDDING_YEAST}
T = 15


def _random_bn(seed, N, K):
    random.seed(seed)
    return regular_boolean_network(N=N, K=K, bias=0.4, bias_constraint="soft", keep_constants=True,
                                   remove_multiedges=True)


def _networks(small=False):
    """The three bio models and three seeded random BNs (N = 12, k = 2 if small, else N = 30, k = 3)."""
    nets = [(name, f) for name, f in BIO.items()]
    N, K = (12, 2) if small else (30, 3)
    nets += [(f"random-{seed}", lambda seed=seed: _random_bn(seed, N, K)) for seed in (1, 2, 3)]
    return nets


def _sources(bn, name):
    # every source of the bio models; five per random BN keeps the test short
    return range(bn.Nnodes) if not name.startswith("random") else range(5)


# --- hand-built examples -------------------------------------------------------------------------

class _Stub:
    """A weighted EG and its unweighted structure: all predicted_impact needs from a network."""

    def __init__(self, edges, names):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(len(names)))
        for u, v, w in edges:
            self.G.add_edge(names.index(u), names.index(v), weight=w)
        self.Nnodes = len(names)

    def structural_graph(self):
        S = nx.DiGraph()
        S.add_nodes_from(self.G)
        S.add_edges_from(self.G.edges(), weight=1.0)
        return S

    def effective_graph(self, bound="mean", threshold=0.0):
        return self.G


_CHAIN = ["s", "c1", "c2", "c3", "c4", "c5", "y"]
# (edges, node names, impact of s on y per rule for t = 1..T), from the method note's README
EXAMPLES = {
    "1-weak-strongest-path": (
        [("s", "y", 0.05), ("s", "x", 0.6), ("x", "y", 0.6)], ["s", "x", "y"],
        {"pnas2021": [0.000, 0.360, 0.360], "inside_light_cone": [0.360, 0.360, 0.360],
         "at_most_t_edges": [0.050, 0.360, 0.360], "global_strongest_path": [0.360, 0.360, 0.360]}),
    "2-strong-2-hop-path-at-t1": (
        [("s", "y", 0.05), ("s", "x", 0.9), ("x", "y", 0.9)], ["s", "x", "y"],
        {"pnas2021": [0.810, 0.810, 0.810], "inside_light_cone": [0.810, 0.810, 0.810],
         "at_most_t_edges": [0.050, 0.810, 0.810], "global_strongest_path": [0.810, 0.810, 0.810]}),
    "3-six-hop-chain": (
        [("s", "y", 0.05)] + [(a, b, 0.7) for a, b in zip(_CHAIN, _CHAIN[1:])], _CHAIN,
        {"pnas2021": [0.000, 0.000, 0.118, 0.118, 0.118, 0.118, 0.118],
         "inside_light_cone": [0.050, 0.050, 0.050, 0.050, 0.118, 0.118, 0.118],
         "at_most_t_edges": [0.050, 0.050, 0.050, 0.050, 0.050, 0.118, 0.118],
         "global_strongest_path": [0.118, 0.118, 0.118, 0.118, 0.118, 0.118, 0.118]}),
}


@pytest.mark.parametrize("path", PATH_RULES)
@pytest.mark.parametrize("example", list(EXAMPLES))
def test_hand_built_examples(example, path):
    edges, names, expected = EXAMPLES[example]
    stub = _Stub(edges, names)
    got = predicted_impact(stub, names.index("s"), len(expected[path]), path=path)[:, names.index("y")]
    # the README prints three decimals
    np.testing.assert_allclose(got, expected[path], atol=5e-4)


# --- references on real networks -----------------------------------------------------------------

@pytest.mark.parametrize("name, factory", _networks())
def test_inside_light_cone_equals_approx_dynamic_impact(name, factory):
    bn = factory()
    for src in _sources(bn, name):
        legacy = bn.approx_dynamic_impact(src, T, biased=True, b_iter=2)
        np.testing.assert_allclose(predicted_impact(bn, src, T, path="inside_light_cone"), legacy[1], atol=1e-12)
        np.testing.assert_allclose(predicted_impact(bn, src, T, path="inside_light_cone", bias_iter=2), legacy[2],
                                   atol=1e-12)


@pytest.mark.parametrize("name, factory", _networks())
def test_interaction_graph_equals_binary_light_cone(name, factory):
    bn = factory()
    for src in _sources(bn, name):
        dist = bn.approx_dynamic_impact(src, T)[0]          # N + 1 outside the cone
        expected = (dist <= bn.Nnodes).astype(float)
        for graph in ("structural", "interaction"):
            np.testing.assert_array_equal(predicted_impact(bn, src, T, graph=graph), expected)


def _approx_dynamic_impact_2021():
    """approx_dynamic_impact exactly as in the 2021 release (ec0be10), extracted from git."""
    cana_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    try:
        src = subprocess.run(["git", "-C", cana_dir, "show", "ec0be10:cana/boolean_network.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history with ec0be10 not available")
    start = src.index("    def approx_dynamic_impact(")
    end = src.index("return impact_matrix[:, 1:]", start) + len("return impact_matrix[:, 1:]")
    ns = {"np": np, "nx": nx}
    exec(textwrap.dedent(src[start:end]), ns)
    return ns["approx_dynamic_impact"]


@pytest.mark.parametrize("name, factory", _networks())
def test_pnas2021_equals_2021_release(name, factory):
    v2021 = _approx_dynamic_impact_2021()
    bn = factory()
    for src in _sources(bn, name):
        np.testing.assert_allclose(predicted_impact(bn, src, T, path="pnas2021"), v2021(bn, src, T)[1], atol=1e-12)


def _brute_force(G, Nnodes, source, max_edges):
    """Best product over simple paths of <= max_edges edges (None: any length); source 0."""
    best = np.zeros(Nnodes)
    for target in range(Nnodes):
        if target == source or target not in G:
            continue
        for p in nx.all_simple_paths(G, source, target, cutoff=max_edges):
            best[target] = max(best[target], np.prod([G[u][v]["weight"] for u, v in zip(p, p[1:])]))
    return best


# weights are <= 1, so a walk is never stronger than the simple path inside it:
# the best walk of <= t edges equals the best simple path of <= t edges
@pytest.mark.parametrize("bias_iter", [None, 2])
@pytest.mark.parametrize("name, factory", _networks(small=True))
def test_at_most_t_edges_equals_brute_force(name, factory, bias_iter):
    bn = factory()
    t_max = 5
    for src in _sources(bn, name):
        got = predicted_impact(bn, src, t_max, path="at_most_t_edges", bias_iter=bias_iter)
        G = bn.effective_graph(threshold=0.0) if bias_iter is None else bn.biased_effective_graph(
            max_iter=bias_iter, threshold=0.0)
        for t in range(1, t_max + 1):
            np.testing.assert_allclose(got[t - 1], _brute_force(G, bn.Nnodes, src, t), atol=1e-12)


def test_bias_aware_at_most_t_edges_differs_from_plain():
    # the bias-aware EG must actually change the prediction somewhere, or the test above proves little
    bn = BUDDING_YEAST()
    diffs = [np.abs(predicted_impact(bn, s, 6, bias_iter=2) - predicted_impact(bn, s, 6)).max()
             for s in range(bn.Nnodes)]
    assert max(diffs) > 0.01


@pytest.mark.parametrize("name, factory", _networks(small=True))
def test_global_strongest_path_equals_brute_force(name, factory):
    bn = factory()
    G = bn.effective_graph(threshold=0.0)
    for src in _sources(bn, name):
        strongest = _brute_force(G, bn.Nnodes, src, None)
        hops = nx.single_source_shortest_path_length(bn.structural_graph(), src)
        got = predicted_impact(bn, src, T, path="global_strongest_path")
        for t in range(1, T + 1):
            reached = np.array([j in hops and hops[j] <= t for j in range(bn.Nnodes)])
            np.testing.assert_allclose(got[t - 1], np.where(reached, strongest, 0.0), atol=1e-12)


def test_bad_options_raise():
    bn = BUDDING_YEAST()
    with pytest.raises(ValueError):
        predicted_impact(bn, 0, 3, path="hop_limited")
    with pytest.raises(ValueError):
        predicted_impact(bn, 0, 3, graph="activity")
    with pytest.raises(ValueError):
        predicted_impact(bn, 0, 3, bias_iter=0)


def test_method_matches_function():
    bn = THALIANA()
    np.testing.assert_array_equal(bn.predicted_impact_node(4, 6, path="pnas2021"),
                                  predicted_impact(bn, 4, 6, path="pnas2021"))
