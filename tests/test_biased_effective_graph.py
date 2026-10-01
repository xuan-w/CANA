# -*- coding: utf-8 -*-
#
# biased_effective_graph drops bias-aware edge effectivenesses below BIASED_EDGE_ZERO_TOL (1e-12):
# float rounding leaves ~1e-16 where the effectiveness is 0, and such an edge would otherwise pass
# threshold=0 and widen hop-based light cones.
#
import random

import cana.boolean_network as cbn
from cana.random_boolean_network import regular_boolean_network


def _random_bn(seed):
    random.seed(seed)
    return regular_boolean_network(N=30, K=3, bias=0.4, bias_constraint="soft", keep_constants=True,
                                   remove_multiedges=True)


def _edges(bn):
    return {(u, v): w for u, v, w in bn.biased_effective_graph(max_iter=2, threshold=0.0).edges(data="weight")}


def test_rounding_noise_edges_dropped_real_weights_kept(monkeypatch):
    # seed 3 has bias-aware edges of weight 1.1e-16 and 2.2e-16 when no tolerance is applied
    monkeypatch.setattr(cbn, "BIASED_EDGE_ZERO_TOL", 0.0)
    raw = _edges(_random_bn(3))
    noise = {e for e, w in raw.items() if 0 < w < 1e-12}
    assert noise and all(raw[e] < 1e-15 for e in noise)
    monkeypatch.setattr(cbn, "BIASED_EDGE_ZERO_TOL", 1e-12)
    fixed = _edges(_random_bn(3))
    assert not noise & set(fixed)                       # the ~1e-16 edges are gone
    assert {e: w for e, w in raw.items() if e not in noise} == fixed   # every real weight kept, unchanged


def test_no_edge_below_tolerance():
    for seed in range(1, 6):
        assert all(w == 0 or w >= 1e-12 for w in _edges(_random_bn(seed)).values())
