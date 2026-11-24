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
    # Rounding noise is injected: with the synchronous bias update (2026-11-23) no seed tried
    # (1-1500) still produces ~1e-16 weights on its own, so the tolerance is tested directly.
    import cana.boolean_node as cbnode
    orig = cbnode.BooleanNode.edge_effectiveness

    def noisy(self, *args, **kwargs):
        e = list(orig(self, *args, **kwargs))
        if kwargs.get("biased"):
            # on inputs with zero plain and zero biased effectiveness (edges absent from the plain
            # EG, as in the original data case), put a float-rounding-sized weight
            plain = orig(self, bound=kwargs.get("bound", "mean"))
            e = [2.2e-16 if p == 0 and w == 0 else w for p, w in zip(plain, e)]
        return e

    monkeypatch.setattr(cbnode.BooleanNode, "edge_effectiveness", noisy)
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
