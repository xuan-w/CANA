# -*- coding: utf-8 -*-
#
# Tests for ``dynamical_impact.py`` (instantaneous and cumulative perturbation impact)
#
import numpy as np

import cana.dynamical_impact as dyn
from cana.boolean_network import BooleanNetwork
from cana.datasets.bio import BUDDING_YEAST
from cana.utils import binstate_compare, flip_binstate_bit


def _two_node_oscillator():
    """x0' = x1, x1' = NOT x0: every state lies on one 4-cycle."""
    logic = {
        0: {'name': 'x0', 'in': [1], 'out': [0, 1]},
        1: {'name': 'x1', 'in': [0], 'out': [1, 0]},
    }
    return BooleanNetwork.from_dict(logic)


def test_cumulative_impact_by_hand():
    # Flipping x0 at t = 0 moves the state a fixed distance around the 4-cycle,
    # so the copies stay apart for ever: x1 differs at t = 1, x0 at t = 2, both
    # at t = 3, ... The cumulative impact of x0 on itself is 0 at t = 1 and
    # stays 1 from t = 2 on; on x1 it is 1 from t = 1.
    bn = _two_node_oscillator()
    inst, cum = bn.dynamical_impact_node(0, n_traj=0, t=4)
    assert np.array_equal(inst, [[0, 1], [1, 0], [0, 1], [1, 0]])
    assert np.array_equal(cum, [[0, 1], [1, 1], [1, 1], [1, 1]])


def test_cumulative_is_monotone_and_bounds_instantaneous():
    bn = BUDDING_YEAST()
    for node in range(bn.Nnodes):
        inst, cum = bn.dynamical_impact_node(node, n_traj=0, t=10)
        assert np.all(np.diff(cum, axis=0) >= 0)
        assert np.all(cum >= inst)
        assert np.array_equal(cum[0], inst[0])


def test_methods_agree_with_one_pass():
    bn = BUDDING_YEAST()
    inst, cum = bn.dynamical_impact_node(3, n_traj=0, t=6)
    assert np.array_equal(bn.partial_derative_node(3, n_traj=0, t=6), inst)
    assert np.array_equal(bn.cumulative_impact_node(3, n_traj=0, t=6), cum)


def _partial_derative_node_before_split(bn, node, configs, t):
    """The loop of partial_derative_node as it was before it moved to cana.dynamical_impact."""
    partial = np.zeros((t, bn.Nnodes), dtype=float)
    for config in configs:
        perturbed_config = flip_binstate_bit(config, node)
        for n_step in range(t):
            config = bn.step(config)
            perturbed_config = bn.step(perturbed_config)
            partial[n_step] += np.logical_not(binstate_compare(config, perturbed_config))
    partial /= len(configs)
    return partial


def test_partial_derative_node_output_unchanged_when_sampling(monkeypatch):
    # random_binstate reseeds from the OS on every call (random.seed(None)), so a
    # seed cannot reproduce a sampled run; feed both loops the same configurations
    bn = BUDDING_YEAST()
    rng = np.random.default_rng(7)
    configs = [''.join(rng.choice(['0', '1'], bn.Nnodes)) for _ in range(50)]
    draws = iter(configs)
    monkeypatch.setattr(dyn, 'random_binstate', lambda length: next(draws))
    new = bn.partial_derative_node(2, n_traj=50, t=8)
    old = _partial_derative_node_before_split(bn, 2, configs, t=8)
    assert np.array_equal(old, new)


def test_rng_makes_sampling_reproducible():
    bn = BUDDING_YEAST()
    a = bn.dynamical_impact_node(2, n_traj=200, t=6, rng=11)
    b = bn.dynamical_impact_node(2, n_traj=200, t=6, rng=np.random.default_rng(11))
    c = bn.dynamical_impact_node(2, n_traj=200, t=6, rng=12)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))
    assert np.array_equal(bn.partial_derative_node(2, n_traj=200, t=6, rng=11), a[0])
    assert np.array_equal(bn.cumulative_impact_node(2, n_traj=200, t=6, rng=11), a[1])


def test_rng_generator_is_advanced_not_copied():
    # two calls on one Generator draw different configurations, as a caller looping over sources expects
    bn = BUDDING_YEAST()
    rng = np.random.default_rng(3)
    first = bn.partial_derative_node(2, n_traj=100, t=4, rng=rng)
    second = bn.partial_derative_node(2, n_traj=100, t=4, rng=rng)
    assert not np.array_equal(first, second)


def test_rng_leaves_global_random_alone():
    import random
    random.seed(5)
    state = random.getstate()
    BUDDING_YEAST().dynamical_impact_node(1, n_traj=50, t=3, rng=0)
    assert random.getstate() == state


def test_rng_sample_matches_exact_truth():
    # uniform over all 2^12 states: 20000 draws put every entry within 0.03 of the exact value
    bn = BUDDING_YEAST()
    exact = bn.dynamical_impact_node(4, n_traj=0, t=8)
    sampled = bn.dynamical_impact_node(4, n_traj=20_000, t=8, rng=1)
    for e, s in zip(exact, sampled):
        assert np.abs(e - s).max() < 0.03


def test_exact_enumeration_ignores_rng():
    bn = BUDDING_YEAST()
    a = bn.dynamical_impact_node(4, n_traj=0, t=5)
    b = bn.dynamical_impact_node(4, n_traj=0, t=5, rng=99)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
