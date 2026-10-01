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
