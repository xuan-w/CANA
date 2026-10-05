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


# ---------------------------------------------------------------------------
# perturbation="flip_pin" and the derived cumulative impacts (user, 2026-10-05)
# ---------------------------------------------------------------------------

def test_flip_pin_by_hand():
    # Pin x0 at its flipped value: from (not a, b) the perturbed copy goes to (not a, a) and stays,
    # while the free copy circles (b, not a) -> (not a, not b) -> (not b, a) -> (a, b). Averaged over
    # the four initial states (a, b), each difference is 1 always, 0 never, or 1/2 (only when a == b,
    # or only when a != b).
    bn = _two_node_oscillator()
    inst, cum = bn.dynamical_impact_node(0, n_traj=0, t=4, perturbation="flip_pin")
    assert np.array_equal(inst, [[.5, 1], [0, .5], [.5, 0], [1, .5]])
    assert np.array_equal(cum, [[.5, 1], [.5, 1], [1, 1], [1, 1]])


def test_flip_is_the_default_and_unchanged():
    bn = BUDDING_YEAST()
    a = bn.dynamical_impact_node(5, n_traj=0, t=6)
    b = bn.dynamical_impact_node(5, n_traj=0, t=6, perturbation="flip")
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_flip_pin_holds_the_source():
    # the perturbed copy's source never moves, so the source differs at step s exactly when the
    # free copy's source equals its initial (unflipped) value at s
    bn = BUDDING_YEAST()
    src, t = 3, 7
    inst, _ = bn.dynamical_impact_node(src, n_traj=0, t=t, perturbation="flip_pin")
    expected = np.zeros(t)
    for statenum in range(bn.Nstates):
        config = start = bn.num2bin(statenum)
        for s in range(t):
            config = bn.step(config)
            expected[s] += config[src] == start[src]
    assert np.allclose(inst[:, src], expected / bn.Nstates)


def test_pinned_step_is_step_then_overwrite():
    bn = BUDDING_YEAST()
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = ''.join(rng.choice(['0', '1'], bn.Nnodes))
        node, pin = int(rng.integers(bn.Nnodes)), str(rng.integers(2))
        stepped = bn.step(state)
        assert bn.pinned_step(state, pinned_binstate=pin, pinned_var=[node]) == stepped[:node] + pin + stepped[node + 1:]


def test_flip_pin_sample_matches_exact_truth():
    bn = BUDDING_YEAST()
    exact = bn.dynamical_impact_node(4, n_traj=0, t=8, perturbation="flip_pin")
    sampled = bn.dynamical_impact_node(4, n_traj=20_000, t=8, rng=1, perturbation="flip_pin")
    for e, s in zip(exact, sampled):
        assert np.abs(e - s).max() < 0.03


def test_flip_pin_cumulative_is_monotone_and_bounds_instantaneous():
    bn = BUDDING_YEAST()
    for node in range(bn.Nnodes):
        inst, cum = bn.dynamical_impact_node(node, n_traj=0, t=10, perturbation="flip_pin")
        assert np.all(np.diff(cum, axis=0) >= 0)
        assert np.all(cum >= inst)


def test_derive_impact_hand_example():
    inst = np.array([1, 0, 0.4, 0, 0, 0])[:, None]
    assert np.allclose(dyn.derive_impact(inst, None, "cumulative_integral")[:, 0], [1, 1, 1.4, 1.4, 1.4, 1.4])
    assert dyn.derive_impact(inst, "stored", "cumulative_max") == "stored"
    assert dyn.derive_impact(inst, None, "instantaneous") is inst
    # stored tensors are (sources, t, Nnodes): the sum runs over t
    stack = np.stack([inst, 2 * inst])
    assert np.allclose(dyn.derive_impact(stack, None, "cumulative_integral")[1, :, 0], [2, 2, 2.8, 2.8, 2.8, 2.8])


def test_cumulative_integral_is_the_mean_of_per_configuration_sums():
    # the sum over steps commutes with the average over configurations, so deriving it from the
    # stored instantaneous impact equals summing the 0/1 differences per configuration first
    bn = BUDDING_YEAST()
    node, t = 2, 6
    inst, _ = bn.dynamical_impact_node(node, n_traj=0, t=t)
    total = np.zeros((t, bn.Nnodes))
    for statenum in range(bn.Nstates):
        config = bn.num2bin(statenum)
        perturbed = flip_binstate_bit(config, node)
        running = np.zeros(bn.Nnodes)
        for s in range(t):
            config, perturbed = bn.step(config), bn.step(perturbed)
            running += np.logical_not(binstate_compare(config, perturbed))
            total[s] += running
    assert np.allclose(dyn.derive_impact(inst, None, "cumulative_integral"), total / bn.Nstates)


def test_unknown_settings_raise():
    import pytest
    bn = BUDDING_YEAST()
    with pytest.raises(ValueError):
        bn.dynamical_impact_node(0, n_traj=0, t=2, perturbation="pin")
    with pytest.raises(ValueError):
        dyn.derive_impact(np.zeros((2, 2)), None, "cumulative")
