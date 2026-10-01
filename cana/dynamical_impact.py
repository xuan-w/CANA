"""Dynamical impact of a single-node perturbation, measured by simulation.

A perturbation flips one node in an initial configuration; the original and
the flipped copy are then stepped side by side. Two ground-truth measures of
how the flip spreads come out of the same pair of trajectories:

- the *instantaneous* impact, P(node j differs at step t), which is what
  :meth:`BooleanNetwork.partial_derative_node` has always returned;
- the *cumulative* impact, P(node j has differed at some step <= t). Per
  initial configuration this is the running OR of the 0/1 differences; it is
  averaged over configurations only at the end, so it is not the running
  maximum of the instantaneous average.

The cumulative measure is the one a cumulative predictor (is j within t hops
on the influence graph, the strongest effective-graph path of at most t hops)
describes, since both only grow with t.

Following :mod:`cana.control.pinning`, these are plain functions that take
what they need from the network as arguments; :class:`BooleanNetwork` keeps
thin methods that call them.
"""
import numpy as np

from cana.cutils import binstate_compare, flip_binstate_bit, random_binstate


def _configurations(num2bin, Nnodes, Nstates, n_traj):
    """Initial configurations and their count: every state if n_traj == 0, else n_traj random ones."""
    if n_traj == 0:
        return (num2bin(statenum) for statenum in range(Nstates)), Nstates
    return (random_binstate(Nnodes) for _ in range(n_traj)), n_traj


def _impact_pass(step, num2bin, Nnodes, Nstates, node, n_traj, t):
    """One simulation pass returning (instantaneous, cumulative), each a (t, Nnodes) array.

    Row s of each array is the state after s + 1 steps.
    """
    instantaneous = np.zeros((t, Nnodes), dtype=float)
    cumulative = np.zeros((t, Nnodes), dtype=float)
    configs, n_configs = _configurations(num2bin, Nnodes, Nstates, n_traj)

    for config in configs:
        perturbed_config = flip_binstate_bit(config, node)
        # nodes that have differed at any step so far, for this initial configuration only
        ever_differed = np.zeros(Nnodes, dtype=bool)
        for n_step in range(t):
            config = step(config)
            perturbed_config = step(perturbed_config)
            differs = np.logical_not(binstate_compare(config, perturbed_config))
            ever_differed |= differs
            instantaneous[n_step] += differs
            cumulative[n_step] += ever_differed

    return instantaneous / n_configs, cumulative / n_configs


def dynamical_impact_node(step, num2bin, Nnodes, Nstates, node, n_traj=10, t=1):
    """Instantaneous and cumulative impact of flipping node, from one pass.

    Args:
        step (callable) : advances a binary-state string by one synchronous step.
        num2bin (callable) : state number to binary-state string (used when n_traj == 0).
        Nnodes (int) : number of nodes.
        Nstates (int) : number of states enumerated when n_traj == 0.
        node (int) : the node index for perturbations.
        n_traj (int) : the number of sampled initial configurations;
            if 0, every state is used and the result is exact.
        t (int) : the number of time steps.

    Returns:
        (tuple of arrays) : (instantaneous, cumulative), each of shape (t, Nnodes).
    """
    return _impact_pass(step, num2bin, Nnodes, Nstates, node, n_traj, t)


def instantaneous_impact_node(step, num2bin, Nnodes, Nstates, node, n_traj=10, t=1):
    """P(j differs at step s) for s = 1..t, shape (t, Nnodes). See :func:`dynamical_impact_node`."""
    return _impact_pass(step, num2bin, Nnodes, Nstates, node, n_traj, t)[0]


def cumulative_impact_node(step, num2bin, Nnodes, Nstates, node, n_traj=10, t=1):
    """P(j has differed at some step <= s) for s = 1..t, shape (t, Nnodes). See :func:`dynamical_impact_node`."""
    return _impact_pass(step, num2bin, Nnodes, Nstates, node, n_traj, t)[1]
