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

Two independent settings (user, 2026-10-05; merged-eg-project PLAN.org "Pin
perturbation and integral cumulative impact"):

- ``perturbation``: ``"flip"`` flips the node once at t = 0 and lets it evolve;
  ``"flip_pin"`` flips it and then holds it at the flipped value for every
  later step (one node pinned to a constant, via
  :meth:`BooleanNetwork.pinned_step`; no attractor is involved). The
  unperturbed copy always runs free.
- ``impact``: ``"instantaneous"``; ``"cumulative_max"``, the cumulative impact
  above (per configuration the running max of the 0/1 difference, averaged
  afterwards); ``"cumulative_integral"``, the running sum of the
  instantaneous impact, i.e. the expected number of steps <= t at which j
  differed. A sum commutes with the average over configurations, so the
  integral is derived exactly from the stored instantaneous impact by
  :func:`derive_impact`; the max does not commute, so the simulation pass
  stores it alongside. The integral grows without bound on limit cycles;
  that is accepted because rho ranks the targets within one step.

Following :mod:`cana.control.pinning`, these are plain functions that take
what they need from the network as arguments; :class:`BooleanNetwork` keeps
thin methods that call them.
"""
import networkx as nx
import numpy as np

from cana.cutils import binstate_compare, flip_binstate_bit, random_binstate


def _configurations(num2bin, Nnodes, Nstates, n_traj, rng=None):
    """Initial configurations and their count: every state if n_traj == 0, else n_traj random ones.

    The random ones are uniform over the 2^Nnodes states. With rng=None they
    come from cana.cutils.random_binstate, which reseeds Python's global
    `random` from the OS on every call, so the draw cannot be reproduced (and
    the global state is clobbered). Pass rng -- a numpy Generator, or an int
    seed for one -- to draw them from the caller's generator instead.
    """
    if n_traj == 0:
        return (num2bin(statenum) for statenum in range(Nstates)), Nstates
    if rng is None:
        return (random_binstate(Nnodes) for _ in range(n_traj)), n_traj
    bits = np.random.default_rng(rng).integers(0, 2, size=(n_traj, Nnodes))   # a Generator is used as is
    return ("".join("1" if b else "0" for b in row) for row in bits), n_traj


PERTURBATIONS = ("flip", "flip_pin")
IMPACTS = ("instantaneous", "cumulative_max", "cumulative_integral")


def _impact_pass(step, num2bin, Nnodes, Nstates, node, n_traj, t, rng=None, perturbation="flip",
                 pinned_step=None):
    """One simulation pass returning (instantaneous, cumulative_max), each a (t, Nnodes) array.

    Row s of each array is the state after s + 1 steps. With perturbation
    "flip_pin", the perturbed copy is advanced by pinned_step with `node`
    held at its flipped value; the unperturbed copy is advanced by step.
    """
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"perturbation must be one of {PERTURBATIONS}, not {perturbation!r}")
    if perturbation == "flip_pin" and pinned_step is None:
        raise ValueError("perturbation 'flip_pin' needs pinned_step (BooleanNetwork.pinned_step)")
    instantaneous = np.zeros((t, Nnodes), dtype=float)
    cumulative = np.zeros((t, Nnodes), dtype=float)
    configs, n_configs = _configurations(num2bin, Nnodes, Nstates, n_traj, rng)

    for config in configs:
        perturbed_config = flip_binstate_bit(config, node)
        if perturbation == "flip_pin":
            pin = perturbed_config[node]
            def perturbed_step(s):
                return pinned_step(s, pinned_binstate=pin, pinned_var=[node])
        else:
            perturbed_step = step
        # nodes that have differed at any step so far, for this initial configuration only
        ever_differed = np.zeros(Nnodes, dtype=bool)
        for n_step in range(t):
            config = step(config)
            perturbed_config = perturbed_step(perturbed_config)
            differs = np.logical_not(binstate_compare(config, perturbed_config))
            ever_differed |= differs
            instantaneous[n_step] += differs
            cumulative[n_step] += ever_differed

    return instantaneous / n_configs, cumulative / n_configs


def derive_impact(instantaneous, cumulative_max, impact):
    """The (t, Nnodes) truth for one impact setting from the two stored arrays of a pass.

    Works on any array whose second-to-last axis is time, e.g. a stored
    (sources, t, Nnodes) tensor. "cumulative_integral" is the running sum of
    the instantaneous impact over steps 1..t (e.g. 1, 0, 0.4, 0 -> 1, 1, 1.4, 1.4).
    """
    if impact == "instantaneous":
        return instantaneous
    if impact == "cumulative_max":
        return cumulative_max
    if impact == "cumulative_integral":
        return np.cumsum(instantaneous, axis=-2)
    raise ValueError(f"impact must be one of {IMPACTS}, not {impact!r}")


def dynamical_impact_node(step, num2bin, Nnodes, Nstates, node, n_traj=10, t=1, rng=None, perturbation="flip",
                          pinned_step=None):
    """Instantaneous and cumulative (max) impact of perturbing node, from one pass.

    Args:
        step (callable) : advances a binary-state string by one synchronous step.
        num2bin (callable) : state number to binary-state string (used when n_traj == 0).
        Nnodes (int) : number of nodes.
        Nstates (int) : number of states enumerated when n_traj == 0.
        node (int) : the node index for perturbations.
        n_traj (int) : the number of sampled initial configurations;
            if 0, every state is used and the result is exact.
        t (int) : the number of time steps.
        rng (numpy Generator, int or None) : source of the sampled initial
            configurations (n_traj > 0); an int seeds a new Generator. None keeps
            CANA's random_binstate, which cannot be seeded (see _configurations).
        perturbation (str) : "flip" (flip once) or "flip_pin" (flip, then hold
            the node at the flipped value); see the module docstring.
        pinned_step (callable) : BooleanNetwork.pinned_step; needed for "flip_pin".

    Returns:
        (tuple of arrays) : (instantaneous, cumulative_max), each of shape (t, Nnodes).
            The cumulative integral is derive_impact(..., "cumulative_integral").
    """
    return _impact_pass(step, num2bin, Nnodes, Nstates, node, n_traj, t, rng, perturbation, pinned_step)


def instantaneous_impact_node(step, num2bin, Nnodes, Nstates, node, n_traj=10, t=1, rng=None):
    """P(j differs at step s) for s = 1..t, shape (t, Nnodes). See :func:`dynamical_impact_node`."""
    return _impact_pass(step, num2bin, Nnodes, Nstates, node, n_traj, t, rng)[0]


def cumulative_impact_node(step, num2bin, Nnodes, Nstates, node, n_traj=10, t=1, rng=None):
    """P(j has differed at some step <= s) for s = 1..t, shape (t, Nnodes). See :func:`dynamical_impact_node`."""
    return _impact_pass(step, num2bin, Nnodes, Nstates, node, n_traj, t, rng)[1]


# ---------------------------------------------------------------------------
# Predicted impact: what a graph says the flip of `source` does by step t.
# ---------------------------------------------------------------------------
#
# The effective graph (EG) predicts the impact of `source` on a target as the
# strength (product of edge effectivenesses) of the best path to it. Its
# implementations have differed in how t limits the path; `path` names the
# rule. See merged-eg-project
# reports/method-notes/eg-light-cone-2021-cutoff-bug-2026-10-01/ for the
# worked examples behind these four rules.
#
#   "at_most_t_edges"        best path of at most t edges (the paper's text). Default.
#   "inside_light_cone"      best path inside the subgraph of nodes within t
#                            hops of the source on the same EG; the path itself
#                            may be longer than t (the 2022 bias_input fork, i.e.
#                            BooleanNetwork.approx_dynamic_impact on this branch;
#                            with bias_iter the fork took the cone from the plain
#                            EG instead, see _inside_light_cone).
#   "global_strongest_path"  the strongest path over the whole EG, counted once
#                            the target is within t hops in the structural graph.
#   "pnas2021"               the 2021 release (ec0be10) verbatim, including its
#                            bug: Dijkstra's cutoff=t limits the path's -log
#                            weight, not its hops. Reproduction only.
#
# All rules give the source itself 0 and unreachable targets 0, as
# approx_dynamic_impact does.

PATH_RULES = ("at_most_t_edges", "inside_light_cone", "global_strongest_path", "pnas2021")
INTERACTION_GRAPH_NAMES = ("structural", "interaction")


def _neg_log_weight(u, v, e):
    """Edge length for Dijkstra: the shortest -log path is the strongest product path."""
    return -np.log(e['weight'])


def interaction_graph_impact(bn, source, t):
    """The interaction graph's prediction: 1 if a target is within s structural hops, for s = 1..t.

    The binary light cone of Gates et al. (2021), M_IG. The source itself is 0.

    Returns:
        (array) : shape (t, Nnodes)
    """
    hops = nx.single_source_shortest_path_length(bn.structural_graph(), source, cutoff=t)
    out = np.zeros((t, bn.Nnodes))
    for target, d in hops.items():
        if target != source:
            out[d - 1:, target] = 1.0
    return out


def _effective_graph(bn, bias_iter, bound, threshold):
    """The plain EG (bias_iter None) or the bias-aware EG after bias_iter rounds of bias propagation."""
    if bias_iter is None:
        return bn.effective_graph(bound=bound, threshold=threshold)
    if bias_iter < 1:
        # max_iter = 0 would reuse whatever input biases the nodes hold from an earlier call
        raise ValueError("bias_iter must be None (plain EG) or an integer >= 1")
    return bn.biased_effective_graph(max_iter=bias_iter, bound=bound, threshold=threshold)


def _at_most_t_edges(G, Nnodes, source, t):
    """t synchronous rounds of max-product relaxation: row s-1 = best product over paths of <= s edges."""
    best = np.zeros(Nnodes)
    best[source] = 1.0
    out = np.zeros((t, Nnodes))
    edges = list(G.edges(data='weight'))
    for step in range(t):
        new = best.copy()
        for u, v, w in edges:
            if best[u] * w > new[v]:
                new[v] = best[u] * w
        best = new
        out[step] = best
        out[step, source] = 0.0
    return out


def _inside_light_cone(G, Nnodes, source, t):
    """Best path within the subgraph of nodes at most s hops from the source in G, for s = 1..t.

    The cone and the path weights come from the same graph G (the plain or the
    bias-aware EG; decided 2026-10-01). The bias_input fork's
    approx_dynamic_impact took the cone from the plain EG even for the
    bias-aware weights; since the bias-aware EG keeps every plain-EG edge and
    can only add edges, its cone is never smaller.
    """
    hops = nx.single_source_shortest_path_length(G, source)
    out = np.zeros((t, Nnodes))
    for step in range(1, t + 1):
        cone = [n for n, d in hops.items() if d <= step]
        dist = nx.single_source_dijkstra_path_length(G.subgraph(cone), source, weight=_neg_log_weight)
        for target, length in dist.items():
            if target != source:
                out[step - 1, target] = np.exp(-length)
    return out


def _global_strongest_path(G, G_str, Nnodes, source, t):
    """Strongest path over the whole EG, from the step the target is within that many structural hops."""
    hops = nx.single_source_shortest_path_length(G_str, source, cutoff=t)
    dist = nx.single_source_dijkstra_path_length(G, source, weight=_neg_log_weight)
    out = np.zeros((t, Nnodes))
    for target, d in hops.items():
        if target != source and target in dist:
            out[d - 1:, target] = np.exp(-dist[target])
    return out


def _pnas2021(G, G_str, Nnodes, source, t):
    """The EG row of approx_dynamic_impact in the 2021 release (ec0be10), kept verbatim in logic.

    Both EG Dijkstra calls pass cutoff=t on the -log weight, so t bounds the
    path's weakness (strength >= e^-t), not its number of edges.
    """
    n_steps = t
    impact = np.zeros((n_steps + 1, Nnodes))
    str_dist, _ = nx.single_source_dijkstra(G_str, source, target=None, cutoff=n_steps)
    str_dist = {n: int(length) for n, length in str_dist.items()}
    eff_dist, eff_paths = nx.single_source_dijkstra(G, source, target=None, cutoff=n_steps, weight=_neg_log_weight)
    for target in range(Nnodes):
        if target == source or str_dist.get(target) is None:
            continue
        eff_path_steps = len(eff_paths[target]) - 1 if target in eff_paths else n_steps + 100
        if eff_path_steps <= n_steps:
            if eff_path_steps > str_dist[target]:
                for istep in range(str_dist[target], eff_path_steps):
                    try:
                        redo, _ = nx.single_source_dijkstra(G, source=source, target=target, cutoff=istep,
                                                            weight=_neg_log_weight)
                        impact[istep, target] = np.exp(-redo)
                    except nx.NetworkXNoPath:
                        pass
            impact[eff_path_steps:, target] = np.exp(-eff_dist[target])
    return impact[1:]


def predicted_impact(bn, source, t, graph="effective", path="at_most_t_edges", bias_iter=None, bound="mean",
                     threshold=0.0):
    """Predicted impact of flipping `source` on every node, for steps 1..t.

    Args:
        bn : a BooleanNetwork (or any object with structural_graph(),
            effective_graph(bound, threshold), biased_effective_graph(max_iter,
            bound, threshold) and Nnodes).
        source (int) : the perturbed node.
        t (int) : the number of steps.
        graph (str) : "effective" (EG), or "structural" / "interaction" (IG, the
            binary light cone; routed to :func:`interaction_graph_impact`, which
            ignores the other options).
        path (str) : how t limits the EG path, one of PATH_RULES (see the
            comment above them).
        bias_iter (int or None) : None for the plain EG; k >= 1 for the
            bias-aware EG after k rounds of bias propagation.
        bound (str) : edge-effectiveness bound passed to the EG ("mean" = edge
            effectiveness, "upper" = activity).
        threshold (float) : EG edges with weight <= threshold are dropped.

    Returns:
        (array) : shape (t, Nnodes); row s-1 is the prediction at step s.
    """
    if graph in INTERACTION_GRAPH_NAMES:
        return interaction_graph_impact(bn, source, t)
    if graph != "effective":
        raise ValueError(f"graph must be 'effective', 'structural' or 'interaction', not {graph!r}")
    if path not in PATH_RULES:
        raise ValueError(f"path must be one of {PATH_RULES}, not {path!r}")
    G = _effective_graph(bn, bias_iter, bound, threshold)
    if path == "inside_light_cone":
        return _inside_light_cone(G, bn.Nnodes, source, t)
    if path == "at_most_t_edges":
        return _at_most_t_edges(G, bn.Nnodes, source, t)
    if path == "global_strongest_path":
        return _global_strongest_path(G, bn.structural_graph(), bn.Nnodes, source, t)
    return _pnas2021(G, bn.structural_graph(), bn.Nnodes, source, t)
