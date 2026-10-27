# Regression test for the step() fix, written for CANA's tests/ folder.
# Kept here for porting; tested in a scratch copy of upstream 394aeba (see notes.org).
"""BooleanNetwork.step must read each node's inputs in the order the node lists
them, the same order its look-up table (``outputs``) is written in."""
import copy
import itertools
import random

from cana.boolean_network import BooleanNetwork
from cana.datasets.bio import THALIANA

# x2 = x1 AND NOT x0, with x2's inputs listed as [x1, x0] (not network order).
UNSORTED_CNET = """.v 3

.l 1 x0
.l 2 x1
.l 3 x2

.n 1 1 1
1 1
0 0

.n 2 1 2
1 1
0 0

.n 3 2 2 1
11 0
10 1
01 0
00 0

.e End of file
"""


def test_step_reads_inputs_in_listed_order():
    bn = BooleanNetwork.from_string_cnet(UNSORTED_CNET, keep_constants=False)
    assert bn.nodes[2].inputs == [1, 0]
    for x0, x1, x2 in itertools.product("01", repeat=3):
        expected = "1" if (x1 == "1" and x0 == "0") else "0"
        assert bn.step(x0 + x1 + x2)[2] == expected


def _permute_node_inputs(logic, i, perm):
    """Relist node i's inputs in the order ``perm`` and reorder its LUT to match."""
    old_in, old_out = logic[i]["in"], logic[i]["out"]
    k = len(old_in)
    new_out = [None] * (2**k)
    for bits in itertools.product("01", repeat=k):
        old_bits = [None] * k
        for j, p in enumerate(perm):
            old_bits[p] = bits[j]
        new_out[int("".join(bits), 2)] = old_out[int("".join(old_bits), 2)]
    logic[i]["in"] = [old_in[p] for p in perm]
    logic[i]["out"] = new_out


def test_shuffled_input_order_gives_same_dynamics():
    """Listing a node's inputs in another order (with the LUT reordered to match)
    is the same Boolean function, so step() must give the same next state."""
    original = THALIANA()
    logic = copy.deepcopy(dict(original.logic))
    rng = random.Random(0)
    n_shuffled = 0
    for i, node_logic in logic.items():
        k = len(node_logic["in"])
        if k >= 2:
            perm = list(range(k))
            while perm == sorted(perm):
                rng.shuffle(perm)
            _permute_node_inputs(logic, i, perm)
            n_shuffled += 1
    assert n_shuffled > 0
    shuffled = BooleanNetwork.from_dict(
        logic, keep_constants=original.keep_constants, name="shuffled"
    )

    n = original.Nnodes
    for _ in range(2000):
        state = "".join(rng.choice("01") for _ in range(n))
        assert shuffled.step(state) == original.step(state)
