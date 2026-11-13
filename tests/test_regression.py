"""Golden designs and play-outs: any change to the game or the heuristic shows up here.

If a change is intended, regenerate and commit the new expectations with
    python -m tests.test_regression
"""

import json
import os

import pytest

from minimetro.design.search import spec_of
from minimetro.game.rules import ActionType
from minimetro.game.simulator import run_playout
from tests.helpers import heuristic_design

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "golden.json")
MAPS = [10_000, 10_017, 10_042, 10_083]
SEEDS = [0, 1]
CAP = 3000


def snapshot(map_seed: int) -> dict:
    result = heuristic_design(map_seed)
    spec = spec_of(result.sim)
    return {
        "actions": [[ActionType(a[0]).name, *a[1:]] for a in result.actions],
        "lines": spec.lines,
        "trains": spec.trains,
        "playouts": [list(run_playout(result.sim, seed, CAP)) for seed in SEEDS],
    }


@pytest.mark.parametrize("map_seed", MAPS)
def test_matches_golden(map_seed):
    with open(GOLDEN) as f:
        expected = json.load(f)[str(map_seed)]
    assert snapshot(map_seed) == expected


if __name__ == "__main__":
    with open(GOLDEN, "w") as f:
        json.dump({str(m): snapshot(m) for m in MAPS}, f, indent=1)
    print(f"wrote {GOLDEN}")
