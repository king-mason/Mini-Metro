"""maps.py: every kind of map is reproducible and well formed."""

import random

import pytest
from hypothesis import given, strategies as st

from bench import suite
from maps import LAYOUTS, LEVELS, MapSpec, generate, grid_points
from simulator import MiniMetroSim

specs = st.builds(MapSpec, st.integers(2, 40), st.sampled_from(list(LAYOUTS)), st.integers(1, 12))


@given(specs, st.integers(0, 10 ** 6))
def test_maps_are_reproducible_and_well_formed(spec, seed):
    a, b = generate(spec, random.Random(seed)), generate(spec, random.Random(seed))
    assert [(s.x, s.y, s.type) for s in a] == [(s.x, s.y, s.type) for s in b]
    assert len(a) == spec.stations
    assert len({(s.x, s.y) for s in a}) == spec.stations     # distinct grid points
    assert len({s.type for s in a}) >= 2                      # passengers always have a target


def test_the_default_is_the_classic_map():
    assert MiniMetroSim(7).stations[0].pos == MiniMetroSim(7, map_spec=LEVELS["classic"].spec(7)).stations[0].pos
    assert all(LEVELS["classic"].spec(m) == MapSpec() for m in range(10))


def test_levels_take_turns_with_layouts():
    level = LEVELS["30"]
    assert [level.spec(m).layout for m in range(3)] == list(LAYOUTS)
    assert {level.spec(m).stations for m in range(9)} == {30}


def test_too_many_stations_are_refused():
    with pytest.raises(ValueError):
        generate(MapSpec(len(grid_points()) + 1), random.Random(0))


def test_bench_levels_use_separate_maps():
    maps = suite("levels", 50)
    assert len({m.seed for m in maps}) == len(maps)
    assert len(suite("classic")) == 100 and all(m.spec == MapSpec() for m in suite("classic"))
