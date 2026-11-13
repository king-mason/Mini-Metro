"""bench.py: the paired comparison."""

import math

import pytest
from hypothesis import given, strategies as st

from minimetro.bench import PolicySpec, compare, default_name, map_scores, share_of_gain


def results(scores: dict, seeds=(0, 1), cap=6000):
    return {"seeds": list(seeds), "cap": cap,
            "maps": [{"map": m, "ticks": [t] * len(seeds)} for m, t in scores.items()]}


ticks = st.floats(min_value=100, max_value=6000)


@given(st.dictionaries(st.integers(0, 100), ticks, min_size=2))
def test_identical_runs_compare_equal(scores):
    c = compare(results(scores), results(scores))
    assert c["ratio"] == pytest.approx(1.0) and c["wins"] == 0 and c["losses"] == 0


@given(st.dictionaries(st.integers(0, 100), ticks, min_size=2), st.floats(min_value=0.5, max_value=2.0))
def test_uniform_improvement_is_its_ratio(scores, factor):
    better = {m: t * factor for m, t in scores.items()}
    c = compare(results(better), results(scores))
    assert c["ratio"] == pytest.approx(factor, rel=1e-6)
    assert c["lo"] <= c["ratio"] <= c["hi"]


def test_only_shared_maps_are_compared():
    c = compare(results({1: 200, 2: 400}), results({2: 200, 3: 50}))
    assert c["n"] == 1 and c["ratio"] == pytest.approx(2.0) and math.isnan(c["lo"])


def test_unpaired_runs_are_refused():
    with pytest.raises(ValueError):
        compare(results({1: 1}, seeds=(0,)), results({1: 1}, seeds=(1,)))
    with pytest.raises(ValueError):
        compare(results({1: 1}, cap=10), results({1: 1}, cap=20))


def test_map_score_is_the_mean_over_seeds():
    r = {"seeds": [0, 1], "cap": 10, "maps": [{"map": 5, "ticks": [2, 4]}]}
    assert map_scores(r) == {("classic", 5): 3.0}


@given(st.dictionaries(st.integers(0, 100), ticks, min_size=2), st.floats(min_value=1.1, max_value=3.0),
       st.floats(min_value=0.0, max_value=1.0))
def test_share_of_gain(scores, teacher_factor, fraction):
    teacher = {m: t * teacher_factor for m, t in scores.items()}
    student = {m: t * teacher_factor ** fraction for m, t in scores.items()}
    g = share_of_gain(results(student), results(teacher), results(scores))
    assert g["share"] == pytest.approx(fraction, abs=1e-6) and g["n"] == len(scores)


def test_partial_runs_get_their_own_name():
    spec = PolicySpec("heuristic", polish_rounds=30)
    assert default_name(spec) == "heuristic_polish30"
    assert default_name(spec, maps=10) == "heuristic_polish30_m10"
