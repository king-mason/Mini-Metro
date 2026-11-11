"""Map generation. A MapSpec describes a kind of map: how many stations, how they are laid
out, how often a station has a rare type. The same spec and seed always give the same map;
the default spec gives the original 20-station maps."""

import math
from dataclasses import dataclass

from config import HEIGHT, NUM_STATIONS, STATION_RADIUS, WIDTH
from objects import Station, StationType
from point import Point

COMMON_TYPES = (StationType.CIRCLE, StationType.SQUARE, StationType.TRIANGLE)
COMMON_WEIGHTS = (0.5, 0.35, 0.15)
RARE_TYPES = (StationType.CROSS, StationType.DIAMOND, StationType.PENTAGON, StationType.RHOMBUS,
              StationType.DROPLET, StationType.STAR, StationType.LEMON)
GRID_SPACING = 4 * STATION_RADIUS


@dataclass(frozen=True)
class MapSpec:
    stations: int = NUM_STATIONS
    layout: str = "uniform"     # a key of LAYOUTS
    rare_every: int = 10        # every k-th station gets a rare type


def grid_points() -> list[Point]:
    """Where stations may go: a grid over the middle of the screen."""
    return [Point(x, y) for x in range(WIDTH // 10, 9 * WIDTH // 10 + 1, GRID_SPACING)
            for y in range(HEIGHT // 5, 4 * HEIGHT // 5 + 1, GRID_SPACING)]


# Layouts yield one free grid point at a time (removing it from `free`).

def uniform(free: list[Point], rng):
    """Anywhere on the grid."""
    while free:
        p = rng.choice(free)
        free.remove(p)
        yield p


def clusters(free: list[Point], rng, towns: int = 3, spread: float = 100.0):
    """Around a few town centers: each station takes the free point nearest a Gaussian draw
    (sd `spread` pixels) around a random town."""
    centers = [rng.choice(free) for _ in range(towns)]
    while free:
        cx, cy = rng.choice(centers)
        x, y = rng.gauss(cx, spread), rng.gauss(cy, spread)
        p = min(free, key=lambda q: (q.x - x) ** 2 + (q.y - y) ** 2)
        free.remove(p)
        yield p


def corridor(free: list[Point], rng, half_width: float = 1.5 * GRID_SPACING, max_tilt: float = 25.0):
    """Along a band through the middle of the screen, tilted up to max_tilt degrees; once the
    band is full, the nearest points outside it."""
    tilt = math.radians(rng.uniform(-max_tilt, max_tilt))
    cx, cy = rng.choice([q for q in free if abs(q.x - WIDTH / 2) < WIDTH / 4])
    nx, ny = -math.sin(tilt), math.cos(tilt)

    def off_axis(q):
        return abs((q.x - cx) * nx + (q.y - cy) * ny)

    while free:
        band = [q for q in free if off_axis(q) <= half_width]
        p = rng.choice(band) if band else min(free, key=off_axis)
        free.remove(p)
        yield p


LAYOUTS = {"uniform": uniform, "clusters": clusters, "corridor": corridor}


@dataclass(frozen=True)
class Level:
    """A kind of map for benchmarking and viewing: a station count, with the layouts taking
    turns by map seed."""
    stations: int
    layouts: tuple[str, ...] = tuple(LAYOUTS)

    def spec(self, map_seed: int) -> MapSpec:
        return MapSpec(self.stations, self.layouts[map_seed % len(self.layouts)])


LEVELS = {
    "classic": Level(NUM_STATIONS, ("uniform",)),   # the original maps
    "30": Level(30),     # bench levels: sized so strong designs rarely last to the cap
    "35": Level(35),
    "40": Level(40),
}


def generate(spec: MapSpec, rng) -> list[Station]:
    """Draw a map. Each station's type is drawn right after its position (the original
    maps' draw order); there are always at least two types."""
    free = grid_points()
    if spec.stations > len(free):
        raise ValueError(f"{spec.stations} stations don't fit on a grid of {len(free)} points")
    positions = LAYOUTS[spec.layout](free, rng)
    stations = []
    for k in range(1, spec.stations + 1):
        pos = next(positions)
        stype = rng.choice(RARE_TYPES) if k % spec.rare_every == 0 else rng.choices(COMMON_TYPES, COMMON_WEIGHTS)[0]
        stations.append(Station(stype, pos))
    types = {s.type for s in stations}
    if len(types) < 2 and len(stations) >= 2:
        stations[-1].type = next(t for t in COMMON_TYPES if t not in types)
    return stations
