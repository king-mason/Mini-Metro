"""Game settings, read once from config.yaml next to this file.

Import the named constants below rather than reading the YAML elsewhere. Times in
config.yaml are at time_scale 1; the constants here are already scaled.
"""

import os

import yaml

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")) as _f:
    CONFIG = yaml.safe_load(_f)

TIME_SCALE = CONFIG.get("time_scale", 1)

WIDTH, HEIGHT = CONFIG["resolution"]
MAX_DIST = (WIDTH ** 2 + HEIGHT ** 2) ** 0.5        # screen diagonal, for normalizing distances

NUM_STATIONS = CONFIG["stations"]["count"]
STATION_RADIUS = CONFIG["stations"]["radius"]
STATION_CAPACITY = CONFIG["stations"]["capacity"]   # waiting passengers before a station is overcrowded
# The game is lost once this many station-ticks over capacity have piled up (all stations, never reset).
MAX_OVERCROWD_TICKS = CONFIG["stations"]["max_overcrowd_ticks"] // TIME_SCALE

MAX_LINES = CONFIG["lines"]["max"]
MAX_TRAINS = CONFIG["trains"]["total"]
MAX_TRAINS_PER_LINE = CONFIG["trains"]["max_per_line"]
TRAIN_CAPACITY = CONFIG["trains"]["capacity"]
TRAIN_SPEED = CONFIG["trains"]["speed"] * TIME_SCALE  # pixels per tick

PASSENGERS = CONFIG["passengers"]                   # spawn schedule (simulator.py scales it)
PRIOR_WEIGHTS = CONFIG["rewards"]                   # structural prior coefficients (heuristic.py)
SCREENSHOT_DIR = CONFIG["paths"]["screenshots"]
RESULTS_DIR = CONFIG["paths"]["results"]
CHECKPOINT_DIR = CONFIG["paths"]["checkpoints"]
DATA_DIR = CONFIG["paths"]["data"]
DEFAULT_CHECKPOINT = CONFIG["paths"]["default_checkpoint"]
