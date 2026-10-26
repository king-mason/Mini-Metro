"""Shared test setup.

The game modules read config.yaml from the working directory at import time, so the tests
run from the repository root wherever pytest is started. Hypothesis examples that play
games are slow, so the default profile has no deadline and fewer examples; run
`pytest --hypothesis-profile=thorough` for a longer search.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # visualizer tests draw off screen
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from hypothesis import HealthCheck, settings  # noqa: E402

settings.register_profile("default", max_examples=25, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("thorough", max_examples=300, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("default")
