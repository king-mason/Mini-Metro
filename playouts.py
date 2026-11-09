"""Playing finished designs out, optionally in worker processes."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

from simulator import MiniMetroSim, run_playout

CAP = 6000   # ticks; spawning has plateaued by ~3000


class Playouts:
    """Scores designs by ticks survived. With workers > 0 the entry script needs an
    `if __name__ == "__main__":` guard (processes are spawned)."""

    def __init__(self, cap: int = CAP, workers: int = 0):
        self.cap = cap
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None

    def ticks(self, sims: list[MiniMetroSim], seeds: list[int]) -> np.ndarray:
        """Ticks survived (capped), shape (len(sims), len(seeds))."""
        jobs = [(sim, seed) for sim in sims for seed in seeds]
        if self.workers > 0:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"))
            job_sims, job_seeds = zip(*jobs)
            results = self._pool.map(run_playout, job_sims, job_seeds, repeat(self.cap))
        else:
            results = (run_playout(sim, seed, self.cap) for sim, seed in jobs)
        return np.array([t for t, _ in results], dtype=float).reshape(len(sims), len(seeds))

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
